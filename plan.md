# PersonaForge 开发计划

## 当前状态 (v0.5)

- ✅ 爬虫模块：LLM驱动生成爬虫代码，爬取小说章节
- ✅ 对话提取：从章节中提取目标角色台词
- ✅ 对话筛选：保留体现角色性格的高质量台词
- ✅ 场景重构：将台词反向构建为 (scene, input, output) 训练对
- ⬜ RAG 知识库：嵌入模块已完成，未接入流水线
- ⬜ 模型微调：数据聚合 + 微调代码待实现
- ⬜ 推理接口：完全未实现

---

## 架构总览

```
用户输入：目录页URL + 角色名 + 别称
       │
       ▼
┌─────────────────────┐
│  Step 0: 数据爬取    │  data/第N章.txt
│  (crawler_agent)     │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Step 1: 台词提取    │  extract/第N章.json
│  (extract.md)        │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Step 2: 台词筛选    │  filter/第N章.json
│  (filter.md)         │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Step 3: 场景重构    │  refactor/第N章.json
│  (refactor.md)       │
└─────────┬───────────┘
          ▼
┌─────────────────────┐      ╔══════════════════╗
│  Step 4: 数据聚合    │──────║  待实现 (第4步)  ║
│  refactor→JSONL     │      ╚══════════════════╝
└─────────┬───────────┘
          ▼
┌─────────────────────┐      ╔══════════════════╗
│  Step 5: LoRA 微调   │──────║  待实现 (第5步)  ║
│  (train_model.py)    │      ╚══════════════════╝
└─────────┬───────────┘
          ▼
┌─────────────────────┐      ╔══════════════════╗
│  Step 6: 角色提示词  │──────║  待实现 (第6步)  ║
│  生成 system prompt  │      ╚══════════════════╝
└─────────┬───────────┘
          ▼
┌─────────────────────┐      ╔══════════════════╗
│  Step 7: 推理对话    │──────║  待实现 (第7步)  ║
│  RAG + LoRA 推理    │      ╚══════════════════╝
└─────────────────────┘
```

---

## Scene 字段处理策略分析（第4-5步核心设计）

### 1. 数据特征分析

`refactor/*.json` 中每条样本格式：
```json
{
  "scene": "众猴面对瀑布飞泉，畏惧不前，纷纷后退，只有孙悟空挺身而出。",
  "input": "大王，这瀑布深不可测，谁敢进去寻源头？我们可不敢！",
  "output": "我进去！我进去！"
}
```

**关键发现：** 数据是按 `(output, scene, input)` 方式构造的——同一句经典台词（output）配了 2 个不同的 `(scene, input)` 对，让模型学习在不同情境下说出同一台词的能力。共约 53 条样本，数据量较小。

### 2. 三种处理方案对比

| 方案 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| **A: 丢弃 scene** | 仅用 `input → output` 训练 | 数据最简，推理时无需额外输入 | 模型失去场景理解能力。同一个 `input` 对应不同 `scene` 时，模型无法区分该用哪个 `output` |
| **B: scene 作为 system prompt** | 按 `system:{scene} user:{input} assistant:{output}` 格式 | 模型能感知场景上下文；推理时可自由提供新场景；与主流 chat template 天然匹配 | 推理时需要构造 scene |
| **C: scene 拼入 input** | 格式为 `{scene}\n{input} → {output}` | 简单直接 | scene 和 user input 界限模糊，推理时格式必须严格一致 |

### 3. 推荐方案：B（scene 作为 system prompt）

**核心理由：**

1. **语义对齐**：`scene` 描述的是"当前发生的剧情/情景"，等价于角色扮演中的"场景设定"，与 chat model 的 `system` 角色天然对应
2. **数据结构的必然选择**：数据中同一 `output` 对应不同 `(scene, input)`，说明模型**需要 scene 来消歧**——仅靠 input 无法决定应输出哪句台词
3. **推理灵活**：上线时用户只需提供 `{"scene": "...", "input": "..."}`，即可在任意新场景下与角色对话
4. **生态兼容**：Qwen、LLaMA 等主流模型的 tokenizer 原生支持 `apply_chat_template()` 处理 system/user/assistant 格式

**训练时的 chat template 格式（以 Qwen 为例）：**
```
<|im_start|>system
{scene}<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

**推理时的输入格式：**
```
<|im_start|>system
{scene}<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant
```

---

## 微调代码设计（第4-5步）

### 数据量评估与应对策略

- 当前数据量：约 53 条（3个章节）
- **问题**：数据量偏少，直接训练容易过拟合
- **应对**：
  - 使用 QLoRA（4-bit）减少过拟合风险
  - 增大 epoch（15-25），配合早停
  - 使用小模型（1.5B-3B），参数量少更适配小数据
  - 加入数据增强：对 `input` 做同义替换扩充，或用 LLM 生成更多变体
  - 数据扩充建议：用 DeepSeek API 基于现有 output 为各章节生成更多 scene+input 组合

### Step 4: 数据聚合器 `pipeline/aggregator.py`

将 `refactor/*.json` 中的三元组统一聚合为 JSONL 格式，每行包含完整的三字段，供训练脚本读取。

```python
"""
pipeline/aggregator.py — 将 refactor/*.json 聚合为训练用 JSONL
"""
import json
from pathlib import Path


class Aggregator:
    def __init__(self, refactor_dir: str, output_dir: str):
        self.refactor_dir = Path(refactor_dir)
        self.output_dir = Path(output_dir)

    def aggregate(self) -> int:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        all_records = []

        for json_file in sorted(self.refactor_dir.glob("*.json")):
            with open(json_file, "r", encoding="utf-8") as f:
                records = json.load(f)
            chapter_name = json_file.stem
            output_file = self.output_dir / f"{chapter_name}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for r in records:
                    line = json.dumps(r, ensure_ascii=False)
                    f.write(line + "\n")
            all_records.extend(records)

        all_path = self.output_dir / "all.jsonl"
        with open(all_path, "w", encoding="utf-8") as f:
            for r in all_records:
                line = json.dumps(r, ensure_ascii=False)
                f.write(line + "\n")

        return len(all_records)
```

### Step 5: 微调脚本 `train_model.py`

**设计原则：**
- 将 scene 通过 `tokenizer.apply_chat_template()` 格式化为 system prompt
- 使用 `SFTTrainer`（trl）简化训练循环
- 只对 assistant 部分的 token 计算 loss（通过 `DataCollatorForCompletionOnlyLM`）
- 支持 ModelScope 下载（国内网络友好）
- 支持数据增强模式

```python
"""
train_model.py — QLoRA 角色扮演微调
用法：
  python train_model.py \
    --data_path data/Reconstructed/all.jsonl \
    --model_id Qwen/Qwen2.5-3B-Instruct \
    --output_dir outputs/sun-wukong-lora
"""
import json
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_chat(record: dict) -> list[dict]:
    return [
        {"role": "system", "content": record["scene"]},
        {"role": "user", "content": record["input"]},
        {"role": "assistant", "content": record["output"]},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/lora")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    args = parser.parse_args()

    # 1. 加载数据
    raw_records = load_jsonl(args.data_path)
    dataset = Dataset.from_list(raw_records)

    # 2. 加载 tokenizer 和 model
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    try:
        from modelscope import snapshot_download
        model_path = snapshot_download(args.model_id)
    except ImportError:
        model_path = args.model_id

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA 配置
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 4. 数据预处理
    def tokenize_fn(examples):
        texts = []
        for scene, inp, out in zip(examples["scene"], examples["input"], examples["output"]):
            messages = format_chat({"scene": scene, "input": inp, "output": out})
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return tokenizer(
            texts,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            return_tensors=None,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    # 只对 assistant 回复部分计算 loss
    response_template = "<|im_start|>assistant"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=tokenizer
    )

    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="tensorboard",
    )

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # 7. 训练
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"模型已保存至: {args.output_dir}")

    # 可选：merge 并保存完整模型
    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    merged = PeftModel.from_pretrained(base_model, args.output_dir).merge_and_unload()
    merged.save_pretrained(f"{args.output_dir}/merged")
    tokenizer.save_pretrained(f"{args.output_dir}/merged")
    print(f"合并后的完整模型已保存至: {args.output_dir}/merged")


if __name__ == "__main__":
    main()
```

### 推理接口 `inference/lora_chat.py`

```python
"""
inference/lora_chat.py — LoRA 角色推理
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class LoraChat:
    def __init__(self, model_id: str, lora_path: str, device: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            lora_path, trust_remote_code=True
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, lora_path)
        self.model.eval()

    def chat(self, scene: str, user_input: str, max_new_tokens: int = 256) -> str:
        messages = [
            {"role": "system", "content": scene},
            {"role": "user", "content": user_input},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return response.strip()
```

---

## 待实现步骤详解

### 第4步：数据聚合

**文件：** `pipeline/aggregator.py`

**目标：** 将所有 `refactor/*.json` 中的训练对合并，生成 `data/Reconstructed/*.jsonl`

**产出物：**
- `data/Reconstructed/第1章.jsonl`
- `data/Reconstructed/第2章.jsonl`
- `data/Reconstructed/第3章.jsonl`
- `data/Reconstructed/all.jsonl`（汇总文件）

### 第5步：LoRA 微调

**文件：** `train_model.py`

**核心要点：**
1. 数据直接读 JSONL 格式：每行 `{"scene":..., "input":..., "output":...}`
2. Scene → system prompt：通过 `tokenizer.apply_chat_template()` 格式化
3. Loss masking：使用 `DataCollatorForCompletionOnlyLM` 只对 assistant 回复计算 loss
4. ModelScope 优先：支持国内模型下载

**调用方式：**
```bash
# 先聚合
python -c "from pipeline.aggregator import Aggregator; Aggregator('refactor', 'data/Reconstructed').aggregate()"

# 再微调
python train_model.py \
  --data_path data/Reconstructed/all.jsonl \
  --model_id Qwen/Qwen2.5-3B-Instruct \
  --output_dir outputs/sun-wukong-lora \
  --lora_r 16 \
  --num_epochs 20
```

### 第6步：角色提示词生成

**文件：** `prompt_generator.py`

**目标：** 使用 LLM 分析角色的所有台词/场景，生成高质量的系统提示词（system prompt），用于推理时引导模型以角色身份回复。

**产出物：**
- `character_prompts/{role}_prompt.md` — 人类可读的角色设定
- `character_prompts/{role}_system.txt` — 可直接用于推理的 system prompt

### 第7步：推理对话接口

| 子模块 | 文件 | 说明 |
|--------|------|------|
| 7a. LoRA 推理 | `inference/lora_chat.py` | 加载 base model + LoRA adapter，按 chat template 推理 |
| 7b. RAG 增强推理 | `inference/rag_chat.py` | 用户输入 → RAG 检索原文 → 拼接上下文 → 注入模型生成回复 |
| 7c. CLI 交互 | `inference/cli_chat.py` | 命令行交互式对话 |
| 7c. Web 交互 | `inference/gradio_chat.py` | Gradio Web UI |

### 第8步：主入口与配置化

**文件：** `main.py` + `config.yaml`

**目标：** 一键运行从爬取到推理的完整流水线

```yaml
# config.yaml
character:
  name: "孙悟空"
  aliases: "行者、齐天大圣、美猴王"
  source_url: "https://www.guoxuemao.com/shu/xiyouji.html"

pipeline:
  extract: true
  filter: true
  refactor: true
  aggregate: true
  train: true
  generate_prompt: true

training:
  model_id: "Qwen/Qwen2.5-3B-Instruct"
  lora_r: 16
  lora_alpha: 32
  num_epochs: 20
  learning_rate: 2e-4
  batch_size: 4
  use_4bit: true

rag:
  enabled: true
  chunk_mode: "paragraph"
  embedding_model: "Qwen/Qwen2.5-Embedding-0.6B"
  top_k: 30
```

---

## 项目结构（目标状态）

```
personaforge/
├── main.py                     # 主入口，一键运行
├── config.yaml                 # 配置文件
├── plan.md                     # 开发计划
├── requirements.txt
├── pyproject.toml
│
├── pipeline/
│   ├── pipeline.py             # 现有流水线编排
│   ├── aggregator.py           # [新增] 数据聚合
│   └── __init__.py
│
├── crawler/
│   ├── crawler_agent.py        # 现有爬虫
│   └── __init__.py
│
├── RAG/
│   ├── embedding.py            # 现有嵌入
│   ├── retrieve.py             # 现有检索
│   └── __init__.py
│
├── inference/                  # [新增] 推理模块
│   ├── lora_chat.py            # LoRA 推理
│   ├── rag_chat.py             # RAG 增强推理
│   ├── cli_chat.py             # 命令行交互
│   ├── gradio_chat.py          # Web 交互
│   └── __init__.py
│
├── prompt_generator.py         # [新增] 角色提示词生成
├── train_model.py              # [重写] 微调脚本
│
├── utils/
│   ├── config.py
│   ├── logger.py
│   ├── tools.py
│   └── prompts/
│
├── data/                       # 原始章节
│   └── Reconstructed/          # [新增] 聚合后的训练数据
├── extract/                    # 提取结果
├── filter/                     # 筛选结果
├── refactor/                   # 重构结果
├── outputs/                    # 训练产出
├── character_prompts/          # [新增] 角色提示词
└── model_sources/              # [新增] 下载的 base model
```

---

## 数据流总结

```
步骤          输入                    输出                    关键文件
─────        ────                   ────                    ────────
0.爬取       目录页URL              data/第N章.txt           crawler_agent.py
1.提取       data/第N章.txt         extract/第N章.json       pipeline.py + extract.md
2.筛选       extract/第N章.json     filter/第N章.json        pipeline.py + filter.md
3.重构       filter/第N章.json      refactor/第N章.json      pipeline.py + refactor.md
4.聚合       refactor/*.json        data/Reconstructed/*.jsonl  aggregator.py
5.微调       data/Reconstructed/*.jsonl  outputs/适配器+合并模型  train_model.py
6.提示词     refactor/*.json        character_prompts/       prompt_generator.py
7.推理       LoRA适配器 + RAG       对话回复                 inference/*.py
```

---

## 开发优先级

| 优先级 | 步骤 | 估计工时 | 说明 |
|--------|------|----------|------|
| P0 | 第4步：数据聚合 | 0.5天 | 阻塞微调的关键环节 |
| P0 | 第5步：模型微调 | 1天 | 核心训练流程 |
| P0 | 第7a步：LoRA 推理 | 0.5天 | 微调后验证效果 |
| P1 | 第6步：提示词生成 | 1天 | 提升角色一致性 |
| P1 | 第7b步：RAG 推理 | 1天 | 增强角色知识准确度 |
| P1 | 数据扩充 | 1天 | 用小模型生成更多 scene+input 对 |
| P2 | 第8步：main.py + config.yaml | 1天 | 提升工程化体验 |
| P2 | 第7c步：Gradio 界面 | 0.5天 | 便捷测试交互 |
| P3 | 多角色支持 | 2天 | 扩展项目能力 |
