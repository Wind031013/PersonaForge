import json
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import random

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

from modelscope import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f"logs/finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def setup_logging(log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger.info(f"日志初始化完成。日志文件: {log_file}")


def load_data_from_directory(data_dir: str) -> List[Dict[str, Any]]:
    """加载指定目录下的所有jsonl文件"""
    logger.info(f"从目录加载数据: {data_dir}")

    data = []
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"数据目录 {data_dir} 不存在")
        raise FileNotFoundError(f"目录未找到: {data_dir}")

    jsonl_files = list(data_path.glob("*.jsonl"))
    logger.info(f"找到 {len(jsonl_files)} 个 JSONL 文件")

    for file_path in sorted(jsonl_files):
        logger.info(f"正在处理文件: {file_path.name}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        if "input" in item and "output" in item:
                            data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"解析 {file_path.name} 第 {line_num} 行失败: {e}"
                        )
            logger.info(f"从 {file_path.name} 加载了数据项")
        except Exception as e:
            logger.error(f"读取文件 {file_path.name} 时出错: {e}")

    logger.info(f"总共加载 {len(data)} 条数据样本")
    return data


def format_data_for_training(
    data: List[Dict[str, Any]], model_name: str = "Qwen"
) -> List[Dict[str, str]]:
    """将数据格式化为训练格式"""
    logger.info("正在格式化训练数据")

    formatted_data = []

    for idx, item in enumerate(data):
        input_text = item.get("input", "")
        output_text = item.get("output", "")

        if model_name == "Qwen":
            instruction = input_text
            response = output_text

            formatted_text = (
                f"<|im_start|>system\n你是一个角色扮演助手，请根据用户的输入，给出符合角色特点、语气口头禅的回复。<|im_end|>\n"
                f"<|im_start|>user\n{instruction}<|im_end|>\n"
                f"<|im_start|>assistant\n{response}<|im_end|>"
            )

            formatted_data.append(
                {"text": formatted_text, "scene": item.get("scene", "")}
            )

        if (idx + 1) % 100 == 0:
            logger.info(f"已格式化 {idx + 1}/{len(data)} 条样本")

    logger.info(f"总共格式化 {len(formatted_data)} 条样本")
    return formatted_data


def prepare_dataset(formatted_data: List[Dict[str, str]], train_split: float = 0.95):
    """准备训练数据集"""
    logger.info("正在准备数据集")

    random.shuffle(formatted_data)

    split_idx = int(len(formatted_data) * train_split)
    train_data = formatted_data[:split_idx]
    eval_data = formatted_data[split_idx:]

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    logger.info(f"训练样本数: {len(train_dataset)}, 评估样本数: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def download_model_from_modelscope(
    model_id: str = "Qwen/Qwen2.5-3B-Instruct", cache_dir: str = "./models"
):
    """从modelscope下载模型"""
    logger.info(f"正在从 ModelScope 下载模型: {model_id}")

    os.makedirs(cache_dir, exist_ok=True)

    model_dir = snapshot_download(model_id, cache_dir=cache_dir, revision="master")

    logger.info(f"模型已下载到: {model_dir}")
    return model_dir


def load_model_and_tokenizer(
    model_dir: str, max_seq_length: int = 2048, use_4bit: bool = False
):
    """加载模型和tokenizer"""
    logger.info("正在加载模型和tokenizer")
    logger.info(f"最大序列长度: {max_seq_length}")
    logger.info(f"使用4bit量化: {use_4bit}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU设备: {torch.cuda.get_device_name(0)}")

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16 if not use_4bit else None,
        trust_remote_code=True,
    )

    logger.info(f"模型设备: {model.device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("模型和tokenizer加载成功")
    return model, tokenizer


def configure_model_for_training(model, lora_r: int = 32, lora_alpha: int = 32):
    """配置模型用于训练"""
    logger.info("正在配置模型用于训练")

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable()
    logger.info("已启用梯度检查点以节省显存")

    logger.info("模型已配置为PEFT训练模式")
    return model


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    use_4bit: bool = False,
):
    """创建训练器"""
    logger.info("正在创建训练器")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to=["tensorboard"],
        run_name=f"qwen_roleplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=lambda x: x["text"],
        processing_class=tokenizer,
    )

    logger.info("训练器创建完成")
    return trainer


def train_model(trainer, resume_from_checkpoint: str = None):
    """训练模型"""
    logger.info("开始训练模型")

    if resume_from_checkpoint:
        logger.info(f"从检查点恢复训练: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    logger.info("训练完成")

    metrics = trainer.state.log_history[-1]
    logger.info(f"最终训练指标: {metrics}")

    return trainer


def save_model_and_tokenizer(model, tokenizer, output_dir: str):
    """保存模型和tokenizer"""
    logger.info(f"正在保存模型和tokenizer到 {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("模型和tokenizer保存成功")


def merge_and_save_model(model, tokenizer, output_dir: str):
    """合并并保存完整模型"""
    logger.info(f"正在合并并保存模型到 {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    config_path = Path(output_dir) / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            model_config = json.load(f)

        keys_to_remove = ["quantization_config", "_load_in_4bit", "_load_in_8bit"]
        modified = False
        for key in keys_to_remove:
            if key in model_config:
                del model_config[key]
                modified = True

        if modified:
            with open(config_path, "w") as f:
                json.dump(model_config, f, indent=2, ensure_ascii=False)
            logger.info("已清理合并模型中的量化配置")

    logger.info("合并后的模型保存成功")


def main():
    setup_logging()

    logger.info("=" * 80)
    logger.info("开始角色扮演模型微调")
    logger.info("=" * 80)

    config = {
        "data_dir": "data/Reconstructed",
        "model_id": "Qwen/Qwen3.5-4B",
        "cache_dir": "./models",
        "output_dir": "./outputs/qwen3_roleplay",
        "merged_output_dir": "./outputs/qwen3_roleplay_merged",
        "max_seq_length": 4096,
        "use_4bit": False,
        "train_split": 0.95,
        "num_train_epochs": 5,
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-4,
        "lora_r": 64,
        "lora_alpha": 64,
    }

    logger.info(f"配置参数: {json.dumps(config, indent=2, ensure_ascii=False)}")

    try:
        logger.info("\n" + "=" * 80)
        logger.info("第1步: 加载训练数据")
        logger.info("=" * 80)

        raw_data = load_data_from_directory(config["data_dir"])

        if len(raw_data) == 0:
            logger.error("未加载数据。请检查数据目录。")
            return

        logger.info("\n" + "=" * 80)
        logger.info("第2步: 格式化数据")
        logger.info("=" * 80)

        formatted_data = format_data_for_training(raw_data, model_name="Qwen")

        logger.info("\n" + "=" * 80)
        logger.info("第3步: 准备数据集")
        logger.info("=" * 80)

        train_dataset, eval_dataset = prepare_dataset(
            formatted_data, train_split=config["train_split"]
        )

        logger.info("\n" + "=" * 80)
        logger.info("第4步: 从 ModelScope 下载模型")
        logger.info("=" * 80)

        model_dir = download_model_from_modelscope(
            model_id=config["model_id"], cache_dir=config["cache_dir"]
        )

        logger.info("\n" + "=" * 80)
        logger.info("第5步: 加载模型和tokenizer")
        logger.info("=" * 80)

        model, tokenizer = load_model_and_tokenizer(
            model_dir,
            max_seq_length=config["max_seq_length"],
            use_4bit=config["use_4bit"],
        )

        logger.info("\n" + "=" * 80)
        logger.info("第6步: 配置模型用于训练")
        logger.info("=" * 80)

        model = configure_model_for_training(
            model, lora_r=config["lora_r"], lora_alpha=config["lora_alpha"]
        )

        logger.info("\n" + "=" * 80)
        logger.info("第7步: 创建训练器")
        logger.info("=" * 80)

        trainer = create_trainer(
            model,
            tokenizer,
            train_dataset,
            eval_dataset,
            config["output_dir"],
            use_4bit=config["use_4bit"],
        )

        logger.info("\n" + "=" * 80)
        logger.info("第8步: 训练模型")
        logger.info("=" * 80)

        trainer = train_model(trainer)

        logger.info("\n" + "=" * 80)
        logger.info("第9步: 保存模型 (LoRA)")
        logger.info("=" * 80)

        save_model_and_tokenizer(model, tokenizer, config["output_dir"])

        logger.info("\n" + "=" * 80)
        logger.info("第10步: 合并并保存完整模型")
        logger.info("=" * 80)

        merge_and_save_model(model, tokenizer, config["merged_output_dir"])

        logger.info("\n" + "=" * 80)
        logger.info("微调完成！")
        logger.info(f"LoRA模型保存位置: {config['output_dir']}")
        logger.info(f"合并模型保存位置: {config['merged_output_dir']}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"微调过程中出错: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
