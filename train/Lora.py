from utils.logger import setup_logger
from pathlib import Path
from utils.config import Config
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

logger = setup_logger(__name__)


class Lora:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        output_dir: str = "lora_output",
    ):
        self.data_dir = Config.REFACTOR_DIR
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.tokenizer = None
        self.model = None

    def load_data(self):
        logger.info(f"正在加载数据: {self.data_dir}")
        data = []
        for f in sorted(self.data_dir.glob("*.json")):
            for item in json.loads(f.read_text(encoding="utf-8")):
                data.append({
                    "scene": item["scene"],
                    "input": item["input"],
                    "output": item["output"],
                })
        logger.info(f"共加载 {len(data)} 条训练数据")
        return data

    def _format_chat(self, example):
        return {
            "text": self.tokenizer.apply_chat_template([
                {"role": "system", "content": example["scene"]},
                {"role": "user", "content": example["input"]},
                {"role": "assistant", "content": example["output"]},
            ], tokenize=False)
        }

    def _prepare_dataset(self, data, test_size=0.1, seed=42):
        dataset = Dataset.from_list([self._format_chat(d) for d in data])
        return dataset.train_test_split(test_size=test_size, seed=seed).values()

    def _load_model_and_tokenizer(self):
        logger.info(f"加载模型: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )

    def train(
        self,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        batch_size=4,
        grad_accum=2,
        epochs=3,
        lr=2e-4,
        logging_steps=5,
        eval_steps=20,
        test_size=0.1,
        seed=42,
    ):
        data = self.load_data()
        self._load_model_and_tokenizer()
        train_set, eval_set = self._prepare_dataset(data, test_size, seed)

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules="all-linear",
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        trainer = SFTTrainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=str(self.output_dir),
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                num_train_epochs=epochs,
                logging_steps=logging_steps,
                eval_strategy="steps",
                eval_steps=eval_steps,
                save_strategy="epoch",
                learning_rate=lr,
                bf16=True,
                remove_unused_columns=False,
                report_to=["tensorboard"],
            ),
            train_dataset=train_set,
            eval_dataset=eval_set,
            tokenizer=self.tokenizer,
            peft_config=peft_config,
            dataset_text_field="text",
        )

        logger.info("开始训练...")
        trainer.train()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))
        logger.info(f"LoRA 权重已保存到 {self.output_dir}")


if __name__ == "__main__":
    Lora().train()