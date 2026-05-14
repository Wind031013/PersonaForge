import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

DATA_DIR = Path("refactor")
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "lora_output"

def load_data():
    records = []
    for f in sorted(DATA_DIR.glob("*.json")):
        for item in json.loads(f.read_text(encoding="utf-8")):
            records.append({
                "scene": item["scene"],
                "input": item["input"],
                "output": item["output"],
            })
    return records

def format_chat(example):
    return {
        "text": tokenizer.apply_chat_template([
            {"role": "system", "content": example["scene"]},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ], tokenize=False)
    }

data = load_data()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "right"

dataset = Dataset.from_list([format_chat(d) for d in data])
train_set, eval_set = dataset.train_test_split(test_size=0.1, seed=42).values()

lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules="all-linear",
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map="auto",
)

trainer = SFTTrainer(
    model=model,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        remove_unused_columns=False,
    ),
    train_dataset=train_set,
    eval_dataset=eval_set,
    tokenizer=tokenizer,
    peft_config=lora_config,
    dataset_text_field="text",
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
