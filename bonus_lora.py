import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
import torch

# ======================
# LOAD DATA
# ======================

train_df = pd.read_csv("train.csv")

train_df = train_df.rename(columns={"score": "labels"})

# ubah label jadi integer
train_df["labels"] = train_df["labels"].astype(int)

dataset = Dataset.from_pandas(
    train_df[["full_text", "labels"]]
)

# ======================
# TOKENIZER
# ======================

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["full_text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

dataset = dataset.map(tokenize)

# ======================
# MODEL
# ======================

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=7
)

# ======================
# LORA CONFIG
# ======================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)

# ======================
# TRAINING
# ======================

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

print("Training selesai!")