#!/usr/bin/env python3
# training/train_agent1.py

import os
import torch
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV = os.path.join(BASE, "../dataset_preparation/dataset.csv")
OUTPUT_DIR  = os.path.join(BASE, "agent1_model")

# ─── Load Dataset ──────────────────────────────────────────────────────────────
# expects a CSV with columns "input" and "output"
raw_datasets = load_dataset("csv", data_files=DATASET_CSV)

# split into train/test
splits = raw_datasets["train"].train_test_split(test_size=0.1)
train_ds = splits["train"]
eval_ds  = splits["test"]

# ─── Load Model & Tokenizer ────────────────────────────────────────────────────
model_name = "t5-small"
tokenizer  = T5Tokenizer.from_pretrained(model_name)
model      = T5ForConditionalGeneration.from_pretrained(model_name)

# ─── Preprocess Function ───────────────────────────────────────────────────────
max_input_length  = 256
max_output_length = 64

def preprocess(example):
    # Tokenize inputs
    model_inputs = tokenizer(
        example["input"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["output"],
            max_length=max_output_length,
            truncation=True,
            padding="max_length",
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# map preprocess over datasets
train_dataset = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
eval_dataset  = eval_ds.map(preprocess, batched=True, remove_columns=eval_ds.column_names)

# ─── Data Collator ─────────────────────────────────────────────────────────────
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ─── Training Arguments ────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    #evaluation_strategy="epoch",
    #save_strategy="epoch",
    learning_rate=5e-5,
    logging_steps=50,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
)

# ─── Initialize Trainer ────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ─── Train & Save ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model and tokenizer saved to {OUTPUT_DIR}")
