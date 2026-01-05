# train_bert_ipatok.py
import os
import pandas as pd
from datasets import Dataset
from transformers import (
    BertConfig, BertForMaskedLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from pathlib import Path
from tokenizer import IpatokHFTokenizer 

DATA_FILE = "wikipron_combined.tsv"  
MODEL_DIR = "bert-ipa-model"
MAX_LEN = 64

def load_ipa_data():
    lines = []
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                _, ipa = parts
                lines.append({"text": ipa})
    return Dataset.from_list(lines)

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    dataset = load_ipa_data()
    print(f"\nLoaded {len(dataset)} IPA sequences.")

    tokenizer = IpatokHFTokenizer.from_pretrained("idk_bert/ipatok_tokenizer")
    sample_text = dataset[0]["text"]
    print("\n Tokenization test")
    print("Original:", sample_text)
    toks = tokenizer.tokenize(sample_text)
    print("Tokens:", toks)
    print("IDs:", tokenizer.convert_tokens_to_ids(toks))
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )
    tokenized_ds = dataset.map(tokenize, batched=True)
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        max_position_embeddings=MAX_LEN,
    )
    model = BertForMaskedLM(config)
    args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=128,
        num_train_epochs=2,
        logging_dir=None,
        report_to=[], 
        logging_steps=1000,
        save_total_limit=2,
        overwrite_output_dir=True,
        no_cuda=True
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
    )
    trainer.train()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print(f"\n Trained IPA-BERT saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()

