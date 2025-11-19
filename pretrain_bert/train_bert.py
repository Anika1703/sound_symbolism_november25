import os
from transformers import (
    BertConfig, BertForMaskedLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, PreTrainedTokenizerFast
)
import torch
from datasets import Dataset
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import pandas as pd

DATA_FILE = "wikipron_combined.tsv"
MODEL_DIR = "bert-ipa-model"
MAX_LEN = 64

# === Step 1: Load IPA data ===
def load_ipa_data():
    lines = []
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                _, ipa = parts
                char_seq = ipa # space-separated characters
                lines.append({"text": char_seq})
    return Dataset.from_list(lines)

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # === Step 2: Load data first (we’ll use it to train vocab) ===
    dataset = load_ipa_data()
    print(f"\n✅ Loaded {len(dataset)} IPA sequences.")

    # === Step 3: Build character-level tokenizer from data ===
    trainer = trainers.WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    tokenizer_model = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer_model.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    tokenizer_model.train_from_iterator([d["text"] for d in dataset], trainer)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_model,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )

    # === Step 4: Sanity check tokenization ===
    sample_text = dataset[0]["text"]
    print("\n🧪 Tokenization test")
    print("Original:", sample_text)
    print("Tokens:", tokenizer.convert_ids_to_tokens(tokenizer(sample_text)["input_ids"]))

    # === Step 5: Tokenize full dataset ===
    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

    tokenized_ds = dataset.map(tokenize, batched=True)

    # === Step 6: Build BERT model ===
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        max_position_embeddings=MAX_LEN,
    )

    model = BertForMaskedLM(config)

    # === Step 7: Training setup ===
    args = TrainingArguments(
        output_dir=MODEL_DIR,
        per_device_train_batch_size=128,
        num_train_epochs=2,
        logging_dir=None,
        report_to=[], 
        logging_steps=1000,
        save_total_limit=2,
        overwrite_output_dir=True,
        no_cuda=True  # Change to False if you have a GPU
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
    )

    # === Step 8: Train and save ===
    trainer.train()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print(f"Trained model and tokenizer saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
