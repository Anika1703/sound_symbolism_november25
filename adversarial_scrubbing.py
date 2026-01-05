"""
Adversarial Sound Symbolism Training - GRADIENT REVERSAL VERSION
Baseline: Multi-task (learn size + bins)
Adversarial: Learn size, scrub bins via gradient reversal with lambda ramping
"""
#current configs are set for the baseline case (no adversarial pressure). set use_adversarial to True and lambda to 1.0 for adversarial case. 
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformers import BertModel
from tokenizer import IpatokHFTokenizer

# GRADIENT REVERSAL LAYER
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        """
        Forward pass is identity; we just stash lambda_ for the backward.
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass multiplies gradient by -lambda_.
        This makes the encoder *maximize* the adversary's loss.
        """
        lambda_ = ctx.lambda_
        return -lambda_ * grad_output, None  # None for lambda_ (no grad)


def grad_reverse(x, lambda_):
    return GradientReversal.apply(x, lambda_)



# DATASET
class IPADataset(Dataset):
    def __init__(self, df, tokenizer, max_length=64):
        self.texts = df["transcription"].astype(str).tolist()
        self.size_labels = torch.tensor(df["label"].values, dtype=torch.long)
        self.bin_labels = torch.tensor(df["bin_id"].values, dtype=torch.long)

        self.encodings = tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.size_labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        return item, self.size_labels[idx], self.bin_labels[idx]


# MODEL

class AdversarialModel(nn.Module):
    #defaults overriden below
    def __init__(self, bert_path, embedding_dim=64, num_bins=3, dropout=0.4):
        super().__init__()

        # Frozen BERT
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False

        bert_dim = self.bert.config.hidden_size

        # Trainable projection (encoder E)
        self.projection = nn.Linear(bert_dim, embedding_dim)

        # Dropout on shared embedding
        self.dropout = nn.Dropout(dropout)

        # Size head (classifier C)
        self.size_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

        # Bin head (adversary A)
        self.bin_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_bins),
        )

    def get_embeddings(self, input_ids, attention_mask, token_type_ids=None):
        """Get embeddings from frozen BERT + trainable projection."""
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            # Mean pooling over non-pad tokens
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)

        # Trainable projection (gets gradients)
        embeddings = self.projection(pooled)
        return embeddings

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        lambda_adv: float = 0.0,
        use_adversarial: bool = False,
    ):
        """
        If use_adversarial and lambda_adv > 0:
          - Size head sees normal embeddings
          - Bin head sees embeddings through Gradient Reversal
            (which flips gradients into the encoder)

        If not:
          - Both heads see normal embeddings (standard multi-task).
        """
        embeddings = self.get_embeddings(input_ids, attention_mask, token_type_ids)
        dropped = self.dropout(embeddings)

        # Size prediction (always standard)
        size_logits = self.size_head(dropped)

        # Bin prediction (possibly adversarial)
        if use_adversarial and lambda_adv > 0:
            bin_input = grad_reverse(dropped, lambda_adv)
        else:
            bin_input = dropped

        bin_logits = self.bin_head(bin_input)

        return size_logits, bin_logits



# TRAINING
def train_epoch(
    model,
    loader,
    proj_optimizer,
    size_optimizer,
    bin_optimizer,
    device,
    lambda_adv,
    use_adversarial,
):
    """
    BASELINE (use_adversarial=False OR lambda_adv=0):
      - Multi-task: learn size AND bins together
      - Loss = size_loss + bin_loss
      - Both size and bin acc should go up

    ADVERSARIAL (use_adversarial=True and lambda_adv>0):
      - Gradient reversal on the bin branch:
          * Bin head tries to predict bins (minimize bin_loss)
          * Encoder (projection) sees flipped gradient, so it
            tries to make bins *hard to predict*
      - Loss (w.r.t. heads) is still size_loss + bin_loss.
        The "adversarial" part happens in the backward pass.
    """
    model.train()

    total_size_loss = 0.0
    total_bin_loss = 0.0
    size_correct = 0
    bin_correct = 0
    total = 0

    size_criterion = nn.CrossEntropyLoss()
    bin_criterion = nn.CrossEntropyLoss()

    for inputs, size_labels, bin_labels in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        size_labels = size_labels.to(device)
        bin_labels = bin_labels.to(device)

        proj_optimizer.zero_grad()
        size_optimizer.zero_grad()
        bin_optimizer.zero_grad()

        size_logits, bin_logits = model(
            **inputs,
            lambda_adv=lambda_adv if use_adversarial else 0.0,
            use_adversarial=use_adversarial,
        )

        size_loss = size_criterion(size_logits, size_labels)
        bin_loss = bin_criterion(bin_logits, bin_labels)

        # Same loss form in both modes; GRL changes how bin_loss
        # gradients flow into the encoder.
        loss = size_loss + bin_loss

        loss.backward()
        proj_optimizer.step()
        size_optimizer.step()
        bin_optimizer.step()

        # Track metrics
        total_size_loss += size_loss.item()
        total_bin_loss += bin_loss.item()

        with torch.no_grad():
            size_preds = size_logits.argmax(1)
            bin_preds = bin_logits.argmax(1)
            size_correct += (size_preds == size_labels).sum().item()
            bin_correct += (bin_preds == bin_labels).sum().item()
            total += len(size_labels)

    return {
        "size_loss": total_size_loss / len(loader),
        "bin_loss": total_bin_loss / len(loader),
        "size_acc": size_correct / total,
        "bin_acc": bin_correct / total,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate on a loader (no adversarial gradients active).

    NOTE: We only care about SIZE on test;
    bin labels on test are dummy and ignored.
    """
    model.eval()

    size_correct = 0
    total = 0

    for inputs, size_labels, _ in loader:  # ignore bin_labels
        inputs = {k: v.to(device) for k, v in inputs.items()}
        size_labels = size_labels.to(device)

        size_logits, _ = model(**inputs)

        size_preds = size_logits.argmax(1)
        size_correct += (size_preds == size_labels).sum().item()
        total += len(size_labels)

    return {
        "size_acc": size_correct / total,
    }



# DATA LOADING

def load_data_for_language(corpus_csv, bin_lookup_csv, target_language):
    """Load data for leave-one-language-out"""
    corpus = pd.read_csv(corpus_csv)
    bins = pd.read_csv(bin_lookup_csv)

    # Normalize
    corpus["lang_name"] = corpus["lang_name"].str.strip().str.lower()
    bins["Target_Language"] = bins["Target_Language"].str.strip().str.lower()
    bins["Comparison_Language"] = bins["Comparison_Language"].str.strip().str.lower()
    target_language = target_language.strip().lower()

    # Get bins for target
    bins_for_target = bins[bins["Target_Language"] == target_language]

    # Merge bins with corpus
    merged = corpus.merge(
        bins_for_target[["Comparison_Language", "Similarity_Bin"]],
        left_on="lang_name",
        right_on="Comparison_Language",
        how="inner",
    )

    # Map bin names to IDs
    bin_map = {"most similar": 0, "somewhat similar": 1, "least similar": 2}
    merged["Similarity_Bin"] = merged["Similarity_Bin"].str.strip().str.lower()
    merged["bin_id"] = merged["Similarity_Bin"].map(bin_map)

    # Train = other languages, Test = target language
    train_df = merged[merged["lang_name"] != target_language].copy()
    test_df = corpus[corpus["lang_name"] == target_language].copy()

    # Dummy bin_id for test (required by Dataset, but NOT used for evaluation)
    test_df["bin_id"] = 0

    print(f"Target: {target_language} | Train: {len(train_df)} | Test: {len(test_df)}")

    return train_df, test_df

def train_one_language(
    target_language, corpus_csv, bin_lookup_csv, config, output_dir, seed=42
):
    """
    Train model for one target language.

    - Uses Ganin-style lambda ramp:
        p = epoch / epochs
        lambda_epoch = lambda_max * (2 / (1 + exp(-10p)) - 1)
    - No validation split
    - Logs train + test SIZE metrics per epoch (plus lambda ramp) to CSV
    - Runs for a fixed number of epochs (no early stopping)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    use_adv = config["use_adversarial"] and config["lambda"] > 0
    mode = "ADVERSARIAL (GRL)" if use_adv else "BASELINE (multi-task)"

    num_bins = 3

    print(f"\n{'='*60}")
    print(f"Target: {target_language}")
    print(f"Mode: {mode} | Lambda_max: {config['lambda']}")
    print(f"{'='*60}")

    # Load data
    train_df, test_df = load_data_for_language(
        corpus_csv, bin_lookup_csv, target_language
    )

    # Tokenizer
    tokenizer = IpatokHFTokenizer.from_pretrained(config["bert_path"])
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Datasets
    train_ds = IPADataset(train_df, tokenizer, config["max_length"])
    test_ds = IPADataset(test_df, tokenizer, config["max_length"])

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=config["batch_size"], shuffle=False
    )

    # Model
    model = AdversarialModel(
        bert_path=config["bert_path"],
        embedding_dim=config["embedding_dim"],
        num_bins=num_bins,
        dropout=config["dropout"],
    ).to(device)

    # Optimizers
    proj_optimizer = optim.Adam(model.projection.parameters(), lr=config["lr"])
    size_optimizer = optim.Adam(model.size_head.parameters(), lr=config["lr"])
    bin_optimizer = optim.Adam(model.bin_head.parameters(), lr=config["lr"])

    history = []

    for epoch in range(1, config["epochs"] + 1):
        # ---------------- Lambda ramp (Ganin schedule) ----------------
        p = epoch / config["epochs"]  # progress in [0,1]
        lambda_epoch = config["lambda"] * (
            2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
        )
        lambda_epoch = float(lambda_epoch)

        # TRAIN
        train_metrics = train_epoch(
            model,
            train_loader,
            proj_optimizer,
            size_optimizer,
            bin_optimizer,
            device,
            lambda_epoch,
            config["use_adversarial"],
        )

        # TEST (size only, no bin metric)
        test_metrics = evaluate(model, test_loader, device)

        print(
            f"[Epoch {epoch:2d}] λ(t)={lambda_epoch:.3f} | "
            f"Train: Size={train_metrics['size_acc']:.3f} Bin={train_metrics['bin_acc']:.3f} | "
            f"Test: Size={test_metrics['size_acc']:.3f}"
        )

        # Log per-epoch metrics (including lambda) for CSV
        history.append(
            {
                "epoch": epoch,
                "lambda_max": config["lambda"],
                "lambda_epoch": lambda_epoch,
                "train_size_loss": train_metrics["size_loss"],
                "train_bin_loss": train_metrics["bin_loss"],
                "train_size_acc": train_metrics["size_acc"],
                "train_bin_acc": train_metrics["bin_acc"],
                "test_size_acc": test_metrics["size_acc"],
            }
        )

    # Convert history to DataFrame & save per-language CSV
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(
        os.path.join(output_dir, f"{target_language}_metrics.csv"), index=False
    )

    last_row = hist_df.iloc[-1]

    print(
        f"✅ Done: {target_language} | "
        f"Final epoch={int(last_row['epoch'])} "
        f"(TestSize={last_row['test_size_acc']:.3f}, "
        f"TrainSize={last_row['train_size_acc']:.3f}, "
        f"TrainBin={last_row['train_bin_acc']:.3f})\n"
    )

    return {
        "language": target_language,
        "lambda_max": config["lambda"],
        "final_test_size_acc": float(last_row["test_size_acc"]),
        "final_train_size_acc": float(last_row["train_size_acc"]),
        "final_train_bin_acc": float(last_row["train_bin_acc"]),
    }

def run_all_languages(corpus_csv, bin_lookup_csv, config, output_dir, num_runs=15):
    """Run experiment on all languages"""

    corpus = pd.read_csv(corpus_csv)
    languages = sorted(corpus["lang_name"].str.strip().str.lower().unique())

    mode = (
        "ADVERSARIAL (GRL)"
        if config["use_adversarial"] and config["lambda"] > 0
        else "BASELINE"
    )
    print(f"\n{'='*70}")
    print(f"Mode: {mode}")
    print(f"Running: {len(languages)} languages × {num_runs} runs")
    print(f"{'='*70}\n")

    # Save config
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    all_results = []

    for run_idx in range(num_runs):
        print(f"\n>>> RUN {run_idx + 1}/{num_runs}\n")

        for lang in languages:
            result = train_one_language(
                lang,
                corpus_csv,
                bin_lookup_csv,
                config,
                os.path.join(output_dir, f"run_{run_idx:02d}", lang),
                seed=42 + run_idx,
            )
            result["run"] = run_idx
            all_results.append(result)

    # Save aggregated results
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)

    print(f"\n{'='*70}")
    print(f"DONE! ({mode})")
    print(
        f"Avg Final Test Size Acc: {df['final_test_size_acc'].mean():.4f} "
        f"± {df['final_test_size_acc'].std():.4f}"
    )
    print(
        f"Avg Final Train Size Acc: {df['final_train_size_acc'].mean():.4f} "
        f"± {df['final_train_size_acc'].std():.4f}"
    )
    print(
        f"Avg Final Train Bin Acc: {df['final_train_bin_acc'].mean():.4f} "
        f"± {df['final_train_bin_acc'].std():.4f}"
    )
    print(f"(Bin chance level: 0.333)")
    print(f"{'='*70}\n")


if __name__ == "__main__":

    adversarial_config = {
        "bert_path": "bert-ipa-model",
        "embedding_dim": 64,
        "dropout": 0.1,
        "max_length": 64,
        "batch_size": 32,
        "epochs": 20,        
        "lr": 1e-4,
        "use_adversarial": False,  # set to True for adversarial GRL mode
        "lambda": 0.0,       # set to 1 for adv 
    }

    run_all_languages(
        corpus_csv="Data/corpus_combined_new.csv",
        bin_lookup_csv="Data/language_bins_lookup.csv",
        config=adversarial_config,
        output_dir="ganin_nov13_v_baseline_5runs",
        num_runs=5,
    )
