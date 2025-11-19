# tokenizer.py
# pip install ipatok transformers pandas

import json
from pathlib import Path
from typing import List, Optional
import pandas as pd
from ipatok import tokenise
from transformers import PreTrainedTokenizer

# -----------------------------
# ipatok config (same as baseline)
# -----------------------------
IPATOK_KW = dict(
    strict=False,
    replace=True,
    diphthongs=False,  # True if you want aɪ, aʊ merged
    tones=True,
    unknown=False,
)

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
SAVE_DIR = Path("./idk_bert/fixed_ipa_tokenizer")

# -----------------------------
# 1) Build vocab from corpus (only needed when creating the tokenizer folder)
# -----------------------------
def build_vocab_from_csv(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    ipa_words = df["transcription"].astype(str).tolist()

    vocab_set = set(SPECIAL_TOKENS)
    for w in ipa_words:
        toks = tokenise(w, **IPATOK_KW)
        vocab_set.update(toks)

    vocab_list = [*SPECIAL_TOKENS] + sorted(vocab_set - set(SPECIAL_TOKENS))
    return {tok: i for i, tok in enumerate(vocab_list)}

# -----------------------------
# 2) HuggingFace-compatible tokenizer (slow) using ipatok
# -----------------------------
class IpatokHFTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab: dict,
        ipatok_kwargs: Optional[dict] = None,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        **kwargs,
    ):
        # set vocab first so HF init can see it
        self.vocab = vocab
        self.id2tok = {i: t for t, i in vocab.items()}
        self.ipatok_kwargs = ipatok_kwargs or {}

        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        return tokenise(text, **self.ipatok_kwargs)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self.id2tok.get(index, "[UNK]")

    def get_vocab(self):
        return dict(self.vocab)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None):
        cls_id = self.vocab["[CLS]"]
        sep_id = self.vocab["[SEP]"]
        if token_ids_1 is None:
            return [cls_id] + token_ids_0 + [sep_id]
        return [cls_id] + token_ids_0 + [sep_id] + [cls_id] + token_ids_1 + [sep_id]

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return [1 if t in (self.vocab["[CLS]"], self.vocab["[SEP]"]) else 0 for t in token_ids_0]
        if token_ids_1 is None:
            return [1] + [0]*len(token_ids_0) + [1]
        return [1] + [0]*len(token_ids_0) + [1] + [1] + [0]*len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [0] * (len(token_ids_0) + 2)
        return [0] * (len(token_ids_0) + 2) + [1] * (len(token_ids_1) + 2)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        name = (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        vocab_path = path / name
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        with open(path / "ipatok_config.json", "w", encoding="utf-8") as f:
            json.dump(self.ipatok_kwargs, f, ensure_ascii=False, indent=2)
        return (str(vocab_path),)

    # ✅ custom loader so .from_pretrained("path") works
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        path = Path(pretrained_model_name_or_path)
        with open(path / "vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)
        ipatok_kwargs = IPATOK_KW
        ip_cfg = path / "ipatok_config.json"
        if ip_cfg.exists():
            with open(ip_cfg, "r", encoding="utf-8") as f:
                ipatok_kwargs = json.load(f)
        return cls(vocab=vocab, ipatok_kwargs=ipatok_kwargs, **kwargs)

# -----------------------------
# 3) One-time build + save helper (run once)
# -----------------------------
def build_and_save_tokenizer(csv_path: str, save_dir: Path = SAVE_DIR):
    vocab = build_vocab_from_csv(csv_path)
    tok = IpatokHFTokenizer(vocab=vocab, ipatok_kwargs=IPATOK_KW)

    # demo print
    df = pd.read_csv(csv_path)
    sample = df["transcription"].astype(str).tolist()[:10]
    print("✅ Tokenization examples (first 10):")
    for w in sample:
        toks = tok.tokenize(w)
        ids = tok.convert_tokens_to_ids(toks)
        print(f"{w} -> {toks} -> {ids}")

    save_dir.mkdir(exist_ok=True, parents=True)
    tok.save_pretrained(save_dir)

    with open(save_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump({
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]"
        }, f, ensure_ascii=False, indent=2)

    with open(save_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump({"tokenizer_class": "IpatokHFTokenizer"}, f, ensure_ascii=False, indent=2)

    print(f"Saved fixed tokenizer to '{save_dir}/'")

# -----------------------------
# If you want to (re)build the tokenizer folder, uncomment:
build_and_save_tokenizer('Data/corpus_clean_train.csv')
# -----------------------------
