### Main Datasets
1. **corpus_combined_new.csv** - 810 size adjectives (27 languages × 30 words)
   - `transcription`: IPA phonemic transcriptions
   - `label`: Binary size labels (0=small, 1=large)
   - `lang_name`: Language identifier

2. **language_bins_lookup.csv** - Phylogenetic similarity between languages
   - Levenshtein distances computed from 40-word Swadesh lists
   - `Similarity_Bin`: Categorical groupings (most/somewhat/least similar)
   - Used for leave-one-language-out experimental design

3. **wikipron_combined.tsv** - Pretraining corpus (~1.6M IPA-transcribed words)
   - Used for BERT masked language modeling pretraining

### Supporting Data
- **IPA/** - Word lists as PDFs for each language (input for baseline classifiers)


### 1. Tokenizer (`tokenizer.py`)
- **IpatokHFTokenizer**: Custom HuggingFace-compatible tokenizer using `ipatok` library

### 2. BERT Pretraining
Trains IPA-BERT from scratch on WikiPron data:
- **Input**: `wikipron_combined.tsv` (1.6M IPA sequences)
- **Architecture**: 2-layer BERT, 128 hidden units, 2 attention heads
- **Task**: Masked language modeling (MLM)
- **Training**: 2 epochs
- **Output**: Frozen BERT encoder saved to `bert-ipa-model/`

### 3. Adversarial Training (`adversarial_scrubbing.py`)
Implements gradient reversal to scrub phylogenetic signal while preserving size information:
**Architecture:**
- Frozen BERT encoder (from pretraining)
- Trainable projection layer (128→64)
- Size classifier (2-layer FFN)
- Bin adversary (2-layer FFN) with gradient reversal layer

**Modes:**
- Baseline (`use_adversarial=False`): Multi-task learning (size + bins)
- Adversarial (`use_adversarial=True, lambda=1.0`): Gradient reversal to suppress bin prediction

### 4. Baseline Classifiers (`baseline_classifiers.py`)
Logistic regression and decision tree classifiers with:
- Bag-of-phoneme features
- Leave-one-language-out evaluation
- Similarity bin analysis (most/somewhat/least similar)
