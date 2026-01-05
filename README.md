1) Data files: corpus_combined_new.csv has the 810 words with their size label (0 for small, 1 for large) and language name. Language_bins_lookup.csv has the Levenstein distances between the languages, as well as the similarity bin the second language falls into with respect to the first. 
2) Word lists as pdfs for each language are in the folder IPA. This is the input format for the baseline classifiers (logistic regression, decision trees). All baseline classification code is in baseline_classifiers.py
3) The pretrained wikipron model is wikipron_combined.tsv
4) Adversarial scrubbing with gradient reversal code is in adversarial_scrubbing.py. This takes in bert-ipa-model which is the frozen IPA Bert model. The custom tokenizer is IpatokHFTokenizer
