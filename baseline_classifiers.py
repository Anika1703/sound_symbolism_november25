import pandas as pd
import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
from collections import defaultdict
import re
from ipatok import tokenise


IPATOK_CONFIG = {
    'strict': False,
    'replace': True,
    'diphthongs': True,
    'tones': True,
    'unknown': False
}
def extract_lines_from_pdf(pdf_path):
    lines = []
    try:
        for page_layout in extract_pages(pdf_path):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    for text_line in element:
                        line = text_line.get_text().strip()
                        if line and not line.isspace():
                            lines.append(line)
        lines = lines[:30]
        return lines
    except Exception as e:
        print(f"ERROR: Processing {pdf_path}: {str(e)}")
        return None

def combine_affricates(tokens):
    TIE_BARS = ['\u0361', '\u035C']
    AFFRICATE_PAIRS = {
        ('t', 's'), ('t', 'ʃ'), ('t', 'ɕ'), ('t', 'ʂ'),
        ('d', 'z'), ('d', 'ʒ'), ('d', 'ʑ'), ('d', 'ʐ'),
        ('ʈ', 'ʂ'), ('ɖ', 'ʐ'),
        ('t', 'θ'), ('d', 'ð'),
    }
    combined = []
    i = 0
    while i < len(tokens):
        if i >= len(tokens) - 1:
            # Last token, can't combine
            combined.append(tokens[i])
            i += 1
            continue
        current = tokens[i]
        next_token = tokens[i + 1]
        starts_with_tie = any(next_token.startswith(tb) for tb in TIE_BARS)
        has_tie = any(tb in next_token for tb in TIE_BARS)
        if starts_with_tie or has_tie:
            combined.append(current + next_token)
            i += 2
            continue
        ends_with_tie = any(current.endswith(tb) for tb in TIE_BARS)
        if ends_with_tie:
            # Combine current (with tie) + next
            combined.append(current + next_token)
            i += 2
            continue
        current_clean = ''.join(c for c in current if c.isalpha() or c in 'ʃʒɕʑʂʐθð')
        next_clean = ''.join(c for c in next_token if c.isalpha() or c in 'ʃʒɕʑʂʐθð')
        if (current_clean, next_clean) in AFFRICATE_PAIRS:
            combined.append(current + '\u0361' + next_token)
            i += 2
            continue
        combined.append(current)
        i += 1
    return combined

def tokenize_with_ipatok(text):
    """
    Tokenize text using ipatok library, then combine affricates.
    """
    try:
        tokens = tokenise(
            text,
            strict=IPATOK_CONFIG['strict'],
            replace=IPATOK_CONFIG['replace'],
            diphthongs=IPATOK_CONFIG['diphthongs'],
            tones=IPATOK_CONFIG['tones'],
            unknown=IPATOK_CONFIG['unknown']
        )
        tokens = combine_affricates(tokens)
        return tokens
    except Exception as e:
        print(f"Warning: ipatok tokenization failed: {e}")
        return []

def custom_ipatok_analyzer(doc):
    """
    Custom analyzer for CountVectorizer using ipatok with affricate combining.
    """
    tokens = tokenize_with_ipatok(doc)
    return tokens

def get_bin_languages(target_lang, bin_df, similarity_type):
    """Get languages in a specific similarity bin for a target language"""
    bin_langs = bin_df[
        (bin_df['Target_Language'] == target_lang) & 
        (bin_df['Similarity_Bin'] == similarity_type)
    ]['Comparison_Language'].tolist()
    return bin_langs

def process_language_data(language, ipa_folder):
    """Process single language data and return features and labels"""
    pdf_path = os.path.join(ipa_folder, f"{language.upper()}.pdf")
    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        return None, None
    lines = extract_lines_from_pdf(pdf_path)
    if not lines:
        print(f"ERROR: No lines extracted from {pdf_path}")
        return None, None
    labels = [0]*15 + [1]*15
    return lines, labels

def get_all_phonemes(vectorizer, model):
    """Extract all phonemes with their coefficients"""
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    coef_df = pd.DataFrame({
        'phoneme': feature_names,
        'coefficient': coefficients
    })
    coef_df = coef_df.reindex(coef_df['coefficient'].abs().sort_values(ascending=False).index)
    top_positive = coef_df.nlargest(10, 'coefficient')
    top_negative = coef_df.nsmallest(10, 'coefficient')
    return coef_df, top_positive, top_negative

def train_and_evaluate(train_data, train_labels, test_data, test_labels, vectorizer):
    """Train models and return accuracies and models"""
    try:
        X_train = vectorizer.transform(train_data).toarray()
        X_test = vectorizer.transform(test_data).toarray()
        feature_names = vectorizer.get_feature_names_out()
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(X_train, train_labels)
        lr_pred = lr_model.predict(X_test)
        lr_accuracy = accuracy_score(test_labels, lr_pred)
        print(f"Logistic Regression accuracy: {lr_accuracy:.4f}")
        results = []
        for max_depth in [3, 4, 5, 6]:
            for min_samples_split in [2, 3, 5, 7, 10]:
                dt = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
                dt.fit(X_train, train_labels)
                train_pred = dt.predict(X_train)
                train_acc = accuracy_score(train_labels, train_pred)
                
                test_pred = dt.predict(X_test)
                test_acc = accuracy_score(test_labels, test_pred)
                gap = train_acc - test_acc
                results.append({
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'gap': gap
                })
                
        results_df = pd.DataFrame(results)
        best_idx = results_df['test_accuracy'].idxmax()
        best_params = {
            'max_depth': results_df.loc[best_idx, 'max_depth'],
            'min_samples_split': results_df.loc[best_idx, 'min_samples_split']
        }
        print(f"Best Decision Tree parameters: {best_params}")
        dt_model = DecisionTreeClassifier(
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            random_state=42
        )
        dt_model.fit(X_train, train_labels)
        dt_pred = dt_model.predict(X_test)
        dt_accuracy = accuracy_score(test_labels, dt_pred)
        print(f"Decision Tree accuracy: {dt_accuracy:.4f}")
        return lr_accuracy, dt_accuracy, lr_model, dt_model, best_params, results
    except Exception as e:
        print(f"ERROR: train_and_evaluate: {str(e)}")
        import traceback
        traceback.print_exc()
        return None 

def main():
    bin_df = pd.read_csv('language_bins_lookup.csv')
    results = []
    phoneme_results = defaultdict(list)
    all_phoneme_coeffs = []
    target_languages = bin_df['Target_Language'].unique()
    ipa_folder = 'IPA'
    for target_lang in target_languages:
        print(f"{target_lang}")
        target_data, target_labels = process_language_data(target_lang, ipa_folder)
        if target_data is None:
            continue
            
        for similarity_bin in ['Most Similar', 'Somewhat Similar', 'Least Similar']:
            print(f"{similarity_bin} bin")
            bin_languages = get_bin_languages(target_lang, bin_df, similarity_bin)
            train_data = []
            train_labels = []
            for lang in bin_languages:
                lang_data, lang_labels = process_language_data(lang, ipa_folder)
                if lang_data is not None:
                    train_data.extend(lang_data)
                    train_labels.extend(lang_labels)
            if not train_data:
                print(f"ERROR: nodata")
                continue
            vectorizer = CountVectorizer(
                analyzer=custom_ipatok_analyzer,
                lowercase=False,
                token_pattern=None
            )
            vectorizer.fit(train_data)
            model_results = train_and_evaluate(train_data, train_labels, target_data, target_labels, vectorizer)
            if model_results is not None:
                lr_acc, dt_acc, lr_model, dt_model, best_params, _ = model_results
                results.append({
                    'Target_Language': target_lang,
                    'Similarity_Bin': similarity_bin,
                    'Logistic_Regression_Accuracy': lr_acc,
                    'Decision_Tree_Accuracy': dt_acc,
                    'DT_Best_Max_Depth': best_params['max_depth'] if best_params else None,
                    'DT_Best_Min_Samples_Split': best_params['min_samples_split'] if best_params else None,
                    'Training_Languages': ', '.join(bin_languages),
                    'Num_Training_Languages': len(bin_languages)
                })
                if lr_model is not None:
                    all_coefs, top_positive, top_negative = get_all_phonemes(vectorizer, lr_model)
                    
                    phoneme_results[f"{target_lang}_{similarity_bin}"] = {
                        'large_phonemes': top_positive,
                        'small_phonemes': top_negative
                    }
                    for _, row in all_coefs.iterrows():
                        all_phoneme_coeffs.append({
                            'Language': target_lang,
                            'Bin': similarity_bin,
                            'Phoneme': row['phoneme'],
                            'Coefficient': row['coefficient']
                        })
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('./ipa_tok_base_classifiers/classification_results_ipatok_affricates.csv', index=False)
        print("\nAccuracy Summary Statistics:")
        summary = results_df.groupby('Similarity_Bin')[
            ['Logistic_Regression_Accuracy', 'Decision_Tree_Accuracy']
        ].agg(['mean'])
        print(summary)
    else:
        print("\nWARNING: No results collected!")
    
    all_phoneme_df = pd.DataFrame(all_phoneme_coeffs)
    all_phoneme_df.to_csv('./ipa_tok_base_classifiers/all_phoneme_coefficients_ipatok_affricates.csv', index=False)
    
    detailed_results = []
    for _, row in all_phoneme_df.iterrows():
        matching_result = next((r for r in results if r['Target_Language'] == row['Language'] and r['Similarity_Bin'] == row['Bin']), None)
        if matching_result:
            detailed_results.append({
                'Language': row['Language'],
                'Bin': row['Bin'],
                'Phoneme': row['Phoneme'],
                'Coefficient': row['Coefficient'],
                'Size_Class': 'large' if row['Coefficient'] > 0 else 'small',
                'Coefficient_Abs': abs(row['Coefficient']),
                'LR_Accuracy': matching_result['Logistic_Regression_Accuracy'],
                'DT_Accuracy': matching_result['Decision_Tree_Accuracy'],
            })

    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv('./ipa_tok_base_classifiers/filtered_ipa_all_res_ipatok_affricates.csv', index=False)

if __name__ == "__main__":
    main()
