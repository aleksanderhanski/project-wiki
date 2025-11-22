import json
import csv
import re

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, wordpunct_tokenize

import requests
from bs4 import BeautifulSoup


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd



def clean_text(text):
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove edition-only tags like [Bedrock Edition only]
    #text = re.sub(r'\[\s*\w+(\s+\w+)*\s+only\s*\]', '', text, flags=re.IGNORECASE)
    # maybe too agressive, commented out for now

    # Remove reference brackets like [1], [2], etc.
    text = re.sub(r'\[\s*\d+\s*\]', '', text)
    
    # Remove standalone brackets like [ ]
    text = re.sub(r'\[\s*\]', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)

    # Remove section headers pattern (words followed by [ ])
    text = re.sub(r'(\w+)\s*\[\s*\]', r'\1', text)

    # Remove lines that are just navigation/menu items
    text = re.sub(r'\b\d+(\.\d+)*\s+[A-Z][a-z]+\b', '', text)  # Like "9.1 Mash-up Packs"
    
    return text.strip()


def process_text(text):

    lemmatizer = WordNetLemmatizer()
    
    stop_words = set(stopwords.words('english'))
    
    tokens = word_tokenize(text.lower())
    
    processed_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words
    ]
    
    return processed_tokens


def download_article_text(filename='project1/input_articles.csv'):
    headers = {
        'User-Agent': 'Olek-and-Karol'
    }

    rows = []

    while True:
        url = input("Enter article URL (or 'q' to quit): ")
        if url.lower() == 'q':
            break
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        parsed = BeautifulSoup(response.text, 'html.parser')
        
        text = ''
        for p in parsed.select('p'):
            text += p.get_text() + ' '

        cleaned_text = clean_text(text)
        processed_tokens = process_text(cleaned_text)
        
        rows.append({
            'url': url,
            'cleaned_text': cleaned_text,
            'processed_tokens': ' '.join(processed_tokens)
        })

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False, encoding='utf-8')


def find_similar_articles(input_csv='input_articles.csv',
                                    base_csv='wiki_processed.csv',
                                    top_k=5,
                                    candidate_count=50,
                                    hybrid_weight=0.2):
    """
    Two-stage similarity search: TF-IDF to narrow candidates, then hybrid (tfidf and semantic) to rank best articles.
    
    top_k: Number of recommendations to return
    candidate_count: Number of candidates to keep after TF-IDF stage
    hybrid_weight: Weight for TF-IDF in hybrid scoring (0-1)
    """
    
    df = pd.read_csv(base_csv)
    new_df = pd.read_csv(input_csv)
    
    base_tokens = df['processed_tokens'].fillna('')
    base_texts = df['cleaned_text'].fillna('')
    
    input_tokens = new_df['processed_tokens'].dropna().astype(str)
    input_tokens = input_tokens[input_tokens.str.strip() != '']    
    input_texts = new_df['cleaned_text'].dropna().astype(str)
    input_texts = input_texts[input_texts.str.strip() != '']
    
    if input_tokens.empty and input_texts.empty:
        print("No valid input articles found")
        return []

    # Exclude input articles
    input_urls = set(new_df['url'].dropna())
    mask = ~df['url'].isin(input_urls)

    # stage 1: tfidf to get candidates
    tfidf_sims_all = compute_tfidf_similarity(input_tokens, base_tokens)
    # exclude input articles
    masked_tfidf = np.where(mask, tfidf_sims_all, -1)
    candidate_idx = masked_tfidf.argsort()[-candidate_count:]

    # stage 2: hybrid scoring on candidates only
    candidate_tokens = base_tokens.iloc[candidate_idx]
    candidate_texts = base_texts.iloc[candidate_idx]
    
    tfidf_sims = compute_tfidf_similarity(input_tokens, candidate_tokens)
    semantic_sims = compute_semantic_similarity(input_texts, candidate_texts)
    
    tfidf_norm = normalize_scores(tfidf_sims)
    semantic_norm = normalize_scores(semantic_sims)
    combined_sims = hybrid_weight * tfidf_norm + (1 - hybrid_weight) * semantic_norm

    top_local_idx = combined_sims.argsort()[-top_k:][::-1]
    top_idx = candidate_idx[top_local_idx]

    results = [
        {
            'url': df.iloc[idx].get('url', ''),
            'similarity': float(combined_sims[local_idx]),
            'preview': str(df.iloc[idx].get('cleaned_text', ''))[:200]
        }
        for local_idx, idx in zip(top_local_idx, top_idx)
    ]

    print("\nInput articles:")
    for url in input_urls:
        print(f"- {url}")
    print(f"\nRECOMMENDATIONS:")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['url']}")
        print(f"Similarity score: {r['similarity']:.4f}")
        print(f"Preview: {r['preview']}...")
        print("-" * 40)


def compute_tfidf_similarity(input_tokens, base_tokens):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(base_tokens)
    input_vectors = vectorizer.transform(input_tokens)
    
    all_sims = cosine_similarity(input_vectors, tfidf_matrix)
    return all_sims.mean(axis=0)


def compute_semantic_similarity(input_texts, base_texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    base_embeddings = model.encode(base_texts.tolist(), show_progress_bar=False)
    input_embeddings = model.encode(input_texts.tolist(), show_progress_bar=False)
    
    all_sims = cosine_similarity(input_embeddings, base_embeddings)
    return all_sims.mean(axis=0)


def normalize_scores(scores):
    min_score = scores.min()
    max_score = scores.max()
    
    if max_score == min_score:
        return np.zeros_like(scores)
    
    return (scores - min_score) / (max_score - min_score)


download_article_text(filename='input_articles.csv')

find_similar_articles('input_articles.csv', 'wiki_processed.csv', top_k=10, hybrid_weight=1.0)