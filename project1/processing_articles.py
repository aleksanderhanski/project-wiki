import json
import re

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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


def process_json_to_csv(input_filename, output_filename):
    with open(input_filename, 'r') as f:
        data = json.load(f)
    
    rows = []
    for entry in data:
        cleaned_text = clean_text(entry['text'])
        processed_tokens = process_text(cleaned_text)
        rows.append({
            'url': entry['url'],
            'cleaned_text': cleaned_text,
            'processed_tokens': ' '.join(processed_tokens)
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_filename, index=False, encoding='utf-8')


process_json_to_csv('output.json', 'wiki_processed.csv')