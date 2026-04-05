# WIkipedia Article Recommender

Implementation of an NLP-based article recommendation engine. Given one or more Wikipedia articles as input, the system identifies the most semantically and topically similar articles from a pre-scraped corpus. It combines TF-IDF keyword matching with neural sentence embeddings for a robust hybrid similarity score.

## Pipeline

**Step 1 - Scrape the Corpus**

Run the Scrapy spider to crawl Wikipedia. All linked articles (no namespaced pages) are scraped and saved to output.json.

```bash
scrapy crawl wikipedia -o output.json
```

The spider is configured with DEPTH_LIMIT: 2 and a custom User-Agent. It extracts paragraph text only — tables, references, navboxes, and edit-section links are excluded.

**Step 2 - Process Articles**

Run
```bash
python project1/processing_articles.py
```
to process scraped articles into tokens in csv file.

**Step 3 - Run the Recommender**

Run
```bash
python project1/main.py
```
to launch recommender. 

You will be prompted to enter Wikipedia article URLs one by one. Type q to finish input and trigger the recommendation. The system will download and process each URL, then find the top similar articles from the corpus.

## Requirements

Python Dependencies

| Package | Purpose |
|------------|---------|
| scrapy | Web crawling / scraping |
| requests + beautifulsoup4 | Article downloading in main.py |
| nltk | Tokenization, stopword removal, lemmatization (WordNet) |
| scikit-learn | TF-IDF vectorization and cosine similarity |
| sentence-transformers | Semantic embeddings |
| pandas | CSV I/O and dataframe operations |
| numpy | Score normalization and array operations |
| matplotlib + seaborn | Corpus statistics visualizations |
| scipy | Hirearchical clustering |

## How Similarity Works

Recommendation uses a two-stage hybrid pipeline:

**Stage 1 — TF-IDF Candidate Filtering**

A TF-IDF matrix is computed over all base article tokens. Input article tokens are projected into the same vocabulary space and cosine similarity is computed against every base article. The top 50 candidates are selected. Input articles (matched by URL) are excluded from results.

**Stage 2 — Hybrid Scoring**

On the 50 candidates, two similarity scores are computed independently:

- TF-IDF cosine similarity (sklearn TfidfVectorizer)
- Semantic cosine similarity (SentenceTransformer all-MiniLM-L6-v2)
    
Both scores are min-max normalized before combining, because TF-IDF values typically fall in the 0.05–0.3 range while semantic embeddings cluster around 0.4–0.8. Without normalization, the embedding model would dominate purely due to scale.

Final score formula:

$$score = 0.2 × tfidf-norm + 0.8 × semantic-norm$$

The 0.8 semantic weight reflects that embeddings capture meaning and context better. The 0.2 TF-IDF weight preserves keyword signal that matters for short texts where specific terms are critical.
