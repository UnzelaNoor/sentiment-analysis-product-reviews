# Sentiment Analysis on Product Reviews

Binary sentiment classification of Amazon product reviews. The project compares two models trained on the data (Naive Bayes and Logistic Regression) with a pretrained lexicon-based model (VADER) on the same test set.

## Dataset

- **Sentiment Labelled Sentences** (UCI Machine Learning Repository), Amazon subset: `amazon_cells_labelled.txt`
- 1,000 short review sentences about cell phones and accessories
- Labels: `1` = positive, `0` = negative
- Perfectly balanced: 500 positive and 500 negative
- No neutral class

## Approach

1. **Explore the data** and check class balance
2. **Clean the text** (NLTK): lowercase, remove punctuation, remove stop words, lemmatize
3. **Vectorize** with TF-IDF
4. **Split** into 80% train and 20% test (800 / 200 sentences, `random_state=42`)
5. **Train** Multinomial Naive Bayes and Logistic Regression
6. **Benchmark** against VADER, which needs no training and is applied to the raw sentences
7. **Evaluate** with accuracy, precision, recall, F1 and confusion matrices
8. **Explore** frequent words with positive and negative WordClouds

## Results (200 test sentences)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Naive Bayes | 0.80 | 0.80 |
| Logistic Regression | 0.77 | 0.77 |
| VADER (pretrained) | 0.795 | 0.78 |

Per-class results:

| Model | Class | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | Negative | 0.80 | 0.75 | 0.78 |
| Naive Bayes | Positive | 0.80 | 0.84 | 0.82 |
| Logistic Regression | Negative | 0.72 | 0.84 | 0.77 |
| Logistic Regression | Positive | 0.84 | 0.71 | 0.77 |
| VADER | Negative | 0.93 | 0.60 | 0.73 |
| VADER | Positive | 0.74 | 0.96 | 0.83 |

Confusion matrices (rows = true class, columns = predicted; order: negative, positive):

| Model | Matrix |
|---|---|
| Naive Bayes | [[70, 23], [17, 90]] |
| Logistic Regression | [[78, 15], [31, 76]] |
| VADER | [[56, 37], [4, 103]] |

## Key observations

- **The three models perform similarly** (77% to 80%). With only 200 test sentences, these differences are too small to declare a clear winner.
- **Naive Bayes** had the best accuracy and the most balanced results across both classes.
- **Logistic Regression** found most negative reviews (recall 0.84) but missed many positive ones (recall 0.71).
- **VADER** matched the trained models without any training, but it leans positive: its high positive recall (0.96) comes with low negative recall (0.60), because sentences with a neutral score of 0 are labelled positive.
- **Negation is a weak spot.** "Not satisfied at all" was predicted as positive by both trained models, because standard stop-word removal deletes "not".

## Limitations and next steps

- Small dataset and a single 200-sentence test set (add k-fold cross-validation)
- Stop-word removal drops negation words (keep "not", or add bigrams with `ngram_range=(1, 2)`)
- The TF-IDF vectorizer was fitted before the train/test split; a scikit-learn `Pipeline` would remove this leakage
- Binary labels only; a neutral class would need a dataset with star ratings
- Try hyperparameter tuning and a transformer-based model (e.g. DistilBERT)
- Deploy as a small Streamlit or Flask demo

## Project structure

```
├── SentimentAnalysis.ipynb     # code, results and plots
├── amazon_cells_labelled.txt   # dataset
└── README.md
```

## How to run

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud vaderSentiment
python -m nltk.downloader stopwords wordnet punkt
```

Open `SentimentAnalysis.ipynb` in Jupyter and run all cells. Keep `amazon_cells_labelled.txt` in the same folder.

## Tools

Python, pandas, NumPy, scikit-learn, NLTK, VADER, Matplotlib, WordCloud
