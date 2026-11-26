# Review Sentiment Analyzer: TF-IDF + Logistic Regression vs. a Pretrained Transformer

Classifies movie and product reviews as positive or negative with a classic
TF-IDF + logistic regression model, then compares it with a pretrained
DistilBERT model (no fine-tuning).

## Files

| File | Purpose |
|---|---|
| `sentiment_compare.py` | Main script: trains TF-IDF + LR on IMDB, evaluates both models on IMDB and Amazon reviews, runs a McNemar significance test, analyses accuracy by review length, saves CSVs and charts to `results/` |
| `finetune_distilbert.py` | Optional (Colab GPU): fine-tunes DistilBERT on IMDB and evaluates it on the same test subsets |
| `app.py` | Optional: small Gradio demo to paste a review and see each model's prediction |
| `requirements.txt` | Python packages |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python sentiment_compare.py                         # full run: 2000 IMDB + 1000 Amazon reviews (about 10-15 min on a laptop CPU)
python sentiment_compare.py --test-size 500 --product-size 300   # quicker
python sentiment_compare.py --skip-product          # IMDB only
python sentiment_compare.py --product-dataset yelp  # Yelp instead of Amazon
python sentiment_compare.py --predict "Great acting, but the plot dragged."
python app.py                                       # demo (run the main script first)
```

Outputs are written to `results/`: `metrics.csv`, `length_analysis.csv`,
`predictions_*.csv`, `results.png`, `length_analysis.png`.
The trained TF-IDF model is cached as `tfidf_logreg_clean.joblib`
(use `--retrain` to rebuild it).

## Method

- **Data:** IMDB (25,000 training reviews, test set shuffled with seed 42 and subsampled to 2,000). Out-of-domain test: 1,000 Amazon product reviews (`fancyzhx/amazon_polarity`). Labels: 0 = negative, 1 = positive.
- **Cleaning:** HTML tags such as `<br />` and extra whitespace are removed.
- **Model 1:** TF-IDF (unigrams and bigrams, `min_df=3`, `max_df=0.9`, sublinear TF, up to 200,000 features) + logistic regression. The regularization strength C is chosen from {0.5, 2, 8, 20} by 3-fold cross-validation (best: C = 8, CV accuracy 0.885 on the earlier uncleaned-text run).
- **Model 2:** `distilbert-base-uncased-finetuned-sst-2-english`, used as-is, inputs truncated to 512 tokens.
- **Evaluation:** accuracy with 95% Wilson confidence interval, F1, confusion matrix, speed (ms per review on CPU), exact McNemar test for the difference between the two models, accuracy by review length.

## Results (CPU run on a MacBook Air)

| Dataset | Model | Accuracy (95% CI) | F1 | Speed |
|---|---|---|---|---|
| IMDB (n=2000) | TF-IDF + LR | 0.903 (0.889-0.915) | 0.903 | 0.27 ms/review |
| IMDB (n=2000) | DistilBERT (pretrained) | 0.891 (0.877-0.904) | 0.888 | 107.5 ms/review |
| Amazon (n=1000) | TF-IDF + LR | 0.855 (0.832-0.875) | 0.865 | 0.10 ms/review |
| Amazon (n=1000) | DistilBERT (pretrained) | 0.900 (0.880-0.917) | 0.900 | 44.6 ms/review |

McNemar exact test:
- IMDB: p = 0.126, not statistically significant (the two models are tied).
- Amazon: p = 0.0005, significant (DistilBERT is better).

Accuracy by review length on IMDB:

| Length bin | n | TF-IDF | DistilBERT |
|---|---|---|---|
| short (<=136 words) | 660 | 0.908 | 0.914 |
| medium (137-218 words) | 671 | 0.902 | 0.900 |
| long (>218 words) | 669 | 0.900 | 0.859 |
| likely truncated (>380 words) | 246 | 0.894 | 0.821 |

## Findings

1. On in-domain movie reviews, TF-IDF + LR matches DistilBERT (difference not significant) while being about 400x faster.
2. On product reviews, TF-IDF + LR loses about 5 points while the pretrained model does not, and the difference is significant.
3. DistilBERT's accuracy falls on long IMDB reviews (to 0.821 above about 380 words), where its 512-token limit likely cuts off the verdict. TF-IDF stays near 0.90 at every length. This is consistent with the truncation explanation but was not tested directly.

## Limitations

- DistilBERT was fine-tuned on SST-2, which is also movie-review text, so the product-review test is a milder domain shift for it than for TF-IDF.
- Amazon labels come from star ratings, so mixed reviews add label noise.
- Length bins hold a few hundred reviews each; the 246-review truncation bin was not tested for significance.
- Speed was measured on one laptop CPU and is only a rough guide.
- TF-IDF was trained only on movie reviews; no model was trained on product reviews.

## Sources

- IMDB: `stanfordnlp/imdb` on Hugging Face (Maas et al., 2011).
- Amazon polarity: `fancyzhx/amazon_polarity` on Hugging Face (Zhang et al., 2015).
- Pretrained model: `distilbert-base-uncased-finetuned-sst-2-english` (Sanh et al., 2019).
