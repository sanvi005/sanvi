# Sentiment Analysis of Reviews: TF-IDF + Logistic Regression vs. DistilBERT

```python
"""
Sentiment analyzer for reviews: TF-IDF + Logistic Regression vs. a pretrained transformer.

Version 2 adds: text cleaning, McNemar significance test, confidence intervals,
product-review (domain shift) evaluation, error analysis by review length,
a results chart, and CSV outputs in ./results.

Usage:
    pip install scikit-learn datasets transformers torch joblib numpy pandas matplotlib scipy
    python sentiment_compare.py                        # full run (2000 IMDB + 1000 product reviews)
    python sentiment_compare.py --test-size 500        # faster
    python sentiment_compare.py --skip-product         # IMDB only
    python sentiment_compare.py --product-dataset yelp # use Yelp instead of Amazon
    python sentiment_compare.py --device cuda          # on Colab GPU
    python sentiment_compare.py --predict "Loved it, would watch again!"
"""
import argparse
import math
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

MODEL_PATH = "tfidf_logreg_clean.joblib"
PRETRAINED = "distilbert-base-uncased-finetuned-sst-2-english"
OUT_DIR = Path("results")
TRUNCATION_WORDS = 380  # ~512 DistilBERT word-piece tokens is roughly 380 English words


# --------------------------------------------------------------------------- #
# Text cleaning + data
# --------------------------------------------------------------------------- #
def clean_text(t):
    """Strip HTML tags (IMDB has <br />), literal '\\n' (Yelp) and extra whitespace."""
    t = re.sub(r"<[^>]+>", " ", t)
    t = t.replace("\\n", " ")
    return re.sub(r"\s+", " ", t).strip()


def load_imdb(test_size, seed=42):
    """IMDB: 25k train / 25k test, labels 0=negative, 1=positive."""
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb")
    X_train = [clean_text(t) for t in ds["train"]["text"]]
    y_train = np.array(ds["train"]["label"])

    test = ds["test"].shuffle(seed=seed).select(range(test_size))
    X_test = [clean_text(t) for t in test["text"]]
    y_test = np.array(test["label"])
    return X_train, y_train, X_test, y_test


def load_product(kind, n, seed=42):
    """Out-of-domain test data. Labels 0=negative, 1=positive."""
    from datasets import load_dataset

    if kind == "amazon":
        ds = load_dataset("fancyzhx/amazon_polarity", split="test")
        ds = ds.shuffle(seed=seed).select(range(n))
        texts = [clean_text(f"{t}. {c}") for t, c in zip(ds["title"], ds["content"])]
    else:
        ds = load_dataset("fancyzhx/yelp_polarity", split="test")
        ds = ds.shuffle(seed=seed).select(range(n))
        texts = [clean_text(t) for t in ds["text"]]
    return texts, np.array(ds["label"])


# --------------------------------------------------------------------------- #
# Model 1: TF-IDF + Logistic Regression
# --------------------------------------------------------------------------- #
def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True,
            max_features=200_000,
        )),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
    ])


def train_tfidf(X_train, y_train):
    grid = GridSearchCV(
        build_pipeline(),
        param_grid={"clf__C": [0.5, 2, 8, 20]},
        cv=3, scoring="accuracy", n_jobs=-1, verbose=1,
    )
    grid.fit(X_train, y_train)
    print(f"Best C: {grid.best_params_['clf__C']}  (CV accuracy {grid.best_score_:.4f})")
    joblib.dump(grid.best_estimator_, MODEL_PATH)
    return grid.best_estimator_


def show_top_features(pipe, k=15):
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    print("\nMost NEGATIVE terms:", ", ".join(names[order[:k]]))
    print("Most POSITIVE terms:", ", ".join(names[order[-k:][::-1]]))


# --------------------------------------------------------------------------- #
# Model 2: pretrained transformer (no fine-tuning)
# --------------------------------------------------------------------------- #
def load_pretrained(device="cpu"):
    from transformers import pipeline
    dev = {"cpu": -1, "cuda": 0, "mps": "mps"}[device]
    return pipeline("sentiment-analysis", model=PRETRAINED, device=dev)


def predict_pretrained(clf, texts, batch_size=32):
    outputs = clf(list(texts), batch_size=batch_size, truncation=True, max_length=512)
    return np.array([1 if o["label"] == "POSITIVE" else 0 for o in outputs])


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def wilson_ci(acc, n, z=1.96):
    """95% Wilson score interval for an accuracy measured on n examples."""
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z * z / n
    centre = (acc + z * z / (2 * n)) / denom
    half = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def mcnemar_exact(y, p1, p2):
    """Exact McNemar test: do the two models differ in error rate?"""
    b = int(((p1 == y) & (p2 != y)).sum())  # only model 1 correct
    c = int(((p1 != y) & (p2 == y)).sum())  # only model 2 correct
    p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    return b, c, p


# --------------------------------------------------------------------------- #
# Evaluation / reporting
# --------------------------------------------------------------------------- #
def evaluate(dataset, name, y_true, y_pred, seconds):
    n = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    lo, hi = wilson_ci(acc, n)
    f1 = f1_score(y_true, y_pred)
    ms = 1000 * seconds / n
    print(f"\n=== {name} on {dataset} (n={n}) ===")
    print(f"Accuracy: {acc:.4f} (95% CI {lo:.3f}-{hi:.3f})   F1: {f1:.4f}   Speed: {ms:.2f} ms/review")
    print("Confusion matrix [rows=true neg/pos, cols=pred neg/pos]:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"], digits=3))
    return {"dataset": dataset, "model": name, "n": n, "accuracy": acc,
            "ci_low": lo, "ci_high": hi, "f1": f1, "ms_per_review": ms}


def compare_models(dataset, y, p_tfidf, p_bert):
    b, c, p = mcnemar_exact(y, p_tfidf, p_bert)
    verdict = "statistically significant (p < 0.05)" if p < 0.05 else "NOT statistically significant (p >= 0.05)"
    print(f"\nMcNemar exact test on {dataset}: TF-IDF-only-correct={b}, DistilBERT-only-correct={c}, "
          f"p={p:.4f} -> difference is {verdict}.")
    return p


def show_disagreements(texts, y_true, p_tfidf, p_bert, n=5):
    mask = p_tfidf != p_bert
    print(f"\nModels disagree on {mask.sum()} / {len(y_true)} reviews ({100 * mask.mean():.1f}%).")
    names = {0: "neg", 1: "pos"}
    for i in np.where(mask)[0][:n]:
        snippet = texts[i][:220]
        print(f"\n  true={names[y_true[i]]}  tfidf={names[p_tfidf[i]]}  "
              f"pretrained={names[p_bert[i]]}\n  \"{snippet}...\"")


def length_analysis(dataset, texts, y, p_tfidf, p_bert):
    """Accuracy by review length. Tests whether DistilBERT suffers on long (truncated) reviews."""
    words = np.array([len(t.split()) for t in texts])
    edges = np.quantile(words, [1 / 3, 2 / 3])
    bin_id = np.digitize(words, edges)
    labels = [f"short (<={int(edges[0])}w)",
              f"medium ({int(edges[0]) + 1}-{int(edges[1])}w)",
              f"long (>{int(edges[1])}w)"]
    rows = []
    for k, lab in enumerate(labels):
        m = bin_id == k
        if m.sum() == 0:
            continue
        rows.append({"dataset": dataset, "length_bin": lab, "n": int(m.sum()),
                     "tfidf_acc": float((p_tfidf[m] == y[m]).mean()),
                     "pretrained_acc": float((p_bert[m] == y[m]).mean())})
    m = words > TRUNCATION_WORDS
    if m.sum() >= 10:
        rows.append({"dataset": dataset, "length_bin": f"likely truncated (>{TRUNCATION_WORDS}w)",
                     "n": int(m.sum()),
                     "tfidf_acc": float((p_tfidf[m] == y[m]).mean()),
                     "pretrained_acc": float((p_bert[m] == y[m]).mean())})
    df = pd.DataFrame(rows)
    print(f"\nAccuracy by review length on {dataset}:")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("(Small bins are noisy. Differences of a few points on n < 100 are weak evidence.)")
    return df


def save_predictions(dataset, texts, y, p_tfidf, p_bert):
    df = pd.DataFrame({"text": texts, "true": y, "tfidf_pred": p_tfidf, "pretrained_pred": p_bert,
                       "words": [len(t.split()) for t in texts]})
    path = OUT_DIR / f"predictions_{dataset}.csv"
    df.to_csv(path, index=False)
    print(f"Saved {path}")


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def plot_results(metrics_df, length_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = list(metrics_df["dataset"].unique())
    models = list(metrics_df["model"].unique())
    x = np.arange(len(datasets))
    width = 0.8 / len(models)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, title in zip(axes, ["accuracy", "f1", "ms_per_review"],
                              ["Accuracy", "F1 score", "Speed (ms/review, log scale)"]):
        for i, m in enumerate(models):
            vals = [metrics_df[(metrics_df.dataset == d) & (metrics_df.model == m)][col].iloc[0]
                    for d in datasets]
            bars = ax.bar(x + i * width, vals, width, label=m)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}" if col != "ms_per_review" else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(datasets)
        ax.set_title(title)
        if col == "ms_per_review":
            ax.set_yscale("log")
        else:
            ax.set_ylim(0.5, 1.0)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "results.png", dpi=150)
    plt.close(fig)

    if length_df is not None and len(length_df):
        imdb = length_df[length_df.dataset == length_df.dataset.iloc[0]]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        xx = np.arange(len(imdb))
        ax.bar(xx - 0.2, imdb["tfidf_acc"], 0.4, label="TF-IDF + LR")
        ax.bar(xx + 0.2, imdb["pretrained_acc"], 0.4, label="DistilBERT (pretrained)")
        ax.set_xticks(xx)
        ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(imdb["length_bin"], imdb["n"])], fontsize=8)
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy by review length ({imdb.dataset.iloc[0]})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "length_analysis.png", dpi=150)
        plt.close(fig)
    print(f"Saved charts to {OUT_DIR}/results.png and {OUT_DIR}/length_analysis.png")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_on(dataset, texts, y, pipe, bert, all_metrics, length_frames):
    t0 = time.time()
    p_tfidf = pipe.predict(texts)
    all_metrics.append(evaluate(dataset, "TF-IDF + LR", y, p_tfidf, time.time() - t0))

    t0 = time.time()
    p_bert = predict_pretrained(bert, texts)
    all_metrics.append(evaluate(dataset, "DistilBERT (pretrained)", y, p_bert, time.time() - t0))

    compare_models(dataset, y, p_tfidf, p_bert)
    show_disagreements(texts, y, p_tfidf, p_bert)
    length_frames.append(length_analysis(dataset, texts, y, p_tfidf, p_bert))
    save_predictions(dataset, texts, y, p_tfidf, p_bert)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-size", type=int, default=2000, help="IMDB test reviews to evaluate")
    ap.add_argument("--product-size", type=int, default=1000, help="product/Yelp reviews to evaluate")
    ap.add_argument("--product-dataset", choices=["amazon", "yelp"], default="amazon")
    ap.add_argument("--skip-product", action="store_true")
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    ap.add_argument("--retrain", action="store_true", help="retrain TF-IDF even if a saved model exists")
    ap.add_argument("--predict", type=str, help="Classify one review with both models")
    args = ap.parse_args()

    if args.predict:
        text = clean_text(args.predict)
        pipe = joblib.load(MODEL_PATH)
        proba = pipe.predict_proba([text])[0, 1]
        print(f"TF-IDF+LR : {'positive' if proba >= 0.5 else 'negative'} (P(pos)={proba:.3f})")
        out = load_pretrained(args.device)(text, truncation=True, max_length=512)[0]
        print(f"Pretrained: {out['label'].lower()} (score={out['score']:.3f})")
        return

    OUT_DIR.mkdir(exist_ok=True)

    print("Loading IMDB...")
    X_train, y_train, X_test, y_test = load_imdb(args.test_size)

    if Path(MODEL_PATH).exists() and not args.retrain:
        print(f"Loading saved TF-IDF model from {MODEL_PATH} (use --retrain to retrain)")
        pipe = joblib.load(MODEL_PATH)
    else:
        print("Training TF-IDF + Logistic Regression...")
        t0 = time.time()
        pipe = train_tfidf(X_train, y_train)
        print(f"Training time: {time.time() - t0:.1f}s")
    show_top_features(pipe)

    print(f"\nLoading pretrained model on {args.device} (first run downloads ~250 MB)...")
    bert = load_pretrained(args.device)

    all_metrics, length_frames = [], []
    run_on("IMDB", X_test, y_test, pipe, bert, all_metrics, length_frames)

    if not args.skip_product:
        print(f"\nLoading {args.product_dataset} reviews (out-of-domain test)...")
        X_prod, y_prod = load_product(args.product_dataset, args.product_size)
        run_on(args.product_dataset.capitalize(), X_prod, y_prod, pipe, bert, all_metrics, length_frames)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False)
    length_df = pd.concat(length_frames, ignore_index=True)
    length_df.to_csv(OUT_DIR / "length_analysis.csv", index=False)
    print(f"\nSaved {OUT_DIR}/metrics.csv and {OUT_DIR}/length_analysis.csv")
    plot_results(metrics_df, length_df)

    print("\n=== SUMMARY ===")
    print(metrics_df[["dataset", "model", "n", "accuracy", "f1", "ms_per_review"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()

```
