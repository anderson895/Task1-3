"""
Part 3 - Task 2: Rule-Based Sentiment Analysis on Tech Tweets
Dataset: tech_tweets.csv  (columns: tweet_id, company, tweet_text, sentiment)

Pipeline:
  1. Data preprocessing - clean tweets (lowercase; strip mentions, hashtags,
     URLs, punctuation)
  2. Sentiment analysis with rule-based tools (TextBlob + VADER):
        polarity > 0 -> positive,  < 0 -> negative,  == 0 -> neutral
  3. Compare predictions to the ground-truth `sentiment` column:
        Accuracy + Confusion Matrix (Seaborn heatmap)
  4. Interpretation (in the responses document)
All figures saved next to this script.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

HERE = os.path.dirname(os.path.abspath(__file__))
sns.set_theme(style="whitegrid")
LABELS = ["negative", "neutral", "positive"]


def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # URLs
    text = re.sub(r"@\w+", " ", text)                # mentions
    text = re.sub(r"#", " ", text)                   # hashtag symbol (keep word)
    text = re.sub(r"[^a-z\s]", " ", text)            # punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text


def label_from_polarity(p):
    if p > 0:
        return "positive"
    if p < 0:
        return "negative"
    return "neutral"


def main():
    df = pd.read_csv(os.path.join(HERE, "tech_tweets.csv"))
    print("Shape:", df.shape)
    print("\nGround-truth distribution:\n", df["sentiment"].value_counts())

    # ---------------------------------------------------- 1. preprocessing
    df["clean_text"] = df["tweet_text"].apply(clean_tweet)
    print("\nExample cleaning:")
    for i in range(3):
        print(f"  raw : {df['tweet_text'].iloc[i]}")
        print(f"  clean: {df['clean_text'].iloc[i]}")

    # ---------------------------------------------- 2. rule-based scoring
    # TextBlob polarity
    df["tb_polarity"] = df["clean_text"].apply(lambda t: TextBlob(t).sentiment.polarity)
    df["tb_pred"] = df["tb_polarity"].apply(label_from_polarity)

    # VADER compound score
    vader = SentimentIntensityAnalyzer()
    df["vader_compound"] = df["clean_text"].apply(
        lambda t: vader.polarity_scores(t)["compound"])
    # VADER's standard thresholds (>=0.05 pos, <=-0.05 neg, else neutral)
    df["vader_pred"] = df["vader_compound"].apply(
        lambda c: "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral"))

    # ------------------------------------------- 3. compare to ground truth
    results = {}
    for name, col in [("TextBlob", "tb_pred"), ("VADER", "vader_pred")]:
        acc = accuracy_score(df["sentiment"], df[col])
        results[name] = acc
        print(f"\n===== {name} =====")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(df["sentiment"], df[col],
                                    labels=LABELS, zero_division=0))

        cm = confusion_matrix(df["sentiment"], df[col], labels=LABELS)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=LABELS, yticklabels=LABELS, ax=ax)
        ax.set_title(f"{name} confusion matrix (acc={acc:.2f})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, f"fig_cm_{name.lower()}.png"), dpi=120)
        plt.close(fig)

    print("\nAccuracy summary:", {k: round(v, 4) for k, v in results.items()})

    # ----------------------------------- examples of wrong TextBlob predictions
    wrong = df[df["tb_pred"] != df["sentiment"]]
    print(f"\nTextBlob misclassified {len(wrong)} / {len(df)} tweets. Examples:")
    for _, r in wrong.head(8).iterrows():
        print(f"  actual={r['sentiment']:8s} pred={r['tb_pred']:8s} "
              f"pol={r['tb_polarity']:+.2f} | {r['tweet_text'][:70]}")

    df.to_csv(os.path.join(HERE, "tweets_sentiment_predictions.csv"), index=False)
    print("\nSaved figures + tweets_sentiment_predictions.csv")


if __name__ == "__main__":
    main()
