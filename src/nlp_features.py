"""
NLP feature extraction for textual engagement analysis.

Provides NLPFeatureExtractor which combines:
- Regex-based linguistic features (self-reference, future orientation)
- BERT-based sentiment analysis
- BERT-based zero-shot topic classification
- Basic text metrics (word count, vocab richness, sentence length)

Models are loaded once and reused across all texts.
GPU acceleration via CUDA when available.
"""

from __future__ import annotations

import re

import torch
from tqdm import tqdm
from transformers import pipeline


# ---------------------------------------------------------------------------
# Regex patterns for linguistic features
# ---------------------------------------------------------------------------

_SELF_REF = re.compile(
    r"\b(i|me|my|mine|myself|i'm|i've|i'd|i'll)\b", re.IGNORECASE
)
_FUTURE = re.compile(
    r"\b(will|going to|plan|plans|planning|hope|hoping|aim|aiming|"
    r"intend|intending|want to|would like|goal|goals|future|tomorrow|"
    r"next week|next month|soon)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_WORD_TOKEN = re.compile(r"[a-zA-Z]+")


# ---------------------------------------------------------------------------
# Default topic labels for zero-shot classification
# ---------------------------------------------------------------------------

DEFAULT_TOPICS = [
    "hope and optimism",
    "anxiety and worry",
    "gratitude and appreciation",
    "struggle and difficulty",
    "social connection and relationships",
]

# Short keys used as column names
TOPIC_KEYS = [
    "topic_hope",
    "topic_anxiety",
    "topic_gratitude",
    "topic_struggle",
    "topic_social",
]


class NLPFeatureExtractor:
    """Extract NLP features from text using BERT models and regex patterns.

    Parameters
    ----------
    device : str or None
        PyTorch device. None = auto-detect (CUDA if available).
    sentiment_model : str
        HuggingFace model ID for sentiment analysis.
    zeroshot_model : str
        HuggingFace model ID for zero-shot classification.
    """

    def __init__(
        self,
        device: str | None = None,
        sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        zeroshot_model: str = "facebook/bart-large-mnli",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"NLPFeatureExtractor: using device={device}")

        print(f"  Loading sentiment model: {sentiment_model}")
        self._sentiment = pipeline(
            "sentiment-analysis",
            model=sentiment_model,
            device=device,
            truncation=True,
            max_length=512,
            top_k=None,
        )

        print(f"  Loading zero-shot model: {zeroshot_model}")
        self._zeroshot = pipeline(
            "zero-shot-classification",
            model=zeroshot_model,
            device=device,
        )
        print("  Models loaded.")

    # ------------------------------------------------------------------
    # Regex-based features
    # ------------------------------------------------------------------

    @staticmethod
    def word_count(text: str) -> int:
        if not text:
            return 0
        return len(text.split())

    @staticmethod
    def unique_words(text: str) -> int:
        if not text:
            return 0
        tokens = _WORD_TOKEN.findall(text.lower())
        return len(set(tokens))

    @staticmethod
    def sentence_count(text: str) -> int:
        if not text or not text.strip():
            return 0
        parts = _SENTENCE_SPLIT.split(text.strip())
        parts = [p for p in parts if p.strip()]
        return max(1, len(parts))

    @staticmethod
    def self_reference_ratio(text: str) -> float:
        if not text:
            return 0.0
        words = text.split()
        if not words:
            return 0.0
        matches = _SELF_REF.findall(text)
        return len(matches) / len(words)

    @staticmethod
    def future_orientation(text: str) -> float:
        if not text:
            return 0.0
        words = text.split()
        if not words:
            return 0.0
        matches = _FUTURE.findall(text)
        return len(matches) / len(words)

    # ------------------------------------------------------------------
    # BERT sentiment
    # ------------------------------------------------------------------

    def sentiment(self, text: str) -> float:
        """Return a compound sentiment score in [-1, 1].

        Maps the 3-class RoBERTa output (negative/neutral/positive)
        to a continuous score: score = P(positive) - P(negative).
        """
        if not text or not text.strip():
            return 0.0
        result = self._sentiment(text[:512])
        # result is a list of dicts with 'label' and 'score' keys
        scores = {r["label"].lower(): r["score"] for r in result}  # type: ignore
        pos = scores.get("positive", scores.get("pos", 0.0))
        neg = scores.get("negative", scores.get("neg", 0.0))
        return float(pos - neg)

    def batch_sentiment(self, texts: list[str], batch_size: int = 32) -> list[float]:
        """Compute sentiment for a list of texts efficiently."""
        results: list[float] = []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size),
                      total=n_batches, desc="Sentiment", unit="batch"):
            batch = [t[:512] if t else "" for t in texts[i : i + batch_size]]
            batch = [t if t.strip() else "neutral" for t in batch]
            preds = self._sentiment(batch)
            for pred in preds:  # type: ignore
                if isinstance(pred, dict):
                    # Single prediction is a dict with 'label' and 'score'
                    scores = {pred["label"].lower(): pred["score"]}
                else:
                    # If list of dicts
                    scores = {r["label"].lower(): r["score"] for r in pred}  # type: ignore
                pos = scores.get("positive", scores.get("pos", 0.0))
                neg = scores.get("negative", scores.get("neg", 0.0))
                results.append(float(pos - neg))
        return results

    # ------------------------------------------------------------------
    # BERT zero-shot topic classification
    # ------------------------------------------------------------------

    def zero_shot_topics(
        self,
        text: str,
        labels: list[str] | None = None,
    ) -> dict[str, float]:
        """Return topic probabilities via zero-shot classification.

        Returns dict mapping TOPIC_KEYS to probabilities.
        """
        if labels is None:
            labels = DEFAULT_TOPICS

        if not text or not text.strip():
            return {k: 0.0 for k in TOPIC_KEYS}

        result = self._zeroshot(text[:512], candidate_labels=labels)
        if isinstance(result, list):
            result = result[0]  # type: ignore
        # result has 'labels' and 'scores' keys
        result_dict = result if isinstance(result, dict) else {}  # type: ignore
        labels_list = result_dict.get("labels", [])  # type: ignore
        scores_list = result_dict.get("scores", [])  # type: ignore
        probs = dict(zip(labels_list, scores_list))
        return {
            key: probs.get(label, 0.0)
            for key, label in zip(TOPIC_KEYS, labels)
        }

    def batch_zero_shot(
        self,
        texts: list[str],
        labels: list[str] | None = None,
        batch_size: int = 8,
    ) -> list[dict[str, float]]:
        """Compute zero-shot topics for a list of texts."""
        if labels is None:
            labels = DEFAULT_TOPICS

        results: list[dict[str, float]] = []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size),
                      total=n_batches, desc="Zero-shot topics", unit="batch"):
            batch = [t[:512] if t and t.strip() else "none" for t in texts[i : i + batch_size]]
            preds = self._zeroshot(batch, candidate_labels=labels)
            if isinstance(preds, dict):
                preds = [preds]  # type: ignore
            for pred in preds:  # type: ignore
                pred_dict = pred if isinstance(pred, dict) else {}  # type: ignore
                pred_labels = pred_dict.get("labels", [])  # type: ignore
                pred_scores = pred_dict.get("scores", [])  # type: ignore
                probs = dict(zip(pred_labels, pred_scores))
                results.append({
                    key: probs.get(label, 0.0)
                    for key, label in zip(TOPIC_KEYS, labels)
                })
        return results

