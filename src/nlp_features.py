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
from typing import Sequence

import numpy as np
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
    # Regex-based features (fast, no model needed)
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
    def vocab_richness(text: str) -> float:
        if not text:
            return 0.0
        tokens = _WORD_TOKEN.findall(text.lower())
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

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
        try:
            result = self._sentiment(text[:512])
            scores = {r["label"].lower(): r["score"] for r in result[0]}
            pos = scores.get("positive", scores.get("pos", 0.0))
            neg = scores.get("negative", scores.get("neg", 0.0))
            return pos - neg
        except Exception:
            return 0.0

    def batch_sentiment(self, texts: list[str], batch_size: int = 32) -> list[float]:
        """Compute sentiment for a list of texts efficiently."""
        results: list[float] = []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size),
                      total=n_batches, desc="Sentiment", unit="batch"):
            batch = [t[:512] if t else "" for t in texts[i : i + batch_size]]
            batch = [t if t.strip() else "neutral" for t in batch]
            try:
                preds = self._sentiment(batch)
                for pred in preds:
                    scores = {r["label"].lower(): r["score"] for r in pred}
                    pos = scores.get("positive", scores.get("pos", 0.0))
                    neg = scores.get("negative", scores.get("neg", 0.0))
                    results.append(pos - neg)
            except Exception:
                results.extend([0.0] * len(batch))
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

        try:
            result = self._zeroshot(text[:512], candidate_labels=labels)
            probs = dict(zip(result["labels"], result["scores"]))
            return {
                key: probs.get(label, 0.0)
                for key, label in zip(TOPIC_KEYS, labels)
            }
        except Exception:
            return {k: 0.0 for k in TOPIC_KEYS}

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
            try:
                preds = self._zeroshot(batch, candidate_labels=labels)
                if isinstance(preds, dict):
                    preds = [preds]
                for pred in preds:
                    probs = dict(zip(pred["labels"], pred["scores"]))
                    results.append({
                        key: probs.get(label, 0.0)
                        for key, label in zip(TOPIC_KEYS, labels)
                    })
            except Exception:
                results.extend([{k: 0.0 for k in TOPIC_KEYS}] * len(batch))
        return results

    # ------------------------------------------------------------------
    # Combined feature extraction
    # ------------------------------------------------------------------

    def extract_text_features(self, text: str) -> dict:
        """Extract all features for a single text.

        Returns a dict with all linguistic, sentiment, and topic features.
        Does NOT include zero-shot topics (use zero_shot_topics separately
        for batch efficiency).
        """
        wc = self.word_count(text)
        uw = self.unique_words(text)
        sc = self.sentence_count(text)

        return {
            "word_count": wc,
            "unique_words": uw,
            "sentence_count": sc,
            "avg_sentence_length": wc / sc if sc > 0 else 0.0,
            "vocab_richness": uw / wc if wc > 0 else 0.0,
            "self_reference_ratio": self.self_reference_ratio(text),
            "future_orientation": self.future_orientation(text),
        }

    def extract_all_features(self, text: str) -> dict:
        """Extract ALL features including sentiment (but not zero-shot topics).

        For batch processing, prefer batch_sentiment + extract_text_features
        separately for better GPU utilisation.
        """
        feats = self.extract_text_features(text)
        feats["compound_score"] = self.sentiment(text)
        return feats
