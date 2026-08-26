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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


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
# Single word tokenizer shared by ALL count/ratio features so that
# numerators and denominators are computed over the same token stream
# (e.g. type-token ratio is guaranteed <= 1). Contractions ("i'm",
# "don't") are kept as one token.
_WORD_TOKEN = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)*")


def _tokenize(text: str) -> list[str]:
    """Lowercased word tokens used by every count/ratio feature."""
    if not text:
        return []
    return _WORD_TOKEN.findall(text.lower())


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
        sentiment_tokenizer = AutoTokenizer.from_pretrained(
            sentiment_model,
            local_files_only=True,
        )
        sentiment_classifier = AutoModelForSequenceClassification.from_pretrained(
            sentiment_model,
            local_files_only=True,
        )
        self._sentiment = pipeline(
            "sentiment-analysis",
            model=sentiment_classifier,
            tokenizer=sentiment_tokenizer,
            device=device,
            truncation=True,
            max_length=512,
            top_k=None,
        )

        # The zero-shot model is only needed for topic features on a full NLP
        # rebuild, so it is loaded on first use rather than here.
        self._zeroshot_model_name = zeroshot_model
        self._zeroshot = None
        print("  Sentiment model loaded.")

    @property
    def zeroshot(self):
        """Zero-shot pipeline, loaded on first access."""
        if self._zeroshot is None:
            print(f"  Loading zero-shot model: {self._zeroshot_model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                self._zeroshot_model_name, local_files_only=True,
            )
            classifier = AutoModelForSequenceClassification.from_pretrained(
                self._zeroshot_model_name, local_files_only=True,
            )
            self._zeroshot = pipeline(
                "zero-shot-classification",
                model=classifier, tokenizer=tokenizer, device=self.device,
            )
        return self._zeroshot

    # ------------------------------------------------------------------
    # Regex-based features
    # ------------------------------------------------------------------

    @staticmethod
    def word_count(text: str) -> int:
        return len(_tokenize(text))

    @staticmethod
    def unique_words(text: str) -> int:
        return len(set(_tokenize(text)))

    @staticmethod
    def sentence_count(text: str) -> int:
        if not text or not text.strip():
            return 0
        parts = _SENTENCE_SPLIT.split(text.strip())
        parts = [p for p in parts if p.strip()]
        return max(1, len(parts))

    @staticmethod
    def self_reference_ratio(text: str) -> float:
        tokens = _tokenize(text)
        if not tokens:
            return 0.0
        matches = _SELF_REF.findall(text)
        return len(matches) / len(tokens)

    @staticmethod
    def future_orientation(text: str) -> float:
        tokens = _tokenize(text)
        if not tokens:
            return 0.0
        matches = _FUTURE.findall(text)
        return len(matches) / len(tokens)

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
        # Token-level truncation to the model's 512-token limit is
        # handled by the pipeline (truncation=True, max_length=512).
        result = self._sentiment(text)
        # result is a list of dicts with 'label' and 'score' keys
        scores = {r["label"].lower(): r["score"] for r in result}  # type: ignore
        pos = scores.get("positive", scores.get("pos", 0.0))
        neg = scores.get("negative", scores.get("neg", 0.0))
        return float(pos - neg)

    def batch_sentiment(self, texts: list[str], batch_size: int = 32) -> list[float]:
        """Compute sentiment for a list of texts efficiently.

        Matches the single-text `sentiment()` contract: empty or
        whitespace-only inputs receive a score of 0.0 without being
        passed through the model (avoids biasing aggregates toward
        the sentiment of a placeholder string).
        """
        # Token-level truncation is handled by the pipeline
        # (truncation=True, max_length=512).
        cleaned = [t if t else "" for t in texts]
        results: list[float] = [0.0] * len(cleaned)

        nonempty_idx = [i for i, t in enumerate(cleaned) if t.strip()]
        if not nonempty_idx:
            return results

        nonempty_texts = [cleaned[i] for i in nonempty_idx]
        n_batches = (len(nonempty_texts) + batch_size - 1) // batch_size
        scored: list[float] = []
        for i in tqdm(range(0, len(nonempty_texts), batch_size),
                      total=n_batches, desc="Sentiment", unit="batch"):
            batch = nonempty_texts[i : i + batch_size]
            preds = self._sentiment(batch)
            for pred in preds:  # type: ignore
                if isinstance(pred, dict):
                    scores = {pred["label"].lower(): pred["score"]}
                else:
                    scores = {r["label"].lower(): r["score"] for r in pred}  # type: ignore
                pos = scores.get("positive", scores.get("pos", 0.0))
                neg = scores.get("negative", scores.get("neg", 0.0))
                scored.append(float(pos - neg))

        for idx, score in zip(nonempty_idx, scored):
            results[idx] = score
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

        # The zero-shot pipeline truncates the premise at the model's
        # token limit (truncation="only_first") — no char slicing needed.
        result = self.zeroshot(text, candidate_labels=labels)
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
        """Compute zero-shot topics for a list of texts.

        Matches the single-text `zero_shot_topics()` contract: empty
        or whitespace-only inputs receive all-zero topic probabilities
        without being passed through the model.
        """
        if labels is None:
            labels = DEFAULT_TOPICS

        empty_result = {k: 0.0 for k in TOPIC_KEYS}
        results: list[dict[str, float]] = [dict(empty_result) for _ in texts]

        # Token-level truncation handled by the zero-shot pipeline.
        cleaned = [t if t else "" for t in texts]
        nonempty_idx = [i for i, t in enumerate(cleaned) if t.strip()]
        if not nonempty_idx:
            return results

        nonempty_texts = [cleaned[i] for i in nonempty_idx]
        n_batches = (len(nonempty_texts) + batch_size - 1) // batch_size
        scored: list[dict[str, float]] = []
        for i in tqdm(range(0, len(nonempty_texts), batch_size),
                      total=n_batches, desc="Zero-shot topics", unit="batch"):
            batch = nonempty_texts[i : i + batch_size]
            preds = self.zeroshot(batch, candidate_labels=labels)
            if isinstance(preds, dict):
                preds = [preds]  # type: ignore
            for pred in preds:  # type: ignore
                pred_dict = pred if isinstance(pred, dict) else {}  # type: ignore
                pred_labels = pred_dict.get("labels", [])  # type: ignore
                pred_scores = pred_dict.get("scores", [])  # type: ignore
                probs = dict(zip(pred_labels, pred_scores))
                scored.append({
                    key: probs.get(label, 0.0)
                    for key, label in zip(TOPIC_KEYS, labels)
                })

        for idx, topic_scores in zip(nonempty_idx, scored):
            results[idx] = topic_scores
        return results

