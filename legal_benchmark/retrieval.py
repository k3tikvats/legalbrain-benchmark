from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self, max_features: int = 100_000):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features,
            min_df=1,
            max_df=0.98,
        )
        self.contexts: list[str] = []
        self.matrix = None

    def fit(self, contexts: list[str]) -> None:
        self.contexts = contexts
        self.matrix = self.vectorizer.fit_transform(contexts)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if self.matrix is None:
            raise RuntimeError("Retriever has not been fitted.")
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        if top_k >= len(scores):
            order = np.argsort(-scores)
        else:
            order = np.argpartition(-scores, top_k)[:top_k]
            order = order[np.argsort(-scores[order])]
        return [(int(index), float(scores[index])) for index in order[:top_k]]


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"(?u)\b\w\w+\b", str(text).lower())


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[Counter[str]] = []
        self.idf: dict[str, float] = {}
        self.doc_lens: list[int] = []
        self.avgdl = 0.0

    def fit(self, contexts: list[str]) -> None:
        tokenized = [simple_tokenize(context) for context in contexts]
        self.doc_freqs = [Counter(tokens) for tokens in tokenized]
        self.doc_lens = [len(tokens) for tokens in tokenized]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)

        document_frequency: Counter[str] = Counter()
        for freqs in self.doc_freqs:
            document_frequency.update(freqs.keys())

        corpus_size = len(self.doc_freqs)
        self.idf = {
            term: math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if not self.doc_freqs:
            raise RuntimeError("Retriever has not been fitted.")
        query_terms = simple_tokenize(query)
        scores = np.zeros(len(self.doc_freqs), dtype=np.float32)
        for term in query_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lens[idx] / max(self.avgdl, 1e-9))
                scores[idx] += idf * (tf * (self.k1 + 1)) / denom
        if top_k >= len(scores):
            order = np.argsort(-scores)
        else:
            order = np.argpartition(-scores, top_k)[:top_k]
            order = order[np.argsort(-scores[order])]
        return [(int(index), float(scores[index])) for index in order[:top_k]]


class DenseRetriever:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 64,
    ):
        from sentence_transformers import SentenceTransformer

        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.embeddings = None

    def fit(self, contexts: list[str]) -> None:
        self.embeddings = self.model.encode(
            contexts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if self.embeddings is None:
            raise RuntimeError("Retriever has not been fitted.")
        query_embedding = self.model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = np.dot(self.embeddings, query_embedding)
        if top_k >= len(scores):
            order = np.argsort(-scores)
        else:
            order = np.argpartition(-scores, top_k)[:top_k]
            order = order[np.argsort(-scores[order])]
        return [(int(index), float(scores[index])) for index in order[:top_k]]


def retrieval_metrics(ranks: list[int | None], ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    metrics: dict[str, float] = {}
    total = len(ranks)
    if total == 0:
        return {f"recall_at_{k}": 0.0 for k in ks} | {"mrr": 0.0}

    for k in ks:
        metrics[f"recall_at_{k}"] = sum(rank is not None and rank <= k for rank in ranks) / total

    reciprocal = [0.0 if rank is None else 1.0 / rank for rank in ranks]
    metrics["mrr"] = sum(reciprocal) / total
    return metrics
