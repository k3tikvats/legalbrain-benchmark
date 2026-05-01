from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def truncate_text(text: str, max_chars: int) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= max_chars else text[:max_chars]


class ExtractiveSentenceAnswerer:
    def answer(self, question: str, context: str) -> str:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
        if not sentences:
            return ""
        if len(sentences) == 1:
            return sentences[0]

        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(sentences + [question])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        return sentences[int(scores.argmax())]


@dataclass
class Seq2SeqAnswerer:
    model_name: str
    device: str = "auto"
    max_context_chars: int = 3500
    max_new_tokens: int = 128

    def __post_init__(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

    def _prompt(self, question: str, context: str) -> str:
        context = truncate_text(context, self.max_context_chars)
        return (
            "Answer the legal question using only the context. "
            "If the answer is not in the context, say that the context does not provide enough information.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

    @torch.inference_mode()
    def answer(self, question: str, context: str) -> str:
        prompt = self._prompt(question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
