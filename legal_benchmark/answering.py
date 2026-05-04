from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

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


@dataclass
class CausalLMAnswerer:
    """Decoder-only causal LM answerer (TinyLlama, Phi-3, LLaMA-3, etc.).

    Supports 4-bit/8-bit quantization via ``bitsandbytes`` for running on
    consumer GPUs (e.g. RTX 3050 6 GB).  Pass ``load_in_4bit=True`` to
    enable BitsAndBytes NF4 quantisation.
    """

    model_name: str
    device: str = "auto"
    max_context_chars: int = 3000
    max_new_tokens: int = 128
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    def __post_init__(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        bnb_config = None
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device == "cuda" and not (self.load_in_4bit or self.load_in_8bit) else None
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            quantization_config=bnb_config,
            device_map=self.device if (self.load_in_4bit or self.load_in_8bit) else None,
        )
        if not (self.load_in_4bit or self.load_in_8bit):
            self.model.to(self.device)
        self.model.eval()

    def _prompt(self, question: str, context: str) -> str:
        context = truncate_text(context, self.max_context_chars)
        return (
            "<|system|>\nYou are a legal assistant. Answer the question using only the provided context. "
            "If the answer cannot be determined from the context, respond with 'The context does not provide enough information.'\n"
            f"<|user|>\nContext:\n{context}\n\nQuestion: {question}\n<|assistant|>\nAnswer:"
        )

    @torch.inference_mode()
    def answer(self, question: str, context: str) -> str:
        prompt = self._prompt(question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@dataclass
class GroqAnswerer:
    """Answerer backed by the Groq free-tier API (Llama 3.1 8B / 70B).

    Requires the ``groq`` Python package and a ``GROQ_API_KEY`` environment
    variable.  The free tier allows ~14,400 requests/day at zero cost.

    Recommended free models:
    - ``llama-3.1-8b-instant``  (fast, low latency)
    - ``llama-3.3-70b-versatile``  (stronger, still free-tier)
    - ``gemma2-9b-it``  (Google Gemma 2)
    """

    model_name: str = "llama-3.1-8b-instant"
    max_context_chars: int = 3000
    max_new_tokens: int = 256
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        try:
            from groq import Groq  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Install the groq package: pip install groq"
            ) from exc

        key = self.api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "Set GROQ_API_KEY environment variable or pass api_key= to GroqAnswerer."
            )
        self._client = Groq(api_key=key)

    def _prompt(self, question: str, context: str) -> list[dict]:
        context = truncate_text(context, self.max_context_chars)
        system = (
            "You are an expert Indian legal assistant. "
            "Answer questions strictly using the provided context. "
            "If the answer is not in the context, say 'The context does not provide enough information.'"
        )
        user = f"Context:\n{context}\n\nQuestion: {question}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def answer(self, question: str, context: str) -> str:
        from groq import Groq  # type: ignore

        messages = self._prompt(question, context)
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

