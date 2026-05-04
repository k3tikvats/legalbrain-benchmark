import re
import string
from collections import Counter


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_list(text: str) -> list[str]:
    return normalize_text(text).split()


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = token_list(prediction)
    ref_tokens = token_list(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = token_list(prediction)
    ref_tokens = token_list(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    previous = [0] * (len(ref_tokens) + 1)
    for pred_token in pred_tokens:
        current = [0]
        for j, ref_token in enumerate(ref_tokens, start=1):
            if pred_token == ref_token:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current

    lcs = previous[-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def context_support_ratio(answer: str, context: str) -> float:
    answer_tokens = [t for t in token_list(answer) if len(t) > 2]
    if not answer_tokens:
        return 0.0
    context_tokens = set(token_list(context))
    supported = sum(1 for token in answer_tokens if token in context_tokens)
    return supported / len(answer_tokens)


def aggregate_metric_dict(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row if isinstance(row[key], (int, float))})
    return {key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows) for key in keys}


# ---------------------------------------------------------------------------
# NLI Faithfulness (requires sentence-transformers cross-encoder)
# ---------------------------------------------------------------------------

_nli_model = None
_nli_model_name: str = "cross-encoder/nli-deberta-v3-small"


def _get_nli_model():
    """Lazy-load the NLI cross-encoder (≈85 MB, runs on CPU)."""
    global _nli_model
    if _nli_model is None:
        try:
            from sentence_transformers import CrossEncoder
            _nli_model = CrossEncoder(_nli_model_name, max_length=512)
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers>=2.7: pip install sentence-transformers"
            ) from exc
    return _nli_model


def nli_faithfulness_score(answer: str, context: str) -> float:
    """Probability that ``context`` entails ``answer`` according to an NLI model.

    Returns the softmax probability of the *entailment* class from
    ``cross-encoder/nli-deberta-v3-small``.  Label order for this model is
    ``[contradiction, entailment, neutral]``.

    Falls back to the lexical ``context_support_ratio`` if the NLI model is
    not available (e.g. no internet connection).
    """
    if not answer or not context:
        return 0.0
    try:
        import numpy as np
        from scipy.special import softmax  # type: ignore

        model = _get_nli_model()
        # Truncate to prevent exceeding model max length
        ctx_trunc = context[:1500]
        ans_trunc = answer[:256]
        raw = model.predict([(ctx_trunc, ans_trunc)])
        probs = softmax(raw[0])
        # DeBERTa NLI label order: 0=contradiction, 1=entailment, 2=neutral
        return float(probs[1])
    except Exception:
        return context_support_ratio(answer, context)


def batch_nli_faithfulness(answers: list[str], contexts: list[str]) -> list[float]:
    """Batch-compute NLI faithfulness for efficiency."""
    if not answers:
        return []
    try:
        import numpy as np
        from scipy.special import softmax  # type: ignore

        model = _get_nli_model()
        pairs = [(ctx[:1500], ans[:256]) for ctx, ans in zip(contexts, answers)]
        raw = model.predict(pairs)
        probs = softmax(raw, axis=1)
        return [float(p[1]) for p in probs]
    except Exception:
        return [context_support_ratio(a, c) for a, c in zip(answers, contexts)]


# ---------------------------------------------------------------------------
# BERTScore (semantic similarity between prediction and reference)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# InLegalBERT Semantic Answer Similarity
# ---------------------------------------------------------------------------

_inlegal_sim_model = None
_INLEGAL_MODEL_NAME = "law-ai/InLegalBERT"


def _get_inlegal_sim_model():
    """Lazy-load InLegalBERT as a SentenceTransformer for sentence-level embeddings."""
    global _inlegal_sim_model
    if _inlegal_sim_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _inlegal_sim_model = SentenceTransformer(_INLEGAL_MODEL_NAME)
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers: pip install sentence-transformers"
            ) from exc
    return _inlegal_sim_model


def batch_inlegal_sim(
    predictions: list[str],
    references: list[str],
    device: str | None = None,
    batch_size: int = 64,
) -> list[float]:
    """Compute domain-adapted Semantic Answer Similarity (SAS) using InLegalBERT.

    Encodes predictions and references with ``law-ai/InLegalBERT`` (a BERT model
    pre-trained on Indian legal text) and returns the cosine similarity of their
    [CLS] embeddings.  This is a domain-adapted variant of the Semantic Answer
    Similarity metric introduced by Risch et al. (2021), adapted here to the
    Indian legal domain.

    Falls back to ``token_f1`` if sentence-transformers is unavailable.
    """
    if not predictions:
        return []
    try:
        import torch
        import torch.nn.functional as F

        model = _get_inlegal_sim_model()
        if device is not None:
            model = model.to(device)
        pred_embs = model.encode(
            predictions, batch_size=batch_size, convert_to_tensor=True,
            show_progress_bar=False, normalize_embeddings=True,
        )
        ref_embs = model.encode(
            references, batch_size=batch_size, convert_to_tensor=True,
            show_progress_bar=False, normalize_embeddings=True,
        )
        # cosine similarity is dot product since embeddings are L2-normalised
        scores = (pred_embs * ref_embs).sum(dim=-1)
        return [float(s) for s in scores.cpu().tolist()]
    except Exception:
        return [token_f1(p, r) for p, r in zip(predictions, references)]


def batch_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "roberta-large",
    device: str | None = None,
    batch_size: int = 64,
) -> list[float]:
    """Compute BERTScore F1 between each prediction and reference.

    Uses ``roberta-large`` by default, which correlates well with human
    judgements and is available without API access.  Returns per-example F1
    scores in [0, 1]; higher is more semantically similar.

    Falls back to ``token_f1`` if ``bert_score`` is not installed.
    """
    if not predictions:
        return []
    try:
        import torch
        from bert_score import score as _bs_score

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _, _, F1 = _bs_score(
            predictions,
            references,
            model_type=model_type,
            device=device,
            batch_size=batch_size,
            verbose=False,
            lang="en",
        )
        return [float(f) for f in F1.tolist()]
    except Exception:
        return [token_f1(p, r) for p, r in zip(predictions, references)]

