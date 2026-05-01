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
