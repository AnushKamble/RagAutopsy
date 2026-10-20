"""Standard question-answering evaluation metrics: Exact Match (EM) and Token F1.

Follows official SQuAD / HotpotQA text normalization conventions.
"""

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Normalize text for answer comparison: lowercase, remove punctuation, articles, and extra whitespace."""
    if not s:
        return ""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    """Check if normalized prediction matches normalized ground truth exactly."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)


def evaluate_answer_correctness(
    prediction: str, gold_answers: list[str]
) -> tuple[bool, float]:
    """Evaluate prediction against multiple reference answers (taking max over references).
    
    Returns:
        (exact_match, token_f1)
    """
    if not gold_answers:
        return False, 0.0

    em = any(compute_exact_match(prediction, gold) for gold in gold_answers)
    f1 = max(compute_token_f1(prediction, gold) for gold in gold_answers)
    return em, f1
