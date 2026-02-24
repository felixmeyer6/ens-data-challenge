"""Evaluation metrics for the HRFlow challenge."""

from typing import Iterable, Sequence


def _to_list(values: Iterable):
    return list(values) if values is not None else []


def mrr_at_k(y_true: Sequence, y_pred: Sequence[Sequence], k: int = 10) -> float:
    """Compute Mean Reciprocal Rank at K."""
    y_true_list = _to_list(y_true)
    y_pred_list = _to_list(y_pred)

    if not y_true_list:
        return 0.0

    reciprocal_ranks = []
    for truth, preds in zip(y_true_list, y_pred_list):
        truth = str(truth)
        preds = [str(x) for x in _to_list(preds)[:k]]

        rr = 0.0
        for rank, pred in enumerate(preds, start=1):
            if pred == truth:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    if not reciprocal_ranks:
        return 0.0
    return float(sum(reciprocal_ranks) / len(reciprocal_ranks))


def mrr_at_10(y_true: Sequence, y_pred: Sequence[Sequence]) -> float:
    """Compute Mean Reciprocal Rank at 10."""
    return mrr_at_k(y_true=y_true, y_pred=y_pred, k=10)


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    """Compute classification accuracy."""
    y_true_list = _to_list(y_true)
    y_pred_list = _to_list(y_pred)

    if not y_true_list:
        return 0.0

    correct = 0
    for truth, pred in zip(y_true_list, y_pred_list):
        if str(truth) == str(pred):
            correct += 1

    return float(correct / len(y_true_list))


def challenge_score(mrr: float, acc: float) -> float:
    """Compute final challenge score = 0.7 * MRR + 0.3 * ACC."""
    return float(0.7 * mrr + 0.3 * acc)
