from collections import Counter
from math import exp, log

import numpy as np


def simple_bleu(predictions, references, max_order=4):
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    prediction_length = 0
    reference_length = 0

    for prediction, reference in zip(predictions, references):
        pred_tokens = prediction.split()
        ref_tokens = reference.split()
        prediction_length += len(pred_tokens)
        reference_length += len(ref_tokens)

        merged_ref_ngram_counts = Counter()
        ref_ngram_counts = _get_ngrams(ref_tokens, max_order)
        merged_ref_ngram_counts |= ref_ngram_counts
        pred_ngram_counts = _get_ngrams(pred_tokens, max_order)
        overlap = pred_ngram_counts & merged_ref_ngram_counts

        for ngram, count in overlap.items():
            matches_by_order[len(ngram) - 1] += count
        for order in range(1, max_order + 1):
            possible_matches = len(pred_tokens) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0.0] * max_order
    for index in range(max_order):
        if possible_matches_by_order[index] > 0:
            precisions[index] = (
                matches_by_order[index] + 1.0
            ) / (possible_matches_by_order[index] + 1.0)

    if min(precisions) > 0:
        geo_mean = exp(sum(log(precision) for precision in precisions) / max_order)
    else:
        geo_mean = 0.0

    if prediction_length == 0:
        return 0.0

    ratio = prediction_length / max(reference_length, 1)
    brevity_penalty = 1.0 if ratio > 1.0 else exp(1.0 - 1.0 / max(ratio, 1e-8))
    return float(geo_mean * brevity_penalty)


def exact_match(predictions, references):
    if not predictions:
        return 0.0
    correct = sum(pred.strip() == ref.strip() for pred, ref in zip(predictions, references))
    return correct / len(predictions)


def mean_per_joint_position_error(predicted, target, mask=None):
    predicted = np.asarray(predicted, dtype=np.float32).reshape(*predicted.shape[:-1], -1, 2)
    target = np.asarray(target, dtype=np.float32).reshape(*target.shape[:-1], -1, 2)
    distances = np.linalg.norm(predicted - target, axis=-1)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        distances = distances[mask]
    if distances.size == 0:
        return 0.0
    return float(distances.mean())


def keypoint_accuracy(predicted, target, threshold=0.05, mask=None):
    predicted = np.asarray(predicted, dtype=np.float32).reshape(*predicted.shape[:-1], -1, 2)
    target = np.asarray(target, dtype=np.float32).reshape(*target.shape[:-1], -1, 2)
    distances = np.linalg.norm(predicted - target, axis=-1)
    correct = distances < threshold
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        correct = correct[mask]
    if correct.size == 0:
        return 0.0
    return float(correct.mean())


def _get_ngrams(tokens, max_order):
    ngram_counts = Counter()
    for order in range(1, max_order + 1):
        for index in range(0, len(tokens) - order + 1):
            ngram = tuple(tokens[index:index + order])
            ngram_counts[ngram] += 1
    return ngram_counts

