from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import f1_score


def accuracy(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    predicted = np.array(scores > threshold).astype(np.int32)
    return np.mean(predicted == labels)


def f1(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    predicted = np.array(scores > threshold).astype(np.int32)
    return f1_score(labels, predicted)


def calculate_metrics(
    scores: List[torch.Tensor], labels: List[np.ndarray], print_log: bool = True
) -> Dict:
    """Compute all the metrics from tracked_metrics dict using scores and labels."""

    assert len(labels) == len(scores), print(
        "Label and score lists are of different size"
    )

    scores_array = np.array(scores).astype(np.float32)
    labels_array = np.array(labels)

    tracked_metrics = {"accuracy": accuracy, "f1-score": f1}

    metric_results = {}
    for key, value in tracked_metrics.items():
        metric_value = value(scores_array, labels_array)
        metric_results[key] = metric_value

    if print_log:
        print(
            " | ".join(
                ["{}: {:.4f}".format(key, value) for key, value in metric_results.items()]
            )
        )

    return metric_results
