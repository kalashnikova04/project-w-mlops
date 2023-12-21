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
    for k, v in tracked_metrics.items():
        metric_value = v(scores_array, labels_array)
        metric_results[k] = metric_value

    if print_log:
        print(" | ".join(["{}: {:.4f}".format(k, v) for k, v in metric_results.items()]))

    return metric_results


def get_score_distributions(epoch_result_dict: Dict) -> Dict:
    """Return per-class score arrays."""
    scores = epoch_result_dict["scores"]
    labels = epoch_result_dict["labels"]

    # save per-class scores
    for class_id in [0, 1]:
        epoch_result_dict["scores_" + str(class_id)] = np.array(scores)[
            np.array(labels) == class_id
        ]

    return epoch_result_dict
