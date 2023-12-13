import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score


def accuracy(scores, labels, threshold=0.5):
    assert type(scores) is np.ndarray and type(labels) is np.ndarray
    predicted = np.array(scores > threshold).astype(np.int32)
    return np.mean(predicted == labels)


def f1(scores, labels, threshold=0.5):
    assert type(scores) is np.ndarray and type(labels) is np.ndarray
    predicted = np.array(scores > threshold).astype(np.int32)
    return f1_score(labels, predicted)


tracked_metrics = {"accuracy": accuracy, "f1-score": f1}


def calculate_metrics(scores, labels, print_log=False):
    """Compute all the metrics from tracked_metrics dict using scores and labels."""

    assert len(labels) == len(scores), print(
        "Label and score lists are of different size"
    )

    scores_array = np.array(scores).astype(np.float32)
    labels_array = np.array(labels)

    metric_results = {}
    for k, v in tracked_metrics.items():
        metric_value = v(scores_array, labels_array)
        metric_results[k] = metric_value

    if print_log:
        print(" | ".join(["{}: {:.4f}".format(k, v) for k, v in metric_results.items()]))

    return metric_results


def compute_loss(model, data_batch):
    """Compute the loss using loss_function for the batch of data and return mean loss value for this batch."""

    img_batch = data_batch["img"]
    label_batch = data_batch["label"]

    logits = model(img_batch)

    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(logits, label_batch)

    return loss, model


def get_score_distributions(epoch_result_dict):
    """Return per-class score arrays."""
    scores = epoch_result_dict["scores"]
    labels = epoch_result_dict["labels"]

    # save per-class scores
    for class_id in [0, 1]:
        epoch_result_dict["scores_" + str(class_id)] = np.array(scores)[
            np.array(labels) == class_id
        ]

    return epoch_result_dict
