import torch
import torch.nn as nn
from utils import calculate_metrics

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()  # we do not need to save gradients on evaluation
def test_model(
    model,
    batch_generator,
    subset_name="test",
    print_log=True,
    loss_function=nn.CrossEntropyLoss,
    plot_scores=False,
):
    """Evaluate the model using data from batch_generator and metrics defined above."""

    # disable dropout / use averages for batch_norm
    model.train(False)

    # save scores, labels and loss values for performance logging
    score_list = []
    label_list = []
    loss_list = []

    for X_batch, y_batch in batch_generator:

        # do the forward pass
        logits = model(X_batch.to(device))
        scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
        labels = y_batch.numpy().tolist()

        # compute loss value
        loss = loss_function(logits, y_batch.to(device))

        # save the necessary data
        loss_list.append(loss.detach().cpu().numpy().tolist())
        score_list.extend(scores)
        label_list.extend(labels)

    if print_log:
        print("Results on {} set | ".format(subset_name), end="")

    metric_results = calculate_metrics(score_list, label_list, print_log)
    metric_results["scores"] = score_list
    metric_results["labels"] = label_list
    metric_results["loss"] = loss_list

    return metric_results
