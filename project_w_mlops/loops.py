import time
from pathlib import Path
from typing import Any, Dict, Tuple

import mlflow
import numpy as np
import torch
from tqdm.auto import tqdm

from .utils import calculate_metrics


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    model.train(True)
    running_loss = 0.0
    score_list, label_list = [], []

    optimizer.zero_grad()

    for batch_idx, (X_batch, y_batch) in enumerate(
        tqdm(data_loader, desc="Training", leave=False)
    ):

        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        logits = model(X_batch)
        scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu()

        loss = loss_fn(logits, y_batch)

        loss.backward()

        optimizer.step()

        score_list.extend(scores)
        label_list.extend(y_batch.numpy().tolist())

        step = epoch * len(data_loader) + batch_idx

        mlflow.log_metric("train_loss", loss.item(), step=step)

        running_loss += loss.item()

    metric_results = calculate_metrics(score_list, label_list)
    mlflow.log_metric("train_accuracy", metric_results["accuracy"], step=step)
    mlflow.log_metric("train_f1-score", metric_results["f1-score"], step=step)

    epoch_loss = running_loss / len(data_loader)
    return epoch_loss


def validate_one_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, Dict]:
    model.eval()
    running_loss = 0.0
    score_list, label_list = [], []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(data_loader, desc="Validating"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)
            scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu()

            loss = loss_fn(logits, y_batch)

            score_list.extend(scores)
            label_list.extend(y_batch.numpy().tolist())

            running_loss += loss.item()

    epoch_loss = running_loss / len(data_loader)
    metric_results = calculate_metrics(score_list, label_list)
    step = epoch * len(data_loader)

    mlflow.log_metric("val_loss", epoch_loss, step=step)
    mlflow.log_metric("val_accuracy", metric_results["accuracy"], step=step)
    mlflow.log_metric("val_f1-score", metric_results["f1-score"], step=step)

    return epoch_loss, metric_results


def train_model(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader[Any],
    val_dataloader: torch.utils.data.DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn,
    n_epochs: int,
    batch_size: int,
    model_path: str,
    ckpt_name: str = None,
):

    top_val_accuracy = 0

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
        )

        val_loss, val_metric_results = validate_one_epoch(
            model=model,
            data_loader=val_dataloader,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
        )

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        print(
            "Epoch {} of {} took {:.3f}s".format(
                epoch + 1, n_epochs, time.time() - start_time
            )
        )

        val_accuracy_value = val_metric_results["accuracy"]

        if val_accuracy_value > top_val_accuracy and ckpt_name is not None:
            top_val_accuracy = val_accuracy_value
            model.eval()

            torch_input = torch.randn(batch_size, 3, 96, 96)
            torch.onnx.export(
                model,
                torch_input,
                Path(model_path, f"{ckpt_name}.onnx"),
                export_params=True,
                opset_version=10,
                do_constant_folding=True,
                input_names=["modelInput"],
                output_names=["modelOutput"],
                dynamic_axes={
                    "modelInput": {0: "batch_size"},
                    "modelOutput": {0: "batch_size"},
                },
            )


def predict(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    root_path_for_preds: str,
    file: str,
):
    model.eval()
    score_list, label_list, predicts = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(data_loader, desc="Testing"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)
            scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu()

            score_list.extend(scores)
            label_list.extend(y_batch.numpy().tolist())
            predicts.extend(logits.max(1).indices.cpu().numpy())

    Path(root_path_for_preds).mkdir(parents=True, exist_ok=True)
    np.savetxt(Path(root_path_for_preds, f"{file}.csv"), predicts, fmt="%d")

    metric_results = calculate_metrics(score_list, label_list)
    print(
        f"Test Accuracy: {metric_results['accuracy']:.4f}, Test f1-score: {metric_results['f1-score']:.4f}"
    )
