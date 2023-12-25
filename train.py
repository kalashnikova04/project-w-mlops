import subprocess

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from project_w_mlops.data import create_dataloader
from project_w_mlops.loops import train_model
from project_w_mlops.model import MyModel


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    subprocess.run(["dvc", "pull"])

    mlflow.set_tracking_uri(uri="http://128.0.1.1:8080")
    mlflow.set_experiment(cfg.artifacts.experiment_name)

    train_dataloader = create_dataloader(
        root_path=cfg.data.root_path,
        dataset=cfg.datasets.train_data,
        size_h=cfg.data.size_h,
        size_w=cfg.data.size_w,
        image_mean=cfg.data.image_mean,
        image_std=cfg.data.image_std,
        batch_size=cfg.data.batch_size,
        shuffle=True,
    )
    val_dataloader = create_dataloader(
        root_path=cfg.data.root_path,
        dataset=cfg.datasets.valid_data,
        size_h=cfg.data.size_h,
        size_w=cfg.data.size_w,
        image_mean=cfg.data.image_mean,
        image_std=cfg.data.image_std,
        batch_size=cfg.data.batch_size,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_cnn_norm = MyModel(
        in_feature=cfg.train.in_feature,
        embedding_size=cfg.train.embedding_size,
        num_classes=cfg.data.num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model_cnn_norm.parameters(), lr=cfg.train.lr)

    loss_fn = torch.nn.CrossEntropyLoss()

    mlflow.log_param("pic_size_h", cfg.data.size_h)
    mlflow.log_param("pic_size_w", cfg.data.size_w)
    mlflow.log_param("batch_size", cfg.data.batch_size)
    mlflow.log_param("embedding_size", cfg.train.embedding_size)
    mlflow.log_param("learning_rate", cfg.train.lr)
    mlflow.log_param("n_epochs", cfg.train.n_epochs)
    mlflow.log_param(
        "commit_id",
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip(),
    )

    train_model(
        model_cnn_norm,
        train_dataloader,
        val_dataloader,
        optimizer,
        device,
        loss_fn,
        n_epochs=cfg.train.n_epochs,
        batch_size=cfg.data.batch_size,
        model_path=cfg.train.models.root_path,
        ckpt_name=cfg.artifacts.experiment_name,
    )


if __name__ == "__main__":
    main()
