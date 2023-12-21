import hydra
import torch
from loops import train_model
from model import MyModel
from omegaconf import DictConfig, OmegaConf
from preprocessing import create_dataloader


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

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
        in_feature=cfg.params.in_feature,
        embedding_size=cfg.params.embedding_size,
        num_classes=cfg.data.num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model_cnn_norm.parameters(), lr=cfg.params.lr)

    loss_fn = torch.nn.CrossEntropyLoss()

    train_model(
        model_cnn_norm,
        train_dataloader,
        val_dataloader,
        optimizer,
        device,
        loss_fn,
        n_epochs=cfg.params.n_epochs,
        batch_size=cfg.data.batch_size,
        model_path=cfg.artifacts.root_path,
        ckpt_name=cfg.artifacts.experiment_name,
    )


if __name__ == "__main__":
    main()
