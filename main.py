import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pipeline import MyModel, create_dataloader, train_model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    train_batch_gen = create_dataloader(
        root_path=cfg.paths.data,
        dataset=cfg.datasets.train_data,
        batch_size=cfg.params.batch_size,
        shuffle=True,
    )
    val_batch_gen = create_dataloader(
        root_path=cfg.paths.data,
        dataset=cfg.datasets.valid_data,
        batch_size=cfg.params.batch_size,
    )
    # test_batch_gen = create_dataloader("test_labeled", cfg.params.batch_size)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_cnn_norm = MyModel(
        in_feature=cfg.params.in_feature,
        embedding_size=cfg.params.embedding_size,
        num_classes=cfg.params.num_classes,
    )
    model_cnn_norm = model_cnn_norm.to(device)
    opt = torch.optim.Adam(model_cnn_norm.parameters(), lr=cfg.params.lr)

    ckpt_name_cnn_norm = "model_cnn_norm.ckpt"

    model_cnn_norm, opt = train_model(
        model_cnn_norm,
        train_batch_gen,
        val_batch_gen,
        opt,
        n_epochs=cfg.params.n_epochs,
        ckpt_name=ckpt_name_cnn_norm,
    )


if __name__ == "__main__":
    main()
