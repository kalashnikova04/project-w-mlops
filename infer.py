from pathlib import Path

import hydra
import onnx
import torch
from omegaconf import DictConfig
from onnx2torch import convert
from project_w_mlops.data import create_dataloader
from project_w_mlops.loops import predict


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    test_dataloader = create_dataloader(
        root_path=cfg.data.root_path,
        dataset=cfg.datasets.test_data,
        size_h=cfg.data.size_h,
        size_w=cfg.data.size_w,
        image_mean=cfg.data.image_mean,
        image_std=cfg.data.image_std,
        batch_size=cfg.data.batch_size,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    onnx_model = onnx.load(
        Path(cfg.train.models.root_path, f"{cfg.artifacts.experiment_name}.onnx")
    )
    torch_model = convert(onnx_model).to(device)

    predict(
        torch_model,
        test_dataloader,
        device,
        cfg.predictions.root_path,
        cfg.artifacts.experiment_name,
    )


if __name__ == "__main__":
    main()
