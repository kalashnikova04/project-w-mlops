import torch
from model import MyModel
from preprocessing import create_dataloader
from train import train_model


def main():
    train_batch_gen = create_dataloader("train_11k")
    val_batch_gen = create_dataloader("val")
    # test_batch_gen = create_dataloader("test_labeled")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_cnn_norm = MyModel(3)
    model_cnn_norm = model_cnn_norm.to(device)
    opt = torch.optim.Adam(model_cnn_norm.parameters(), lr=1e-3)
    ckpt_name_cnn_norm = "model_cnn_norm.ckpt"
    model_cnn_norm, opt = train_model(
        model_cnn_norm, train_batch_gen, val_batch_gen, opt, ckpt_name=ckpt_name_cnn_norm
    )


if __name__ == "__main__":
    main()
