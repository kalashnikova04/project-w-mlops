import torch
import torch.nn as nn


def conv_block_3x3(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(),
    )


# a special module that converts [batch, channel, w, h] to [batch, units]: tf/keras style
class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class MyModel(torch.nn.Module):
    def __init__(self, in_feature, embedding_size, num_classes):
        super(MyModel, self).__init__()

        self.model = nn.Sequential(
            conv_block_3x3(in_feature, in_feature * 10),
            nn.MaxPool2d(2),
            conv_block_3x3(in_feature * 10, in_feature * 100),
            nn.MaxPool2d(2),
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
        )
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Sequential(
            nn.Linear(in_feature * 100, embedding_size, bias=False),
            nn.BatchNorm1d(num_features=embedding_size),
            nn.ReLU(),
        )
        self.pred = nn.Sequential(nn.Linear(embedding_size, num_classes, bias=False))

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.pred(x)
        return x
