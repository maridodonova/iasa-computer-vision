import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            in_size: tuple
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.sigmoid = nn.Sigmoid()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sigmoid(self.conv1(x))
        x = self.pool1(x)

        x = self.sigmoid(self.conv2(x))
        x = self.pool2(x)

        x = self.flatten(x)

        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)

        return x
