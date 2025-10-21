import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1
    ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)

        output += identity
        output = self.relu(output)

        return output


class SEBlock(nn.Module):
    def __init__(
            self,
            kernel_size: tuple,
            num_channels: int,
            reducted_dim: int
    ) -> None:
        super(SEBlock, self).__init__()
        self.squeeze = nn.AvgPool2d(kernel_size)

        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels, reducted_dim),
            nn.ReLU(),
            nn.Linear(reducted_dim, num_channels),
            nn.Sigmoid(),
            nn.Unflatten(dim=1, unflattened_size=(num_channels, 1, 1))
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.squeeze(x)
        weights = self.excitation(weights)
        scaled = x * weights
        return scaled
