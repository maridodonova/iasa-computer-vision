import torch
import torch.nn as nn
from src.models.blocks import ResBlock


class ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self.make_layer(64, 64, 2, stride=1)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)
        self.layer4 = self.make_layer(256, 512, 2, stride=2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def make_layer(
            self,
            in_channels: int,
            out_channels: int,
            num_blocks: int,
            stride: int
    ) -> nn.Sequential:
        layers = [ResBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier(x)

        return x
