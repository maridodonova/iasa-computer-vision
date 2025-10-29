import torch
import torch.nn as nn
from src.models.blocks import SEBlock, CBAM


class VGGNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1 * 1 * 512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.classifier(x)
        return x


class VGGNetSE(VGGNet):
    def __init__(self, reduction: int) -> None:
        super().__init__()
        self.se_12 = SEBlock(in_channels=64, reduction=reduction)
        self.se_23 = SEBlock(in_channels=128, reduction=reduction)
        self.se_34 = SEBlock(in_channels=256, reduction=reduction)
        self.se_45 = SEBlock(in_channels=512, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.se_12(x)
        x = self.block_2(x)
        x = self.se_23(x)
        x = self.block_3(x)
        x = self.se_34(x)
        x = self.block_4(x)
        x = self.se_45(x)
        x = self.block_5(x)
        x = self.classifier(x)
        return x


class VGGNetCBAM(VGGNet):
    def __init__(self, reduction: int) -> None:
        super().__init__()
        self.cbam_12 = SEBlock(in_channels=64, reduction=reduction)
        self.cbam_23 = SEBlock(in_channels=128, reduction=reduction)
        self.cbam_34 = SEBlock(in_channels=256, reduction=reduction)
        self.cbam_45 = SEBlock(in_channels=512, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.cbam_12(x)
        x = self.block_2(x)
        x = self.cbam_23(x)
        x = self.block_3(x)
        x = self.cbam_34(x)
        x = self.block_4(x)
        x = self.cbam_45(x)
        x = self.block_5(x)
        x = self.classifier(x)
        return x
