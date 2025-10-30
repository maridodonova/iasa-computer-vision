import torch
import torch.nn as nn
from src.models.blocks import SEBlock, CBAM


class VGG13(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int
    ) -> None:
        super().__init__()
        self.block_1 = self._make_block(in_channels, 64, n_convs=2)
        self.block_2 = self._make_block(64, 128, n_convs=2)
        self.block_3 = self._make_block(128, 256, n_convs=2)
        self.block_4 = self._make_block(256, 512, n_convs=2)
        self.block_5 = self._make_block(512, 512, n_convs=2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1 * 1 * 512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def _make_block(
            self,
            in_channels: int,
            out_channels: int,
            n_convs: int
    ) -> nn.Sequential:
        layers = []

        for _ in range(n_convs):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
                nn.ReLU()
            ]
            in_channels = out_channels

        layers += [nn.MaxPool2d(kernel_size=2)]

        return nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.classifier(x)
        return x


class VGG13SE(VGG13):
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


class VGG13CBAM(VGG13):
    def __init__(self, reduction: int) -> None:
        super().__init__()
        self.cbam_12 = CBAM(in_channels=64, reduction=reduction)
        self.cbam_23 = CBAM(in_channels=128, reduction=reduction)
        self.cbam_34 = CBAM(in_channels=256, reduction=reduction)
        self.cbam_45 = CBAM(in_channels=512, reduction=reduction)

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
