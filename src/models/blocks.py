import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1
    ) -> None:
        super().__init__()
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
        super().__init__()
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


class ChannelAttentionModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            reduction: int
        ) -> None:
        super().__init__()

        assert reduction >= 1, "reduction must be >= 1"
        assert in_channels >= 1, "in_channels must be >= 1"
        assert reduction <= in_channels, "reduction must be <= in_channels"

        self.in_channels = in_channels
        self.reduction = reduction
        self.reduced_channels = self.in_channels // reduction

        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.reduced_channels),
            nn.ReLU(),
            nn.Linear(self.reduced_channels, self.in_channels)
        )
        self.unflatten = nn.Unflatten(dim=-1, unflattened_size=(self.in_channels, 1, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_dims = x.dim()
        assert num_dims == 3 or num_dims == 4, "x shape should be (B, C, H, W) or (C, H, W)"
        assert x.shape[-3] == self.in_channels, "x has invalid channel dim"

        spatial_dims = (-2, -1)
        avg_pool_out = torch.mean(x, dim=spatial_dims)
        max_pool_out = torch.amax(x, dim=spatial_dims)

        output = self.unflatten(avg_pool_out + max_pool_out)
        output = self.sigmoid(output)

        return output


class SpatialAttentionModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_dims = x.dim()
        assert num_dims == 3 or num_dims == 4, "x shape should be (B, C, H, W) or (C, H, W)"

        channel_dim = -3
        avg_pool_out = torch.mean(x, dim=channel_dim, keepdim=True)
        max_pool_out = torch.amax(x, dim=channel_dim, keepdim=True)

        output = torch.cat([avg_pool_out, max_pool_out], dim=channel_dim)
        output = self.conv(output)
        output = self.sigmoid(output)

        return output


class CBAM(nn.Module):
    def __init__(
            self,
            in_channels: int,
            reduction: int
    ) -> None:
        super().__init__()
        self.cam = ChannelAttentionModule(in_channels, reduction)
        self.sam = SpatialAttentionModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cam_output = self.cam(x)
        output = cam_output * x

        sam_output = self.sam(output)
        output = sam_output * output

        return output
