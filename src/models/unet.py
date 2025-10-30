import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int
    ) -> None:
        super().__init__()

        encoder_channel_nums = [in_channels, 64, 128, 256, 512]
        decoder_channel_nums = [1024, 512, 256, 128, 64]

        self.encoder_blocks = self._init_blocks(encoder_channel_nums)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = self._make_conv_block(512, 1024)

        self.up_convs = self._init_up_conv_blocks(decoder_channel_nums)
        self.decoder_blocks = self._init_blocks(decoder_channel_nums)

        self.conv1x1 = nn.Conv2d(64, 3, kernel_size=1)

    def _make_conv_block(
            self,
            in_channels: int,
            out_channels: int,
            n_convs: int = 2
    ) -> nn.Sequential:
        layers = []

        # Add conv 3x3, ReLU
        for _ in range(n_convs):
            layers += [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1
                ),
                nn.ReLU()
            ]
            in_channels = out_channels

        return nn.Sequential(*layers)
    
    def _init_blocks(self, channel_nums: list) -> nn.ModuleList:
        blocks = [
            self._make_conv_block(channel_nums[i-1], channel_nums[i])
            for i in range(1, len(channel_nums))
        ]
        return nn.ModuleList(blocks)
    
    def _init_up_conv_blocks(self, channel_nums: list) -> nn.ModuleList:
        blocks = [
            nn.ConvTranspose2d(
                in_channels=channel_nums[i-1],
                out_channels=channel_nums[i],
                kernel_size=2,
                stride=2
            ) for i in range(1, len(channel_nums))
        ]
        return nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_outputs = []

        out = x
        for enc_block in self.encoder_blocks:
            out = enc_block(out)
            encoder_outputs.append(out)
            out = self.max_pool(out)

        out = self.bottleneck(out)

        for up_conv, dec_block, enc_out in zip(self.up_convs, self.decoder_blocks, reversed(encoder_outputs)):
            out = up_conv(out)
            out = torch.cat([enc_out, out], dim=-3)
            out = dec_block(out)
        
        out = self.conv1x1(out)

        return out
