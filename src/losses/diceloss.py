import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth: int = 1):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        pred = nn.functional.softmax(pred, dim=1)
        num_classes = pred.shape[1]

        ref = nn.functional.one_hot(ref, num_classes=num_classes).permute(0, 3, 1, 2)

        intersection = (pred * ref).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + ref.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean(dim=1).mean()
