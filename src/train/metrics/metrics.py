import torch


class Metric:
    def __init__(self) -> None:
        self._logits = []
        self._refs = []
        self._compute_history = []

    def update(self, logits: torch.Tensor, refs: torch.Tensor | None = None) -> None:
        self._logits.append(logits.detach().cpu())
        if refs is not None:
            self._refs.append(refs.detach().cpu())

    def clear(self) -> None:
        self._logits = []
        self._refs = []
    
    def clear_history(self) -> None:
        self._compute_history = []
    
    def get_history(self) -> list:
        return self._compute_history.copy()


class Loss(Metric):
    def update(self, value):
        return super().update(value)
    
    def compute(self) -> torch.Tensor:
        self._logits = torch.Tensor(self._logits)
        loss = self._logits.mean()

        self._compute_history.append(loss)

        return loss


class Accuracy(Metric):
    def compute(self) -> torch.Tensor:
        self._logits = torch.cat(self._logits)
        self._refs = torch.cat(self._refs)

        preds = torch.argmax(self._logits, dim=1)
        correct = (preds == self._refs).sum()
        total = self._refs.numel()

        accuracy = correct / total

        self._compute_history.append(accuracy)

        return accuracy
