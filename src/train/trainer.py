import torch
from src.train.metrics.metric_collection import MetricCollection


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: str = "cpu"
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.device = device
        self.model = model.to(self.device)

        self._metrics = { mode: MetricCollection(["loss", "accuracy"]) for mode in ["train", "val", "test"] }

    def _run_epoch(
            self,
            loader: torch.utils.data.DataLoader,
            mode: str = "train"
    ) -> None:
        # Set mode
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

        self._metrics[mode].clear()

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            if mode == "train":
                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

            self._metrics[mode].update({
                "loss": { "value": loss },
                "accuracy": { "logits": logits, "refs": labels }
            })

        self._metrics[mode].compute()


    def train(
            self,
            num_epochs: int,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            verbose: bool = True
    ) -> dict:
        for epoch in range(num_epochs):
            self._run_epoch(train_loader)
            self._run_epoch(val_loader, mode="val")

            history = { mode: self._metrics[mode].get_history() for mode in ["train", "val"] }

            if verbose:
                print(
                    f"Epoch [{epoch+1:2d}/{num_epochs}]:",
                    f"Train Loss: {history['train']['loss'][-1]:.4f},",
                    f"Val Loss: {history['val']['loss'][-1]:.4f},",
                    f"Train Accuracy: {history['train']['accuracy'][-1]:.4f},",
                    f"Val Accuracy: {history['val']['accuracy'][-1]:.4f}."
                )

        return history
    
    def test(
            self,
            test_loader: torch.utils.data.DataLoader,
            verbose: bool = True
    ) -> dict:
        self._run_epoch(test_loader, mode="test")

        history = self._metrics["test"].get_history()

        if verbose:
            print(
                f"Loss: {history['loss'][-1]:.4f},",
                f"Accuracy: {history['accuracy'][-1]:.4f}."
            )

        return history
