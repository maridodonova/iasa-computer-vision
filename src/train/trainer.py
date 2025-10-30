import torch
import src.train.metrics.metrics as metrics


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

        self._metrics = {
            mode: { "loss": metrics.Loss(), "accuracy": metrics.Accuracy() }
            for mode in ["train", "val", "test"]
        }

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

        for metric in self._metrics[mode].values():
            metric.clear()

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

            self._metrics[mode]["loss"].update(loss)
            self._metrics[mode]["accuracy"].update(logits, labels)

        for metric in self._metrics[mode].values():
            metric.compute()


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

            train_history = { "loss": {}, "accuracy": {} }
            for mode in ["train", "val"]:
                for name, metric in self._metrics[mode].items():
                    train_history[name][mode] = metric.get_history()

            if verbose:
                print(
                    f"Epoch [{epoch+1:2d}/{num_epochs}]:",
                    f"Train Loss: {train_history['loss']['train'][-1]:.4f},",
                    f"Val Loss: {train_history['loss']['val'][-1]:.4f},",
                    f"Train Accuracy: {train_history['accuracy']['train'][-1]:.4f},",
                    f"Val Accuracy: {train_history['accuracy']['val'][-1]:.4f}."
                )

        return train_history
    
    def test(
            self,
            test_loader: torch.utils.data.DataLoader,
            verbose: bool = True
    ) -> dict:
        self._run_epoch(test_loader, mode="test")

        test_history = { name: metric.get_history() for name, metric in self._metrics["test"].items() }

        if verbose:
            print(
                f"Loss: {test_history['loss'][-1]:.4f},",
                f"Accuracy: {test_history['accuracy'][-1]:.4f}."
            )

        return test_history
