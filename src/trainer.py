import torch


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

        self.__train_history = {
            "loss": { "train": [], "val": [] },
            "accuracy": { "train": [], "val": [] }
        }
        self.__test_history = { "loss": [], "accuracy": [] }

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

        # Init counters
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            if mode == "train":
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(loader)
        accuracy = correct / total

        # Update history
        if mode == "test":
            self.__test_history["loss"].append(epoch_loss)
            self.__test_history["accuracy"].append(accuracy)
        else:
            self.__train_history["loss"][mode].append(epoch_loss)
            self.__train_history["accuracy"][mode].append(accuracy)

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

            if verbose:
                print(
                    f"Epoch [{epoch+1:2d}/{num_epochs}]:",
                    f"Train Loss: {self.__train_history['loss']['train'][-1]:.4f},",
                    f"Val Loss: {self.__train_history['loss']['val'][-1]:.4f},",
                    f"Train Accuracy: {self.__train_history['accuracy']['train'][-1]:.4f},",
                    f"Val Accuracy: {self.__train_history['accuracy']['val'][-1]:.4f}."
                )

        return self.__train_history.copy()
    
    def test(
            self,
            test_loader: torch.utils.data.DataLoader,
            verbose: bool = True
    ) -> dict:
        self._run_epoch(test_loader, mode="test")

        if verbose:
            print(
                f"Loss: {self.__test_history['loss'][-1]:.4f}",
                f"Accuracy: {self.__test_history['accuracy'][-1]:.4f}"
            )

        return self.__test_history.copy()
