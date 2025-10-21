import torch


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer,
        criterion: torch.nn.Module,
        device: str = "cpu"
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.device = device
        self.model = model.to(self.device)

        self.__history = {
            "loss": {"train": [], "val": []},
            "accuracy": {"train": [], "val": []}
        }
    
    def _run_epoch(
            self,
            loader: torch.utils.data.DataLoader,
            validation: bool = False
        ) -> None:
        # Set mode
        if validation:
            self.model.eval()
        else:
            self.model.train()

        # Init counters
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            if not validation:
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

        # Update history
        mode = "val" if validation else "train"
        self.__history["loss"][mode].append(running_loss / len(loader))
        self.__history["accuracy"][mode].append(correct / total)

    def train(
            self,
            num_epochs: int,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            verbose: bool = True
        ) -> dict:
        for epoch in range(num_epochs):
            self._run_epoch(train_loader)
            self._run_epoch(val_loader, validation=True)

            if verbose:
                print(
                    f"Epoch [{epoch+1:2d}/{num_epochs}]:",
                    f"Train Loss: {self.__history['loss']['train'][-1]:.4f},",
                    f"Val Loss: {self.__history['loss']['val'][-1]:.4f},",
                    f"Train Accuracy: {self.__history['accuracy']['train'][-1]:.4f},",
                    f"Val Accuracy: {self.__history['accuracy']['val'][-1]:.4f}."
                )

        return self.__history.copy()
