import torch


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device="cpu"
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.device = device
        self.model = model.to(self.device)

    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate_epoch(self, loader):
        self.model.eval()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(loader)
        accuracy = correct / total

        return avg_loss, accuracy
    
    def train(self, num_epochs, train_loader, val_loader, verbose=True):
        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []

        for epoch in range(num_epochs):
            train_loss, train_accuracy = self.train_epoch(train_loader)
            val_loss, val_accuracy = self.validate_epoch(val_loader)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

        if verbose:
            print(
                f"Epoch [{epoch+1}/{num_epochs}]:",
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f},",
                f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}."
            )

        return train_losses, train_accuracies, val_losses, val_accuracies
