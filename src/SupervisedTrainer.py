import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from DataPreprocessing import DataPreprocessing
from device import device

class SupervisedTrainer:
    def __init__(self, dataset_path, num_epochs=50, batch_size=64, lr=1e-3, model_name="supervised"):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_save_path = f"../models/{model_name}.pth"

        # Prepare data
        data_loader = DataPreprocessing(dataset_path, batch_size)
        self.train_loader, self.val_loader, _ = data_loader.create_dataloaders()
        self.num_classes = len(self.train_loader.dataset.classes)

        # Initialize model
        self.model = self._build_model().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _build_model(self):
        # Use resnet18 architecture
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        return model

    def train(self):
        best_val_acc = 0

        # Metrics for plots
        metrics ={
           "train_loss" : [],
            "val_loss" : [],
            "train_acc" : [],
            "val_acc" : []
        }

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            total_samples = 0
            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_samples += images.size(0)

            epoch_loss = running_loss / total_samples
            train_acc = correct / total
            val_acc, val_loss = self.evaluate()

            metrics["train_loss"].append(epoch_loss)
            metrics["val_loss"].append(val_loss)
            metrics["train_acc"].append(train_acc)
            metrics["val_acc"].append(val_acc)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.model_save_path)

        print(f"\nSupervised Training complete. Model saved to {self.model_save_path}. Best validation accuracy: {best_val_acc:.4f}")
        return metrics

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = val_loss / len(self.val_loader)
        accuracy = correct / total
        return accuracy, avg_loss
