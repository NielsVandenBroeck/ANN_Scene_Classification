import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from DataPreprocessing import DataPreprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm
from device import device


class LinearProbeTrainer:
    def __init__(self, encoder_path, dataset_path, model_type="simclr", num_epochs=50, batch_size=64, lr=1e-3, model_name="simclr_linear.pth"):
        self.encoder_path = encoder_path
        self.model_type = model_type  # "simclr" or "supcon"
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.model_save_path = f"../models/{model_name}.pth"

        # Load frozen encoder
        self.encoder = self._load_encoder()
        self.encoder.eval()

        # Prepare data
        dataloader = DataPreprocessing(dataset_path, batch_size=batch_size)
        self.train_loader, self.val_loader, _ = dataloader.create_dataloaders()
        self.num_classes = len(self.train_loader.dataset.classes)

        # Classifier head
        self.classifier = nn.Linear(512, self.num_classes).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.lr)

    def _load_encoder(self):
        base_model = models.resnet18(pretrained=False)
        encoder = nn.Sequential(*list(base_model.children())[:-1])  # remove FC
        encoder.load_state_dict(torch.load(self.encoder_path, map_location=device))
        encoder.to(device)
        return encoder

    def train(self):
        best_acc = 0

        metrics ={
           "train_loss" : [],
            "val_loss" : [],
            "train_acc" : [],
            "val_acc" : []
        }

        for epoch in range(self.num_epochs):
            self.classifier.train()
            running_loss = 0.0
            correct, total = 0, 0
            total_samples = 0
            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
                images, labels = images.to(device), labels.to(device)

                with torch.no_grad():
                    features = self.encoder(images).squeeze()  # [B, 512, 1, 1] -> [B, 512]

                outputs = self.classifier(features)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
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
            print(f"Epoch {epoch+1}: Train Loss = {epoch_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.classifier.state_dict(), self.model_save_path)

        print(f"\nLinear probe training complete. Best Val Accuracy: {best_acc:.4f}")
        print(f"Saved linear classifier to {self.model_save_path}")
        return metrics

    def evaluate(self):
        self.classifier.eval()
        self.encoder.eval()  # Make sure the encoder is also in eval mode

        correct, total = 0, 0
        total_loss = 0.0

        criterion = nn.CrossEntropyLoss()  # Define the loss function

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(device), labels.to(device)
                features = self.encoder(images).squeeze()
                outputs = self.classifier(features)

                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                total += labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total
        return accuracy, avg_loss