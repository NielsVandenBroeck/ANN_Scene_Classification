import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import models
from DataPreprocessing import DataPreprocessing
from tqdm import tqdm
from device import device


class EmbeddingVisualizer:
    def __init__(self, dataset_path, model_type, weights_path, batch_size=64, num_classes=15):
        self.dataset_path = dataset_path
        self.model_type = model_type  # 'supervised', 'simclr', 'supcon'
        self.weights_path = weights_path
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.model = self._load_model()
        self.test_loader = self._load_test_data()

    def _load_model(self):
        # Load pretrained encoder
        if self.model_type == "supervised":
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        else:
            resnet = models.resnet18(pretrained=False)
            model = nn.Sequential(*list(resnet.children())[:-1])  # encoder only

        model.load_state_dict(torch.load(self.weights_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    def _load_test_data(self):
        # Load test set using standard transforms
        dataloader = DataPreprocessing(self.dataset_path, batch_size=self.batch_size)
        _, _, test_loader = dataloader.create_dataloaders()
        return test_loader

    def extract_embeddings(self):
        # Extract embeddings from the test set
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Extracting Embeddings"):
                images = images.to(device)
                outputs = self.model(images)

                if self.model_type == "supervised":
                    embeddings = outputs  # [B, num_classes]
                else:
                    embeddings = outputs.squeeze()  # [B, 512, 1, 1] -> [B, 512]

                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)

        all_embeddings = torch.cat(all_embeddings).numpy()
        all_labels = torch.cat(all_labels).numpy()
        return all_embeddings, all_labels

    def plot_tsne(self, embeddings, labels):
        print("Applying t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30)
        reduced = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab20", s=12)
        plt.colorbar(scatter)
        plt.title(f"t-SNE ({self.model_type})")
        plt.xlabel("t-SNE dim 1")
        plt.ylabel("t-SNE dim 2")
        plt.tight_layout()
        save_path = os.path.join(f"../plots/{self.model_type}_tsne.png")
        plt.savefig(save_path)
        plt.show()

    def run(self):
        embeddings, labels = self.extract_embeddings()
        self.plot_tsne(embeddings, labels)
