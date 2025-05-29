import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from DataPreprocessing import DataPreprocessing
from device import device

torch.autograd.set_detect_anomaly(True)

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ContrastiveTrainer:
    def __init__(self, dataset_path, model="simclr", num_epochs=50, batch_size=64, temperature=0.5, lr=1e-4, model_name="simclr_encoder"):
        self.dataset_path = dataset_path
        self.model = model
        self.batch_size = batch_size
        self.temperature = temperature
        self.lr = lr
        self.num_epochs = num_epochs
        self.model_save_path = f"../models/{model_name}.pth"

        # Prepare SimCLR-style augmented dataset (overriding transforms inside)
        self.data_loader = DataPreprocessing(dataset_path, batch_size=batch_size, simclr=True)
        self.train_loader, _, _ = self.data_loader.create_dataloaders(include_labels=(model == "supcon"))

        # Build model
        self.encoder = self._build_encoder().to(device)
        self.projection_head = ProjectionHead().to(device)

        self.optimizer = optim.Adam(list(self.encoder.parameters()) +
                                    list(self.projection_head.parameters()), lr=self.lr)

    def _build_encoder(self):
        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]  # remove FC layer
        return nn.Sequential(*modules)  # output shape: [B, 512, 1, 1]

    def train(self):
        self.encoder.train()
        self.projection_head.train()

        for epoch in range(self.num_epochs):
            total_loss = 0

            for (x1, x2), labels in self.train_loader:  # now returns tuple of augmentations
                x1, x2 = x1.to(device), x2.to(device)
                if self.model == "supcon":
                    labels = labels.to(device)

                # Encode and project
                h1 = self.encoder(x1).squeeze()  # [B, 512]
                h2 = self.encoder(x2).squeeze()
                z1 = self.projection_head(h1)
                z2 = self.projection_head(h2)

                if self.model == "supcon":
                    loss = self._supervised_contrastive_loss(z1, z2, labels)
                else: # simclr
                    loss = self._nt_xent_loss(z1, z2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

        # Save encoder for linear probing later
        torch.save(self.encoder.state_dict(), self.model_save_path)
        print(f"\nSimCLR training complete. Encoder saved to {self.model_save_path}.")


    def _nt_xent_loss(self, z_i, z_j):
        batch_size = z_i.size(0)

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)

        sim = torch.matmul(z, z.T)  # [2N, 2N]
        sim /= self.temperature

        labels = torch.arange(batch_size).to(device)
        labels = torch.cat([labels, labels], dim=0)

        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
        sim = sim.masked_fill(mask, float('-inf'))  # remove self-similarity

        positives = torch.exp((z_i * z_j).sum(dim=-1) / self.temperature)
        positives = torch.cat([positives, positives], dim=0)

        denominator = torch.exp(sim).sum(dim=1)
        loss = -torch.log(positives / denominator).mean()
        return loss

    def _supervised_contrastive_loss(self, z1, z2, labels):
        z = torch.cat([z1, z2], dim=0)  # 2N x D
        z = F.normalize(z, dim=1)
        labels = torch.cat([labels, labels], dim=0)  # 2N

        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)  # 2N x 2N

        sim = torch.matmul(z, z.T) / self.temperature
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Mask out self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
        mask *= logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        loss = - (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        return loss.mean()