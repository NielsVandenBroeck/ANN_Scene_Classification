import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from utils import TransformTwice

class SupConDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        img, label = self.base_dataset[index]
        x1, x2 = self.transform(img)
        return (x1, x2), label

class DataPreprocessing:
    def __init__(self, dataset_path, batch_size=64, num_workers=4, input_size=224, simclr=False):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size

        # Standard normalization for pretrained ResNet models
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if simclr:
            self.train_transform = TransformTwice(transforms.Compose([
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=9),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]))
        else:
            self.train_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

        self.test_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def create_dataloaders(self, include_labels=False):
        train_path = f"{self.dataset_path}/train"
        val_path = f"{self.dataset_path}/val"
        test_path = f"{self.dataset_path}/test"

        if include_labels:
            base_train = datasets.ImageFolder(train_path)
            train_dataset = SupConDataset(base_train, self.train_transform)
        else:
            train_dataset = datasets.ImageFolder(train_path, transform=self.train_transform)

        val_dataset = datasets.ImageFolder(val_path, transform=self.test_transform)
        test_dataset = datasets.ImageFolder(test_path, transform=self.test_transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                 shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader, test_loader

    def contrastive_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(self.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])