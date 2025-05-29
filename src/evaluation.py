import torch
import torch.nn as nn
from torchvision import models
from DataPreprocessing import DataPreprocessing

def evaluate_model(model_type, encoder_path, dataset_path, classifier_path=None, batch_size=64, num_classes=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder
    if model_type == "supervised":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(encoder_path, map_location=device))
        model.to(device)
        model.eval()
    else:
        # SimCLR or SupCon
        backbone = models.resnet18(pretrained=False)
        encoder = nn.Sequential(*list(backbone.children())[:-1])
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        encoder.to(device)
        encoder.eval()

        # Load classifier head
        classifier = nn.Linear(512, num_classes)
        if classifier_path is None:
            raise ValueError("Must provide classifier_path for SimCLR/SupCon evaluation.")
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.to(device)
        classifier.eval()

    # Load test data
    dataloader = DataPreprocessing(dataset_path, batch_size=batch_size)
    _, _, test_loader = dataloader.create_dataloaders()

    # Evaluate
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            if model_type == "supervised":
                outputs = model(images)
            else:
                features = encoder(images).squeeze()
                outputs = classifier(features)

            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = correct / total
    print(f"[{model_type.upper()}] Test Accuracy: {accuracy:.4f}")
    return accuracy, y_true, y_pred

def main():
    dataset_path = "../dataset"
    evaluate_model(model_type="supervised", encoder_path="../models/supervised.pth", dataset_path=dataset_path)
    evaluate_model(model_type="simclr", encoder_path="../models/simclr_encoder.pth", classifier_path="../models/simclr_linear.pth", dataset_path=dataset_path)
    evaluate_model(model_type="supcon", encoder_path="../models/supcon_encoder.pth", classifier_path="../models/supcon_linear.pth", dataset_path=dataset_path)

if __name__ == "__main__":
    main()
