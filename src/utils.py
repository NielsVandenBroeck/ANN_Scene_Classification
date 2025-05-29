from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os


# Wrapper class that applies the same base transformation twice to a single input (for contrastive learning)
class TransformTwice:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        # transformation twice
        return self.base_transform(x), self.base_transform(x)

def plot_loss_and_accuracy(metrics, title="Training Performance"):
    train_loss = metrics["train_loss"]
    val_loss = metrics["val_loss"]
    train_acc = metrics["train_acc"]
    val_acc = metrics["val_acc"]

    epochs = range(1, len(train_loss) + 1)

    # Plot Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Loss', color='tab:blue', marker='o')
    plt.plot(epochs, val_loss, label='Val Loss', color='tab:orange', marker='o')
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_path = f"../plots/{title.lower().replace(' ', '_')}_loss.png"
    plt.savefig(loss_path)
    plt.show()
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label='Train Accuracy', color='tab:blue', marker='o')
    plt.plot(epochs, val_acc, label='Val Accuracy', color='tab:orange', marker='o')
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_path = f"../plots/{title.lower().replace(' ', '_')}_accuracy.png"
    plt.savefig(acc_path)
    plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        xticks_rotation=45,
        cmap="Blues",
        ax=ax
    )
    ax.set_title(title)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    cm_path = f"../plots/{title.lower().replace(' ', '_')}.png"
    plt.savefig(cm_path, dpi=300)
    plt.show()
