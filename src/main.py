from matplotlib import pyplot as plt

from SupervisedTrainer import SupervisedTrainer
from embedding import EmbeddingVisualizer
from LinearProbeTrainer import LinearProbeTrainer
from ContrastiveTrainer import ContrastiveTrainer
from evaluation import evaluate_model
from utils import plot_loss_and_accuracy, plot_confusion_matrix


def run_all(dataset_path):
    accuracies = {}

    # Supervised Training
    print("\n--- Testing Supervised Model ---")
    sup = SupervisedTrainer(dataset_path, model_name="supervised.pth", num_epochs=50)
    supervised_metrics = sup.train()
    plot_loss_and_accuracy(supervised_metrics, title="Supervised Training Performance")

    sup_acc, sup_y_true, sup_y_pred = evaluate_model(model_type="supervised", encoder_path="../models/supervised.pth", dataset_path=dataset_path)
    plot_confusion_matrix(sup_y_true,sup_y_pred, title="Supervised Training Confusion Matrix")
    accuracies["Supervised Accuracy"] = [sup_acc]

    ev = EmbeddingVisualizer(
        dataset_path=dataset_path,
        model_type="supervised",
        weights_path="../models/supervised.pth"
    )
    ev.run()


    # SimCLR Encoder Training
    print("\n--- Testing SimCLR Encoder ---")
    simclr = ContrastiveTrainer(dataset_path, model="simclr", model_name="simclr_encoder", num_epochs=50)
    simclr.train()

    ev = EmbeddingVisualizer(
        dataset_path=dataset_path,
        model_type="simclr",
        weights_path="../models/simclr_encoder.pth"
    )
    ev.run()


    # SupCon Encoder Training
    print("\n--- Testing SupCon Encoder ---")
    supcon = ContrastiveTrainer(dataset_path, model="supcon", model_name="supcon_encoder", num_epochs=50)
    supcon.train()

    ev = EmbeddingVisualizer(
        dataset_path=dataset_path,
        model_type="supcon",
        weights_path="../models/supcon_encoder.pth"
    )
    ev.run()


    # Linear Probe (SimCLR)
    print("\n--- Testing Linear Probe for SimCLR ---")
    simclr_lp = LinearProbeTrainer(
        encoder_path="../models/simclr_encoder.pth",
        dataset_path=dataset_path,
        model_name="simclr_linear",
        num_epochs=50
    )
    simclr_lp_metrics = simclr_lp.train()
    plot_loss_and_accuracy(simclr_lp_metrics, title="SimCLR Linear Probe Training Performance")

    simclr_lp_acc, simclr_lp_y_true, simclr_lp_y_pred = evaluate_model(model_type="simclr", encoder_path="../models/simclr_encoder.pth", classifier_path="../models/simclr_linear.pth", dataset_path=dataset_path)
    plot_confusion_matrix(simclr_lp_y_true, simclr_lp_y_pred, title="SimCLR Linear Probe Training Confusion Matrix")
    accuracies["SimCLR + LinearProbe Accuracy"] = [simclr_lp_acc]


    # Linear Probe (SupCon)
    print("\n--- Testing Linear Probe for SupCon ---")
    supcon_lp = LinearProbeTrainer(
        encoder_path="../models/supcon_encoder.pth",
        dataset_path=dataset_path,
        model_name="supcon_linear",
        num_epochs=50
    )
    supcon_lp_metrics = supcon_lp.train()
    plot_loss_and_accuracy(supcon_lp_metrics, title="SupCon Linear Probe Training Performance")

    supcon_lp_acc, supcon_lp_y_true, supcon_lp_y_pred  = evaluate_model(model_type="supcon", encoder_path="../models/supcon_encoder.pth", classifier_path="../models/supcon_linear.pth", dataset_path=dataset_path)
    plot_confusion_matrix(supcon_lp_y_true, supcon_lp_y_pred, title="SupCon Linear Probe Training Confusion Matrix")
    accuracies["SupCon + LinearProbe Accuracy"] = [supcon_lp_acc]


    # Plot Summary Accuracy Comparison
    print("\n--- Accuracy Summary ---")
    for k, v in accuracies.items():
        print(f"{k}: {v[0]:.4f}")

def main():
    dataset_path = "../dataset"
    run_all(dataset_path)

if __name__ == '__main__':
    main()