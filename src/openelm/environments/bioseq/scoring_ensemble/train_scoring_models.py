
import os
import numpy as np
import torch

from torch.utils.data import DataLoader

from openelm.environments.bioseq.utr_fitness_function.utr_scoring_ensemble.utr_log_dataset import UTRLogDataset
from openelm.environments.bioseq.scoring_ensemble.scoring_model import ScoringNetwork
from trainer import COMTrainer
from openelm.utils.plots import plot_learning_curves

DEFAULT_OFFLINE_DATA_PATH = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\OpenELM_GenomicQD\design-bench_forked\design_bench_data\utr\oracle_data\original_v0_minmax_orig\sampled_offline_relabeled_data\sampled_data_fraction_1_3_seed_42"  #todo: not absolute path
DEFAULT_X_PATH = os.path.join(DEFAULT_OFFLINE_DATA_PATH, "x.npy")
DEFAULT_Y_PATH = os.path.join(DEFAULT_OFFLINE_DATA_PATH, "y.npy")

def main(
    data_x_path: str,
    data_y_path: str,
    save_dir: str,
    seq_len: int,
    K: int,
    L: int,
    use_conservative: bool,
    device: str,
    epochs: int,
    batch_size: int
):
    """Trains L scoring models and saves them to disk."""

    # Check if GPU is available
    if torch.cuda.is_available() and device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1) Load data
    X = np.load(data_x_path)
    Y = np.load(data_y_path)
    if Y.ndim == 2:  # flatten if shape is (N,1)
        Y = Y[:, 0]

    disk_target = os.path.join(DEFAULT_OFFLINE_DATA_PATH, "preprocessed_one_hot_log_data")
    dataset = UTRLogDataset(X, Y, disk_target=disk_target)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create a save directory
    os.makedirs(save_dir, exist_ok=True)

    # 2) Train L scoring models
    for i in range(2, L): # todo: change to range(L)
        print(f"\n=== Training scoring model #{i+1}/{L} ===")

        # (A) instantiate the model
        torch.manual_seed(i + 1234)  # for reproducibility
        model = ScoringNetwork(seq_len, K).to(device)

        # (B) create trainer with or without COM
        trainer = COMTrainer(
            model=model,
            device=device,
            lr=3e-4,
            alpha_init=0.1,
            alpha_lr=0.01,
            overestimation_limit=2.0,
            particle_steps=50,
            particle_lr=2.0,
            entropy_coeff=0.0,
            noise_std=0.0,
            use_conservative=use_conservative
        )

        learning_stats = {
            "train/mse": [],
            "train/overestimation": [],
            "train/alpha": []
        }
        all_epochs_stats = {
            "mse": [],
            "overestimation": [],
            "alpha": []
        }

        # (C) train
        for epoch in range(epochs):
            epoch_stats = {"mse": [], "overestimation": [], "alpha": []}
            for x_batch, y_batch in loader:
                stats = trainer.train_step(x_batch, y_batch)

                # update epoch stats
                epoch_stats["mse"].append(stats["train/mse"])
                if "train/overestimation" in stats:
                    epoch_stats["overestimation"].append(stats["train/overestimation"])
                if "train/alpha" in stats:
                    epoch_stats["alpha"].append(stats["train/alpha"])
                # update learning stats
                learning_stats["train/mse"].append(stats["train/mse"])
                if "train/overestimation" in stats:
                    learning_stats["train/overestimation"].append(stats["train/overestimation"])
                if "train/alpha" in stats:
                    learning_stats["train/alpha"].append(stats["train/alpha"])


            epoch_stats = {k: np.mean(v) for k, v in epoch_stats.items()}

            all_epochs_stats["mse"].append(epoch_stats["mse"])
            if "overestimation" in epoch_stats:
                all_epochs_stats["overestimation"].append(epoch_stats["overestimation"])
            if "alpha" in epoch_stats:
                all_epochs_stats["alpha"].append(epoch_stats["alpha"])


            print(f"Epoch {epoch + 1}: mse={epoch_stats['mse']:.4f}")


        # (D) save model
        model_name = f"scoring_model_{i}.pt"
        torch.save(model.state_dict(), os.path.join(save_dir, model_name))
        print(f"Saved model #{i+1} to {model_name}")
        # (E) save learning stats
        logs_dir = os.path.join(save_dir, "logs_model_{}".format(i))
        learning_stats_name = f"learning_stats_{i}.npz"
        os.makedirs(logs_dir, exist_ok=True)
        np.savez(os.path.join(logs_dir, learning_stats_name), **learning_stats)
        plot_learning_curves(learning_stats, logs_dir, i)
        plot_learning_curves(all_epochs_stats, logs_dir, i,
                             title="Learning Curve (Epochs)",
                             x_label="Epoch",
                             y_label="Score",
                             legend_loc="upper left",
                             show_legend=True, save_fig=True, fig_name="learning_curve_epochs")
        # plot only alpha and overestimation
        if "overestimation" in all_epochs_stats:
            com_logs_epochs = {
                "overestimation": all_epochs_stats["overestimation"],
                "alpha": all_epochs_stats["alpha"]
            }
            plot_learning_curves(com_logs_epochs, logs_dir, i,
                                 title="Learning Curve (Epochs)",
                                 x_label="Epoch",
                                 y_label="Score",
                                 legend_loc="upper left",
                                 show_legend=True, save_fig=True, fig_name="learning_curve_epochs_com_logs")
            com_logs_minibatch = {
                "overestimation": learning_stats["train/overestimation"],
                "alpha": learning_stats["train/alpha"]
            }
            plot_learning_curves(com_logs_minibatch, logs_dir, i,
                                    title="Learning Curve (Minibatch)",
                                    x_label="Minibatch",
                                    y_label="Score",
                                    legend_loc="upper left",
                                    show_legend=True, save_fig=True, fig_name="learning_curve_minibatch_com_logs")
        # plot only mse
        plot_learning_curves({'mse': all_epochs_stats['mse']}, logs_dir, i,
                                title="Learning Curve (Epochs)",
                                x_label="Epoch",
                                y_label="Score",
                                legend_loc="upper left",
                                show_legend=True, save_fig=True, fig_name="learning_curve_epochs_mse")
        plot_learning_curves({'mse': learning_stats['train/mse']}, logs_dir, i,
                                title="Learning Curve (Minibatch)",
                                x_label="Minibatch",
                                y_label="Score",
                                legend_loc="upper left",
                                show_legend=True, save_fig=True, fig_name="learning_curve_minibatch_mse")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=str, default=DEFAULT_X_PATH)
    parser.add_argument("--y", type=str, default=DEFAULT_Y_PATH)
    parser.add_argument("--save-dir", type=str, default="./scoring_models")
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--L", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--use-conservative", action="store_true", default=True,
                        help="Use conservative COM training")
    args = parser.parse_args()

    main(
        data_x_path=args.x,
        data_y_path=args.y,
        save_dir=args.save_dir,
        seq_len=args.seq_len,
        K=args.K,
        L=args.L,
        use_conservative=args.use_conservative,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device="cuda"
    )
