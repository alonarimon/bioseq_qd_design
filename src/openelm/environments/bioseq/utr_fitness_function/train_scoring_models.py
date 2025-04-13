
import os
import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset

from openelm.environments.bioseq.utr_fitness_function.utr_log_dataset import UTRLogDataset
from scoring_model import ScoringNetwork
from trainer import COMTrainer
from openelm.utils.plots import plot_learning_curves

DEFAULT_OFFLINE_DATA_PATH = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\OpenELM_GenomicQD\design-bench_forked\design_bench_data\utr\sampled_data_fraction_1_3_seed_42" #todo: not absolute path
DEFAULT_X_PATH = os.path.join(DEFAULT_OFFLINE_DATA_PATH, "sampled_x.npy")
DEFAULT_Y_PATH = os.path.join(DEFAULT_OFFLINE_DATA_PATH, "sampled_y.npy")

def main(
    data_x_path: str = DEFAULT_X_PATH,
    data_y_path: str = DEFAULT_Y_PATH,
    save_dir: str = "./scoring_models",
    seq_len: int = 50,
    K: int = 4,
    L: int = 18,
    use_conservative: bool = True,
    device: str = "cuda",
    epochs: int = 50,
    batch_size: int = 128
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

    dataset = UTRLogDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create a save directory
    os.makedirs(save_dir, exist_ok=True)

    # 2) Train L scoring models
    for i in range(L):
        print(f"\n=== Training scoring model #{i+1}/{L} ===")

        # (A) instantiate the model
        torch.manual_seed(i + 1234)  # for reproducibility
        model = ScoringNetwork(seq_len, K).to(device)

        # (B) create trainer with or without COM
        trainer = COMTrainer(
            model=model,
            lr=3e-4,
            alpha_init=0.1,
            alpha_lr=1e-2,
            overestimation_limit=0.5,
            particle_steps=50,
            particle_lr=0.05,
            entropy_coeff=0.0,
            noise_std=0.0,
            use_conservative=use_conservative,
            device=device
        )

        learning_stats = {
            "train/mse": [],
            "train/rank_corr": [],
            "train/overestimation": [],
            "train/alpha": []
        }

        # (C) train
        for epoch in range(epochs):
            epoch_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)
            epoch_stats = {"mse": [], "rank_corr": []}
            for x_batch, y_batch in loader:
                stats = trainer.train_step(x_batch, y_batch)
                # update epoch stats
                epoch_stats["mse"].append(stats["train/mse"])
                epoch_stats["rank_corr"].append(stats["train/rank_corr"])
                # update learning stats
                learning_stats["train/mse"].append(stats["train/mse"])
                learning_stats["train/rank_corr"].append(stats["train/rank_corr"])
                if "train/overestimation" in stats:
                    learning_stats["train/overestimation"].append(stats["train/overestimation"])
                if "train/alpha" in stats:
                    learning_stats["train/alpha"].append(stats["train/alpha"])

            epoch_bar.set_postfix({
                "mse": np.mean(epoch_stats["mse"]),
                "rank_corr": np.mean(epoch_stats["rank_corr"]),
            })

        # (D) save model
        model_name = f"scoring_model_{i}.pt"
        torch.save(model.state_dict(), os.path.join(save_dir, model_name))
        print(f"Saved model #{i+1} to {model_name}")
        # (E) save learning stats
        learning_stats_name = f"learning_stats_{i}.npz"
        np.savez(os.path.join(save_dir, learning_stats_name), **learning_stats)
        plot_learning_curves(learning_stats, save_dir, i)



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
        batch_size=args.batch_size
    )
