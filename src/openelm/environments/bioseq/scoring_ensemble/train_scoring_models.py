
import os
import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import logging

from openelm.environments.bioseq.scoring_ensemble.bioseq_log_dataset import BioseqLogDataset
from openelm.environments.bioseq.scoring_ensemble.scoring_model import ScoringNetwork
from openelm.environments.bioseq.scoring_ensemble.trainer import COMTrainer


logger = logging.getLogger(__name__)


def train_scoring_models(
    data_x: np.ndarray,
    data_y: np.ndarray,
    disk_target_data: str,
    validation_fraction: float,
    save_dir: str,
    seq_len: int,
    alphabet_size: int,
    ensemble_size: int,
    use_conservative: bool,
    device: str,
    epochs: int,
    batch_size: int,
):
    """Trains L scoring models and saves them to disk."""

    # Check if GPU is available
    if torch.cuda.is_available() and device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # 1) Load data
    if data_y.ndim == 2:  # flatten if shape is (N,1)
        data_y = data_y[:, 0]
    
    train_size = int(len(data_x) * (1 - validation_fraction))
    x_train = data_x[:train_size]
    y_train = data_y[:train_size]
    x_val = data_x[train_size:]
    y_val = data_y[train_size:]
    logger.info(f"Training data shape: {x_train.shape}, {y_train.shape}")
    logger.info(f"Validation data shape: {x_val.shape}, {y_val.shape}")

    disk_target_preprocessed_data = os.path.join(disk_target_data, "preprocessed_one_hot_log_data")
    dataset_train = BioseqLogDataset(x_train, y_train, disk_target=os.path.join(disk_target_preprocessed_data, "train"))
    dataset_val = BioseqLogDataset(x_val, y_val, disk_target=os.path.join(disk_target_preprocessed_data, "val"))
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    # Create a save directory
    os.makedirs(save_dir, exist_ok=True)

    # 2) Train L scoring models
    for i in range(ensemble_size): # todo: change to range(L)
        print(f"\n=== Training scoring model #{i+1}/{ensemble_size} ===")

        # (A) instantiate the model
        model = ScoringNetwork(seq_len, alphabet_size).to(device)

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

        for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
            epoch_stats = {"train/mse": [], "val/mse": [], "overestimation": [], "alpha": []}
            
            # (C) train
            for x_batch, y_batch in loader_train:
                stats = trainer.train_step(x_batch, y_batch)
                # update epoch stats
                epoch_stats["train/mse"].append(stats["train/mse"])
                if "train/overestimation" in stats:
                    epoch_stats["overestimation"].append(stats["train/overestimation"])
                if "train/alpha" in stats:
                    epoch_stats["alpha"].append(stats["train/alpha"])
                
            # (D) validate
            with torch.no_grad():
                for x_batch, y_batch in loader_val:
                    y_pred = model(x_batch.to(device)).cpu().numpy()
                    mse = np.mean((y_pred - y_batch.numpy()) ** 2)
                    epoch_stats["val/mse"].append(mse)

            epoch_stats = {k: np.mean(v) for k, v in epoch_stats.items()}
            wandb.log({"model": i, "epoch": epoch, **epoch_stats})

        # (D) save model
        model_name = f"scoring_model_{i}.pt"
        torch.save(model.state_dict(), os.path.join(save_dir, model_name))
        logger.info(f"Saved model #{i+1} to {model_name}")

