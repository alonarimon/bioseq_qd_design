from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from openelm.configs import FitnessBioEnsembleConfig, ModelConfig, FitnessHelixMRNAConfig
from openelm.environments.bioseq.bioseq import logger
from openelm.environments.bioseq.utr_fitness_function.fitness_model import get_fitness_model
import logging

bioseq_base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent.parent

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.getLogger("helical").setLevel(logging.WARNING)


# Create log directory with date
log_dir = bioseq_base_dir / "logs" / "fitness_funcs_tests" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir.mkdir(parents=True, exist_ok=True)

# Create file handler inside the dated folder
file_handler = logging.FileHandler(log_dir / 'summary.log')
file_handler.setLevel(logging.DEBUG)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler (prints to terminal)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

def test_fitness_funcs(fitness_configs: ModelConfig, val_x: np.ndarray, val_y: np.ndarray):
    """
    Test the fitness functions for the given configurations.
    Args:
        fitness_configs (ModelConfig): The fitness function configurations.
        val_x (np.ndarray): The input data for validation.
        val_y (np.ndarray): The expected output data for validation.
    """
    logger.info("Testing fitness functions...")
    logger.info(f"fitness_configs: {fitness_configs}")
    fitness_func = get_fitness_model(fitness_configs)
    fitness_func.eval()
    device = 'cuda' if fitness_configs.cuda else 'cpu'
    logger.info(f"Using device: {device}")
    fitness_func.to(device)
    # val_x = torch.from_numpy(val_x).to(device)
    # val_y = torch.from_numpy(val_y) #todo: need to fix this and make them get tensors?
    logger.info(f"val_x shape: {val_x.shape}")
    logger.info(f"val_y shape: {val_y.shape}")
    # curently the fitness gets batch size of 1 only # todo
    outputs = []
    for i in tqdm(range(val_x.shape[0]), desc="Testing fitness functions", unit="sample"):
        fitness_func_output = fitness_func(val_x[i])
        outputs.append(fitness_func_output)
    outputs = np.array(outputs)
    val_y = val_y.squeeze()
    logger.info(f"fitness_func_output shape: {outputs.shape}")

    # calculate mse, spearman_corr and pearson_corr
    mse = mean_squared_error(val_y, outputs)
    spearman_corr, _ = spearmanr(val_y, outputs)
    pearson_corr, _ = pearsonr(val_y, outputs)

    logger.info(f"\n"
      f"Metrics on validation set:\n"
      f"MSE = {mse:.6f}\n"
      f"Spearman = {spearman_corr:.6f}\n"
      f"Pearson = {pearson_corr:.6f}")

if __name__ == "__main__":
    logger.info(f"base_dir: {bioseq_base_dir}")
    utr_relabeled_data_path = bioseq_base_dir / "design-bench-detached" / "design_bench_data" / "utr" / "oracle_data" / "original_v0_minmax_orig" / "sampled_offline_relabeled_data" / "sampled_data_fraction_1_3_seed_42"
    logger.info(f"utr_relabeled_data_path: {utr_relabeled_data_path}")
    val_x = np.load(utr_relabeled_data_path / "sampled_validation_x.npy")
    val_y = np.load(utr_relabeled_data_path / "sampled_validation_y.npy")
    logger.info(f"val_x shape: {val_x.shape}")
    logger.info(f"val_y shape: {val_y.shape}")

    # Test the fitness functions
   # test_fitness_funcs(FitnessBioEnsembleConfig(), val_x, val_y)
   # test_fitness_funcs(FitnessHelixMRNAConfig(), val_x, val_y)

    # costume configs
    fitness_configs = FitnessBioEnsembleConfig()
    logger.info("fitness ensemble with beta 0.0")
    fitness_configs.beta = 0.0
    test_fitness_funcs(fitness_configs, val_x, val_y)
    logger.info("fitness ensemble with beta 0.0 and ensemble size 1")
    fitness_configs.ensemble_size = 1
    test_fitness_funcs(fitness_configs, val_x, val_y)


