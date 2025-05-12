from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from openelm.configs import FitnessBioEnsembleConfig, ModelConfig, FitnessHelixMRNAConfig
from openelm.environments.bioseq.bioseq import logger, RNAGenotype
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

EXPECTED_RESULTS = {
    "fitness_bio_ensemble": {
        4: {"mse": 0.007004, "spearman": 0.400000, "pearson": 0.424333}, # first 4 samples
        None: {"mse": 0.008355, "spearman": 0.767470, "pearson": 0.823375}, # all samples
    },
    "fitness_helix_mrna": {
        4: {"mse": 0.001624, "spearman": 0.800000, "pearson": 0.882930}, # first 4 samples
        None: {"mse": 0.003593, "spearman": 0.874574, "pearson": 0.884289}, # all samples
    }
}


@pytest.fixture(scope="module")
def validation_data_utr():
    logger.info(f"base_dir: {bioseq_base_dir}")
    utr_relabeled_data_path = bioseq_base_dir / "design-bench-detached" / "design_bench_data" / "utr" / "oracle_data" / "original_v0_minmax_orig" / "sampled_offline_relabeled_data" / "sampled_data_fraction_1_3_seed_42"
    logger.info(f"utr_relabeled_data_path: {utr_relabeled_data_path}")

    val_x = np.load(utr_relabeled_data_path / "sampled_validation_x.npy")
    val_y = np.load(utr_relabeled_data_path / "sampled_validation_y.npy")

    return val_x, val_y

def evaluate_fitness_funcs(fitness_configs: ModelConfig, val_x: np.ndarray, val_y: np.ndarray):
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
    val_x_genotypes = [RNAGenotype(sequence=seq) for seq in val_x]
    # val_y = torch.from_numpy(val_y) #todo: need to fix this and make them get tensors?
    logger.info(f"val_x shape: {val_x.shape}")
    logger.info(f"val_y shape: {val_y.shape}")
    # go over batch size
    outputs = []
    number_of_batches = val_x.shape[0] / fitness_func.config.batch_size
    number_of_batches = int(np.ceil(number_of_batches)) # round up
    for i in tqdm(range(number_of_batches), desc="Processing batches"):
        # get the batch
        batch_start = i * fitness_func.config.batch_size
        batch_end = (i + 1) * fitness_func.config.batch_size if i < number_of_batches - 1 else val_x.shape[0]
        batch_x = val_x_genotypes[batch_start:batch_end]
        fitness_func_output = fitness_func(batch_x)
        outputs.append(fitness_func_output)

    outputs = np.concatenate(outputs, axis=0)

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

    return mse, spearman_corr, pearson_corr

@pytest.mark.parametrize("n_samples", [4, None])  # int for number of samples or None for all
@pytest.mark.parametrize("config_class", [FitnessBioEnsembleConfig, FitnessHelixMRNAConfig])
def test_fitness_funcs_on_val(validation_data_utr, config_class, n_samples):

    val_x, val_y = validation_data_utr

    if n_samples is not None:
        val_x = val_x[:n_samples]
        val_y = val_y[:n_samples]

    config = config_class()

    logger.info(f"Testing {config.model_name} on {len(val_x)} samples")
    mse, spearman_corr, pearson_corr = evaluate_fitness_funcs(config, val_x, val_y)

    expected = EXPECTED_RESULTS[config.model_name][n_samples]

    assert mse == pytest.approx(expected["mse"], abs=1e-6)
    assert spearman_corr == pytest.approx(expected["spearman"], abs=1e-6)
    assert pearson_corr == pytest.approx(expected["pearson"], abs=1e-6)

    logger.info(f"Test passed for {config.model_name} with {n_samples} samples!")






