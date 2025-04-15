import os

from design_bench.disk_resource import DiskResource
from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.oracles.tensorflow import ResNetOracle
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
import numpy as np

ORACLE_NAME = "resnet_k8_normalized_minmax_and_z_2"

dataset_path = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\OpenELM_GenomicQD\design-bench_forked\design_bench_data\utr"
oracle_data_path = os.path.join(dataset_path, "oracle_data")
oracle_data_path = os.path.join(oracle_data_path, ORACLE_NAME)

# Load validation split
val_x = [DiskResource(os.path.join(oracle_data_path, "split-val-x-0.npy"))]
val_y = [DiskResource(os.path.join(oracle_data_path, "split-val-y-0.npy"))]
val_dataset = DiscreteDataset(val_x, val_y, num_classes=4)

print("Validation dataset shape:", val_dataset.x.shape)  # (19999, 50)
print("Validation dataset y shape:", val_dataset.y.shape)  # (19999, 1)
print("Validation dataset x:", val_dataset.x[:5])
print("Validation dataset y:", val_dataset.y[:5])

oracle_model_path = os.path.join(oracle_data_path, "oracle")

# Load the saved oracle (fit=False ensures it loads from disk)
oracle = ResNetOracle(
    val_dataset,
    noise_std=0.0,
    fit=False,  # do not retrain
    is_absolute=True,
    disk_target=oracle_model_path

)

print("Oracle params:\n",
      "rank_correlation:", oracle.params["rank_correlation"],
        "\nmodel_kwargs:", oracle.params["model_kwargs"],
        "\nsplit_kwargs:", oracle.params["split_kwargs"])

# Evaluate on validation data
y_pred = oracle.predict(val_dataset.x).squeeze()
y_true = val_dataset.y.squeeze()

# Metrics
mse = mean_squared_error(y_true, y_pred)
spearman_corr, _ = spearmanr(y_true, y_pred)
pearson_corr, _ = pearsonr(y_true, y_pred)

print("Validation y stats:")
print("min:", y_true.min(), "max:", y_true.max(), "mean:", y_true.mean(), "std:", y_true.std())
print(f"\n"
      f"Metrics on validation set:\n"
      f"MSE = {mse:.6f}\n"
      f"Spearman = {spearman_corr:.6f}\n"
      f"Pearson = {pearson_corr:.6f}")

# print performace on the sampled data
sampled_data_path = os.path.join(dataset_path, "sampled_data_fraction_1_3_seed_42")
sampled_x = np.load(os.path.join(sampled_data_path, "sampled_x.npy"))
sampled_y = np.load(os.path.join(sampled_data_path, "sampled_y.npy"))
sampled_y = sampled_y.squeeze()
print("Sampled data shape:", sampled_x.shape)  # (19999, 50)
print("Sampled data y shape:", sampled_y.shape)  # (19999, 1)
# Predict on sampled data
sampled_y_pred = oracle.predict(sampled_x).squeeze()
sampled_y_true = sampled_y.squeeze()
print(f"sampled_y_pred[:5]:\n{sampled_y_pred[:5]},\n"
      f"sampled_y_true[:5]:\n{sampled_y_true[:5]}")
# Metrics
mse = mean_squared_error(sampled_y_true, sampled_y_pred)
spearman_corr, _ = spearmanr(sampled_y_true, sampled_y_pred)
pearson_corr, _ = pearsonr(sampled_y_true, sampled_y_pred)
print("Sampled data y stats:")
print("min:", sampled_y_true.min(), "max:", sampled_y_true.max(), "mean:", sampled_y_true.mean(), "std:", sampled_y_true.std())
print(f"\n"
      f"Metrics on sampled data:\n"
      f"MSE = {mse:.6f}\n"
      f"Spearman = {spearman_corr:.6f}\n"
      f"Pearson = {pearson_corr:.6f}")
