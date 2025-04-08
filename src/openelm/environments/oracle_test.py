from design_bench.disk_resource import DiskResource
from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.oracles.tensorflow import ResNetOracle
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
import numpy as np

# Load validation split
val_x = [DiskResource(r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\OpenELM_GenomicQD\design-bench_forked\design_bench_data\utr\oracle_data\resnet_normalized0_split-val-x-0.npy")]
val_y = [DiskResource(r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\OpenELM_GenomicQD\design-bench_forked\design_bench_data\utr\oracle_data\resnet_normalized0_split-val-y-0.npy")]
val_dataset = DiscreteDataset(val_x, val_y, num_classes=4)

print("Validation dataset shape:", val_dataset.x.shape)  # (19999, 50)
print("Validation dataset y shape:", val_dataset.y.shape)  # (19999, 1)
print("Validation dataset x:", val_dataset.x[:5])
print("Validation dataset y:", val_dataset.y[:5])

# Load the saved oracle (fit=False ensures it loads from disk)
oracle = ResNetOracle(
    val_dataset,
    noise_std=0.0,
    fit=False,  # <- important: do not retrain
    is_absolute=False,
    disk_target="utr/oracle_resnet_v0_normalized"

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
