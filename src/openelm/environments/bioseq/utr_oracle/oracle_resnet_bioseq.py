import os

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error

import design_bench

original_utr_data_X = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\OpenELM_GenomicQD\design-bench_forked\design_bench_data\utr\normalized_minmax_Y\utr-x.npy"
original_utr_data_y = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\OpenELM_GenomicQD\design-bench_forked\design_bench_data\utr\normalized_minmax_Y\utr-y.npy"
original_utr_data_X = np.load(original_utr_data_X)
original_utr_data_y = np.load(original_utr_data_y)
print("Original UTR data X shape:", original_utr_data_X.shape)
print("Original UTR data y shape:", original_utr_data_y.shape)
print("Original UTR data X:", original_utr_data_X[:5])
print("Original UTR data y:", original_utr_data_y[:5])


task = design_bench.make("UTR-ResNet-v0", relabel=False) #todo: make sure its the same architecture and traininng as in the paper

X, y = task.x, task.y
print("=====after relabeling========")
print("x shape:", X.shape)        # (140000, 50)
print("y shape:", y.shape)        # (140000, 1)
print("x:", X[:5])
print("y:", y[:5])

# an instance of the DatasetBuilder class from design_bench.datasets.dataset_builder
dataset = task.dataset

# an instance of the OracleBuilder class from design_bench.oracles.oracle_builder
oracle = task.oracle

# check how the model was fit
print("Oracle params:\n",
      "rank_correlation:", oracle.params["rank_correlation"],
        "\nmodel_kwargs:", oracle.params["model_kwargs"],
        "\nsplit_kwargs:", oracle.params["split_kwargs"])

path_to_oracle_data = r'C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\OpenELM_GenomicQD\design-bench_forked\design_bench_data\utr/oracle_data/original_v0_minmax_orig_2/oracle_train_split/'  #todo: not absolute path
# Load the oracle data
oracle_train_x = np.load(os.path.join(path_to_oracle_data, "split-train-x-0.npy"))
oracle_train_y = np.load(os.path.join(path_to_oracle_data, "split-train-y-0.npy"))
oracle_val_x = np.load(os.path.join(path_to_oracle_data, "split-val-x-0.npy"))
oracle_val_y = np.load(os.path.join(path_to_oracle_data, "split-val-y-0.npy"))
print("Oracle train x shape:", oracle_train_x.shape)  # (260001, 50)
print("Oracle train y shape:", oracle_train_y.shape)  # (260001, 1)
print("Oracle val x shape:", oracle_val_x.shape)      # (19999, 50)
print("Oracle val y shape:", oracle_val_y.shape)      # (19999, 1)

# evaluate oracle on the validation set
y_val_true = oracle_val_y
y_val_pred = oracle.predict(oracle_val_x)
print("y_val_true shape:", y_val_true.shape)  # (19999, 1)
print("y_val_pred shape:", y_val_pred.shape)  # (19999, 1)
print("y_val_true:", y_val_true[:5])
print("y_val_pred:", y_val_pred[:5])
# Calculate metrics
mse = mean_squared_error(y_val_true, y_val_pred)
spearman_corr, _ = spearmanr(y_val_true, y_val_pred)
pearson_corr = pearsonr(y_val_true, y_val_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")
print(f"Pearson Correlation: {pearson_corr.statistic.item():.4f}, p-value: {pearson_corr.pvalue.item():.4f}")



