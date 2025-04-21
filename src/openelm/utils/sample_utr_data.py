import numpy as np
import torch
import os

from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.disk_resource import DiskResource
from design_bench.oracles.tensorflow import ResNetOracle

utr_data_dir = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\bioseq_qd_design\design-bench-detached\design_bench_data\utr"  #todo: not absolute path

def relabel_utr_data(data_dir:str, oracle_name:str):
    """
    Relabel UTR data using the specified oracle.
    @param data_dir: Directory containing the UTR data files.
    @param oracle_name: Name of the oracle to use for relabeling.
    """
    x_files = sorted([f for f in os.listdir(data_dir) if f.startswith("utr-x")])
    y_files = sorted([f for f in os.listdir(data_dir) if f.startswith("utr-y")])

    # Load and concatenate
    X_all = np.concatenate([np.load(os.path.join(data_dir, f)) for f in x_files], axis=0)
    Y_all = np.concatenate([np.load(os.path.join(data_dir, f)) for f in y_files], axis=0)

    print("X_all shape:", X_all.shape)
    print("Y_all shape:", Y_all.shape)
    print("X_all:", X_all[:3])
    print("Y_all:", Y_all[:3])

    oracle_data_path = os.path.join(data_dir, "oracle_data", oracle_name)
    # Load validation split
    val_x = [DiskResource(os.path.join(oracle_data_path, 'oracle_train_split', "split-val-x-0.npy"))]
    val_y = [DiskResource(os.path.join(oracle_data_path, 'oracle_train_split', "split-val-y-0.npy"))]
    val_dataset = DiscreteDataset(val_x, val_y, num_classes=4)
    print("Validation dataset shape:", val_dataset.x.shape)  # (19999, 50)
    print("Validation dataset y shape:", val_dataset.y.shape)  # (19999, 1)
    print("Validation dataset x:", val_dataset.x[:3])
    print("Validation dataset y:", val_dataset.y[:3])

    # Relabel using the oracle
    oracle_model_path = os.path.join(oracle_data_path, "oracle")
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

    Y_relabelled = oracle.predict(X_all)

    print("Y_relabelled shape:", Y_relabelled.shape)
    print("Y_relabelled:", Y_relabelled[:5])

    # Save relabelled data
    np.save(os.path.join(oracle_data_path, "relabelled_y.npy"), Y_relabelled)
    np.save(os.path.join(oracle_data_path, "x.npy"), X_all)

def sample_utr_data(oracle_dir:str, fraction:float=1/3, seed:int=42):
    """
    Sample a subset of UTR data from the specified directory.
    @param oracle_dir: Directory containing the oracle data files.
    @param fraction: Fraction of data to sample.
    @param seed: Random seed for reproducibility.
    @param oracle_name: Name of the oracle used for relabeling.
    @return: Tuple of sampled X and Y data.
    """
    x_files = sorted([f for f in os.listdir(oracle_dir) if f.startswith("x")])
    y_files = sorted([f for f in os.listdir(oracle_dir) if f.startswith("relabelled_y")])
    y_files_debug = sorted([f for f in os.listdir(oracle_dir) if f.startswith("utr-tensorflow_resnet-y")])

    # Load and concatenate
    X_all = np.concatenate([np.load(os.path.join(oracle_dir, f)) for f in x_files], axis=0)
    Y_all = np.concatenate([np.load(os.path.join(oracle_dir, f)) for f in y_files], axis=0)
    Y_all_debug = np.concatenate([np.load(os.path.join(oracle_dir, f)) for f in y_files_debug], axis=0)

    print("X_all shape:", X_all.shape)
    print("Y_all shape:", Y_all.shape)
    print("Y_all min:", np.min(Y_all), "max:", np.max(Y_all), "mean:", np.mean(Y_all), "std:", np.std(Y_all))
    print("Y_all:", Y_all[:3])
    print("Y_all_debug shape:", Y_all_debug.shape)
    print("Y_all_debug min:", np.min(Y_all_debug), "max:", np.max(Y_all_debug), "mean:", np.mean(Y_all_debug), "std:", np.std(Y_all_debug))
    print("Y_all_debug:", Y_all_debug[:3])

    sampled_x, sampled_y, non_sampled_x, non_sampled_y, indices = sample_weighted_subset(X_all, Y_all, fraction=fraction, seed=seed)


    save_dir = os.path.join(oracle_dir, "sampled_offline_relabeled_data", "sampled_data_fraction_{}_seed_{}".format(fraction, seed))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "x.npy"), sampled_x)
        np.save(os.path.join(save_dir, "y.npy"), sampled_y)
        np.save(os.path.join(save_dir, "non_sampled_x.npy"), non_sampled_x)
        np.save(os.path.join(save_dir, "non_sampled_y.npy"), non_sampled_y)

    return sampled_x, sampled_y, non_sampled_x, non_sampled_y, indices

def sample_weighted_subset(X, Y, fraction=1/3, seed=42):
    """
    Sample a weighted subset of the data.
    @param X: Input data.
    @param Y: Labels.
    @param fraction: Fraction of data to sample.
    @param seed: Random seed for reproducibility.
    @return: Tuple of sampled X and Y data, non-sampled X and Y data, and indices of sampled data.
    """
    torch.manual_seed(seed)
    scores = torch.tensor(Y, dtype=torch.float32).squeeze()
    weights = torch.exp(-scores)
    weights /= weights.sum()
    N = int(len(X) * fraction)
    indices = torch.multinomial(weights, N, replacement=False)
    sampled_x = X[indices]
    sampled_y = Y[indices]
    non_sampled_indices = torch.ones(len(X), dtype=torch.bool)
    non_sampled_indices[indices] = False
    non_sampled_x = X[non_sampled_indices]
    non_sampled_y = Y[non_sampled_indices]
    print("Sampled indices:", indices.shape, 'non sampled indices:', non_sampled_indices.shape)
    print("Sampled X shape:", sampled_x.shape, "Sampled Y shape:", sampled_y.shape)
    print("Non-sampled X shape:", non_sampled_x.shape, "Non-sampled Y shape:", non_sampled_y.shape)
    return sampled_x, sampled_y, non_sampled_x, non_sampled_y, indices

def sample_validation_dataset(non_sampled_x, non_sampled_y, fraction=1/3, seed=42, save_dir=None):
    """
    Sample randomly a validation dataset from the non-sampled data.
    @param non_sampled_x: Non-sampled input data.
    @param non_sampled_y: Non-sampled labels.
    @param fraction: Fraction of data to sample.
    @param seed: Random seed for reproducibility.
    @return: Tuple of sampled X and Y data for validation.
    """
    torch.manual_seed(seed)
    N = int(len(non_sampled_x) * fraction)
    indices = torch.randperm(len(non_sampled_x))[:N]
    sampled_x = non_sampled_x[indices]
    sampled_y = non_sampled_y[indices]
    print("Sampled validation indices:", indices.shape)
    print("Sampled validation X shape:", sampled_x.shape, "Sampled validation Y shape:", sampled_y.shape)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "sampled_validation_x.npy"), sampled_x)
        np.save(os.path.join(save_dir, "sampled_validation_y.npy"), sampled_y)
    return sampled_x, sampled_y

# this script should be called once to relabel the data and then sample it to create the offline relabeled dataset
if __name__ == "__main__":
    relabel_utr_data(utr_data_dir, "original_v0_minmax_orig")
    oracle_dir = os.path.join(utr_data_dir, "oracle_data", "original_v0_minmax_orig")
    sampled_x, sampled_y, non_sampled_x, non_sampled_y, indices = sample_utr_data(oracle_dir, fraction=1/3, seed=42)
    sampled_validation_x, sampled_validation_y = (
        sample_validation_dataset(non_sampled_x, non_sampled_y, fraction=1/3, seed=42,
                                  save_dir=os.path.join(oracle_dir, "sampled_offline_relabeled_data", "sampled_data_fraction_1_3_seed_42")))

    print("Sampled data saved to:", oracle_dir)

    # make sure no overlaps between sampled_validation_x and sampled_x
    print("Sampled validation X shape:", sampled_validation_x.shape)
    print("Sampled X shape:", sampled_x.shape)
    val_set = set(map(tuple, sampled_validation_x))
    train_set = set(map(tuple, sampled_x))
    overlap = val_set & train_set
    print("Number of overlapping sequences:", len(overlap))
    if overlap:
        print("Overlapping sequences (up to 5 shown):", list(overlap)[:5])
    else:
        print("Sampled validation X and X are disjoint ✅")


