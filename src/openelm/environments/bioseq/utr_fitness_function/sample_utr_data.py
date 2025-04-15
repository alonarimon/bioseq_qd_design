import numpy as np
import torch
import os

from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.disk_resource import DiskResource
from design_bench.oracles.tensorflow import ResNetOracle

utr_data_dir = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\OpenELM_GenomicQD\design-bench_forked\design_bench_data\utr" #todo: not absolute path

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

    sampled_x, sampled_y = sample_weighted_subset(X_all, Y_all, fraction=fraction, seed=seed)

    print("Sampled X shape:", sampled_x.shape)
    print("Sampled Y shape:", sampled_y.shape)

    save_dir = os.path.join(oracle_dir, "sampled_offline_relabeled_data", "sampled_data_fraction_{}_seed_{}".format(fraction, seed))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "x.npy"), sampled_x)
        np.save(os.path.join(save_dir, "y.npy"), sampled_y)

    return sampled_x, sampled_y

def sample_weighted_subset(X, Y, fraction=1/3, seed=42):
    torch.manual_seed(seed)
    scores = torch.tensor(Y, dtype=torch.float32).squeeze()
    weights = torch.exp(-scores)
    weights /= weights.sum()
    N = int(len(X) * fraction)
    indices = torch.multinomial(weights, N, replacement=False)
    return X[indices], Y[indices]

# this script should be called once to relabel the data and then sample it to create the offline relabeled dataset
if __name__ == "__main__":
    #relabel_utr_data(utr_data_dir, "original_v0_minmax_orig")
    oracle_dir = os.path.join(utr_data_dir, "oracle_data", "original_v0_minmax_orig")
    sampled_x, sampled_y = sample_utr_data(oracle_dir, fraction=1/3, seed=42)
    print("Sampled data saved to:", oracle_dir)
    print(f"sampled_x[:5]:\n{sampled_x[:5]}")
    print(f"sampled_y[:5]:\n{sampled_y[:5]}")
