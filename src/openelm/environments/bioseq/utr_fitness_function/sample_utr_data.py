import numpy as np
import torch
import os


utr_data_dir = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\OpenELM_GenomicQD\design-bench_forked\design_bench_data\utr" #todo: not absolute path


def sample_utr_data(data_dir:str, fraction:float=1/3, seed:int=42, save_dir:str=None):
    """
    Sample a subset of UTR data from the specified directory.
    @param data_dir: Directory containing the UTR data files.
    @param fraction: Fraction of data to sample.
    @param seed: Random seed for reproducibility.
    @param save_dir: Directory to save the sampled data.
    @return: Tuple of sampled X and Y data.
    """
    x_files = sorted([f for f in os.listdir(data_dir) if f.startswith("utr-x")])
    y_files = sorted([f for f in os.listdir(data_dir) if f.startswith("utr-tensorflow_resnet")])

    # Load and concatenate
    X_all = np.concatenate([np.load(os.path.join(data_dir, f)) for f in x_files], axis=0)
    Y_all = np.concatenate([np.load(os.path.join(data_dir, f)) for f in y_files], axis=0)

    print("X_all shape:", X_all.shape)
    print("Y_all shape:", Y_all.shape)

    sampled_x, sampled_y = sample_weighted_subset(X_all, Y_all, fraction=fraction, seed=seed)

    print("Sampled X shape:", sampled_x.shape)
    print("Sampled Y shape:", sampled_y.shape)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "sampled_x.npy"), sampled_x)
        np.save(os.path.join(save_dir, "sampled_y.npy"), sampled_y)

    return sampled_x, sampled_y

def sample_weighted_subset(X, Y, fraction=1/3, seed=42):
    torch.manual_seed(seed)
    scores = torch.tensor(Y, dtype=torch.float32).squeeze()
    weights = torch.exp(-scores)
    weights /= weights.sum()
    N = int(len(X) * fraction)
    indices = torch.multinomial(weights, N, replacement=False)
    return X[indices], Y[indices]

# this script should be called once to sample the data after the data was relabeled and saved to disk
if __name__ == "__main__":
    save_dir = os.path.join(utr_data_dir, "sampled_data_fraction_1_3_seed_42")
    sampled_x, sampled_y = sample_utr_data(utr_data_dir, fraction=1/3, seed=42, save_dir=save_dir)
    print("Sampled data saved to:", save_dir)
    print(f"sampled_x[:5]:\n{sampled_x[:5]}")
    print(f"sampled_y[:5]:\n{sampled_y[:5]}")
