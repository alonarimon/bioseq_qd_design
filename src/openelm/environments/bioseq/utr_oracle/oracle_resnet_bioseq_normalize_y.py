import os

import numpy as np

from design_bench.datasets.discrete.utr_dataset import UTRDataset
from design_bench.disk_resource import DiskResource
from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.oracles.tensorflow import ResNetOracle

utr_data_path = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\OpenELM_GenomicQD\design-bench_forked\design_bench_data\utr" #todo: not absolute path
# Load UTR dataset
x_orig_files = sorted([f for f in os.listdir(utr_data_path) if f.startswith("utr-x")])
y_orig_files = sorted([f for f in os.listdir(utr_data_path) if f.startswith("utr-y")])
# Load and concatenate
X_orig_all = np.concatenate([np.load(os.path.join(utr_data_path, f)) for f in x_orig_files], axis=0)
Y_orig_all = np.concatenate([np.load(os.path.join(utr_data_path, f)) for f in y_orig_files], axis=0)
print("X_orig_all shape:", X_orig_all.shape)
print("Y_orig_all shape:", Y_orig_all.shape)
print("X_orig_all:", X_orig_all[:3])
print("Y_orig_all:", Y_orig_all[:3])


dataset = UTRDataset()
print('=====after loading to UTRDataset========')
print("Dataset x shape:", dataset.x.shape)  # (280000, 50)
print("Dataset y shape:", dataset.y.shape)  # (280000, 1)
print("Dataset x:", dataset.x[:3])
print("Dataset y:", dataset.y[:3])


# relabel and save normalized y values to disk
def normalize_y_minmax(x, y):
    return (y - y.min()) / (y.max() - y.min())
dataset.relabel(
    normalize_y_minmax,
    to_disk=True,
    disk_target="utr/normalize_y_minmax_debug",
    is_absolute=False
)
print('=====after relabeling========')
print("Dataset x shape:", dataset.x.shape)  # (280000, 50)
print("Dataset y shape:", dataset.y.shape)  # (280000, 1)
print("Dataset x:", dataset.x[:3])
print("Dataset y:", dataset.y[:3])

ORACLE_NAME = "debug"

# train resnet oracle on normalized dataset
oracle = ResNetOracle(
    dataset,
    noise_std=0.0,
    internal_batch_size=128,
    max_samples=None,
    distribution=None,
    max_percentile=100,
    min_percentile=0,
    fit=True,
    is_absolute=False,
    disk_target=os.path.join("utr/oracle_data", ORACLE_NAME, "oracle"),
    model_kwargs=dict(
        hidden_size=120,
        num_blocks=2,
        activation='relu',
        learning_rate=0.001,
        epochs=5,
        batch_size=128,
        shuffle_buffer=260000,
        kernel_size=8,
    ),
    split_kwargs=dict(val_fraction=0.07142857142,
                          subset=None,
                          shard_size=280000,
                          to_disk=True,
                          disk_target=os.path.join("utr/oracle_data", ORACLE_NAME, "split"),
                          is_absolute=False))


