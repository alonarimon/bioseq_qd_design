import os

from design_bench.datasets.discrete.utr_dataset import UTRDataset
from design_bench.disk_resource import DiskResource
from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.oracles.tensorflow import ResNetOracle

# relabel and save normalized y values to disk
def normalize_y_minmax(x, y):
    return (y - y.min()) / (y.max() - y.min())

dataset = UTRDataset()
dataset.relabel(
    normalize_y_minmax,
    to_disk=True,
    disk_target="utr/oracle_data/normalize_y_minmax",
    is_absolute=False
)

ORACLE_NAME = "resnet_k8_normalized_minmax_and_z"

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


