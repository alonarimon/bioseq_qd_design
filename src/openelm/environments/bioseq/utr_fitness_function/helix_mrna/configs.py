from dataclasses import dataclass

import os
import sys
import pathlib

project_root = os.path.abspath(os.path.join(pathlib.Path(__file__).parent.absolute(), "../../../../../.."))

@dataclass
class HelixFineTuneConfig:
    debug: bool = False,
    wandb: bool = True,
    seed: int = 42,
    batch_size: int = 32
    val_batch_size: int = 1
    val_fraction: float = 0.0
    epochs: int = 50
    max_length: int = 50
    loss: str = "mse"
    output_size: int = 1
    device: str = "cuda"
    alphabet: list[str] = ("A", "C", "G", "U")
    data_dir: str = os.path.join(project_root, "design-bench-detached", "design_bench_data", "utr", "oracle_data", "original_v0_minmax_orig", "sampled_offline_relabeled_data", "sampled_data_fraction_1_3_seed_42")
    save_base_dir: str = "logs/helix_mrna_fine_tune"
