from dataclasses import dataclass

@dataclass
class HelixFineTuneConfig:
    batch_size: int = 1
    epochs: int = 1
    max_length: int = 50
    loss: str = "mse"
    output_size: int = 1
    device: str = "cuda"
    alphabet: list[str] = ("A", "C", "G", "U")
    data_dir: str = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\bioseq_qd_design\design-bench-detached\design_bench_data\utr\oracle_data\original_v0_minmax_orig\sampled_offline_relabeled_data\sampled_data_fraction_1_3_seed_42" # todo: not absolute path
    save_base_dir: str = "experiments/helix_finetune"
