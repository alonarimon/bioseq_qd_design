from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pathlib import Path

bioseq_base_dir = Path(__file__).resolve().parents[2]


@dataclass
class BaseConfig:
    output_dir: str = "logs/"


@dataclass
class ModelConfig(BaseConfig):
    fp16: bool = True
    cuda: bool = True
    gpus: int = 1
    seed: Optional[int] = None
    deterministic: bool = False
    top_p: float = 0.95
    temp: float = 1.1
    gen_max_len: int = 512
    batch_size: int = 128
    model_type: str = "bio_random"  # Can be "hf", "openai", etc
    model_path: str = MISSING  # Can be HF model name or path to local model
    logits_only: bool = False
    do_sample: bool = True
    num_return_sequences: int = 1
    trust_remote_code: bool = True  # needed for mosaicml/mpt-7b-instruct
    load_in_8bit: bool = False  # need to install bitsandbytes
    load_in_4bit: bool = False

    def __post_init__(self):
        model_path = Path(self.model_path)
        if not model_path.is_absolute():
            self.model_path = str((bioseq_base_dir / model_path).resolve())


@dataclass
class PromptModelConfig(ModelConfig):
    model_name: str = "prompt"
    model_path: str = "Salesforce/codegen-350M-mono"


@dataclass
class DiffModelConfig(ModelConfig):
    model_name: str = "diff"
    model_path: str = "CarperAI/diff-codegen-350m-v2"

@dataclass
class BioRandomModelConfig(ModelConfig):
    model_name: str = "bio_random"
    model_path: str = "bio_random" #todo

@dataclass
class FitnessBioEnsembleConfig(ModelConfig):
    model_type: str = "bio_ensemble"  # Can be "hf", "openai", etc
    model_name: str = "fitness_bio_ensemble"
    model_path: str = r"src/openelm/environments/bioseq/utr_fitness_function/one_shot_scoring_ensemble/scoring_models"  # Path to the scoring model
    ensemble_size: int = 4 # Number of scoring models to use for fitness evaluation #todo: 18
    beta: float = 2.0  # Penalty term factor
    alphabet_size: int = 4 # Size of the alphabet (e.g., 4 for nucleotides ACGU)
    sequence_length: int = 50 # Length of the sequence to be evaluated
    batch_size: int = 128

@dataclass
class FitnessHelixMRNAConfig(ModelConfig):
    model_type: str = "helix_mrna"
    model_name: str = "fitness_helix_mrna"
    model_path: str = r"logs\helix_mrna_fine_tune\exp_2025-04-21_11-15-25\model"
    batch_size: int = 1

@dataclass
class QDConfig(BaseConfig):
    init_steps: int = 1 
    total_steps: int =  5000  #100000
    history_length: int = 1
    save_history: bool = False
    save_snapshot_interval: int = 1000
    log_snapshot_dir: str = ""
    seed: Optional[int] = 42
    save_np_rng_state: bool = False
    load_np_rng_state: bool = False
    crossover: bool = False
    crossover_parents: int = 2
    eval_with_oracle_on_snapshot: bool = True
    number_of_final_solutions: int = 128
    eval_with_oracle_interval : int = 5000 # evaluation with oracle is usually slow, so recommend to do it not often


@dataclass
class MAPElitesConfig(QDConfig):
    qd_name: str = "mapelites"
    map_grid_size: tuple[int, ...] = field(default_factory=lambda: (5,))


@dataclass
class CVTMAPElitesConfig(QDConfig):
    qd_name: str = "cvtmapelites"
    n_niches: int = 2000
    cvt_samples: int = 10000


@dataclass
class EnvConfig(BaseConfig):
    timeout: float = 5.0  # Seconds
    sandbox: bool = False
    sandbox_server: str = "http://localhost:5000"
    processes: int = 1
    batch_size: int = 128  # Batch size of MAP-Elites
    env_name: str = MISSING
    debug: bool = False
    seed: Optional[int] = 42


@dataclass
class SodaraceEnvConfig(EnvConfig):
    env_name: str = "sodarace"
    eval_ms: int = 1000  # Milliseconds
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            # Height, Width, Mass dimensions
            [0, 500],
            [0, 500],
            [0, 1000],
        ]
    )
    starting_seeds: list[str] = field(default_factory=lambda: ["square"])
    instruction: int = 2
    crossover: bool = False


@dataclass
class ImageEnvConfig(EnvConfig):
    env_name: str = "image_evolution"
    behavior_mode: str = "3-channel"
    target: str = "circle"


@dataclass
class StringEnvConfig(EnvConfig):
    env_name: str = "string_evolution"
    target: str = "MapElites"


@dataclass
class P3ProblemEnvConfig(EnvConfig):
    env_name: str = "p3_problem"
    prompt_size: str = "long"  # med or long
    timeout: float = 1.0  # timeout for running a solution
    starting_seed: int = field(
        default_factory=lambda: 3
    )  # index of p3 dataset to use as puzzle to mutate
    embedding_model_type: str = "hf"  # openai or hf
    embedding_model_path: str = MISSING  # e.g. hf: Salesforce/codegen-350M-mono ; openai: text-embedding-ada-002


@dataclass
class P3ProbSolEnvConfig(EnvConfig):
    env_name: str = "p3_probsol"
    prompt_size: str = "long"  # med or long
    timeout: float = 1.0  # timeout for running a solution
    starting_seed: int = field(
        default_factory=lambda: 3
    )  # index of p3 dataset to use as puzzle to mutate
    eval_k: int = 100  # k for pass@k for fitness
    embedding_model_type: str = "hf"  # openai or hf
    embedding_model_path: str = MISSING  # e.g. hf: Salesforce/codegen-350M-mono ; openai: text-embedding-ada-002


@dataclass
class QDEnvConfig(EnvConfig):
    env_name: str = "qdaif"
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            [0, 5],
            [0, 5],
        ]
    )

@dataclass
class QDBioRNAEnvConfig(EnvConfig):
    env_name: str = "qd_bio_rna"
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            [0, 1],
            [0, 1],
        ]
    )
    sequence_length: int = 50
    alphabet: list[int] = field(default_factory=lambda: [0, 1, 2, 3]) # [A, C, G, U]
    size_of_refs_collection: int =  2048 # Number of reference sequences to use for novelty evaluation and BD
    offline_data_dir: str = r"design-bench-detached\design_bench_data\utr\oracle_data\original_v0_minmax_orig\sampled_offline_relabeled_data\sampled_data_fraction_1_3_seed_42"  # Path to the offline data directory
    offline_data_x_file: str = "x.npy"  # Name of the offline data X file
    offline_data_y_file: str = "y.npy"  # Name of the offline data Y file
    oracle_model_path: str = r"design-bench-detached\design_bench_data\utr\oracle_data\original_v0_minmax_orig"  # Path to the oracle model
    fitness_model_config: ModelConfig = field(default_factory=lambda: FitnessBioEnsembleConfig())
    bd_type: str = "similarity_based" #"nucleotides_frequencies": The phenotype is a vector of frequencies of the letters A, C, G (U can be inferred). "similarity_based": The phenotype is a vector of the similarity of the sequence to the offline ref data.
    normalize_bd: bool = True  # Whether to normalize the behavior descriptor according the offline data min-max
    distance_normalization_constant: float = 14.3378899  # Constant for distance normalization (for the similarity-based BD). -1 means constant will be automatically calculated from the offline data.
    initial_population_sample_seed: int = 123  # initial population sample seed

    def __post_init__(self):
        path_fields = ['offline_data_dir', 'oracle_model_path']
        for field_name in path_fields:
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                if isinstance(value, str):
                    path = Path(value)
                    if not path.is_absolute():
                        setattr(self, field_name, str((bioseq_base_dir / path).resolve()))

@dataclass
class PromptEnvConfig(EnvConfig):
    env_name: str = "prompt_evolution"
    task_name: str = "antonym"  # toy or antonym or animal or cot
    evals_per_prompt: int = 10


defaults_elm = [
    {"model": "bio_random"},
    {"qd": "cvtmapelites"},
    {"env": "qd_bio_rna"},
    "_self_",
]


@dataclass
class ELMConfig(BaseConfig):
    hydra: Any = field(
        default_factory=lambda: {
            "run": {
                "dir": "logs/elm/${hydra.job.override_dirname}/${now:%y-%m-%d_%H-%M}"
            }
        }
    )
    defaults: list[Any] = field(default_factory=lambda: defaults_elm)
    model: Any = MISSING
    qd: Any = MISSING
    env: Any = MISSING
    run_name: Optional[str] = None


defaults_p3 = [
    {"model": "prompt"},
    {"env": "p3"},
    "_self_",
]


@dataclass
class P3Config(BaseConfig):
    hydra: Any = field(
        default_factory=lambda: {
            "run": {
                "dir": "logs/p3/${hydra.job.override_dirname}/${now:%y-%m-%d_%H-%M}"
            }
        }
    )
    defaults: list[Any] = field(default_factory=lambda: defaults_p3)
    model: Any = MISSING
    env: Any = MISSING
    run_name: Optional[str] = None
    # --- The below are for run_p3.py
    iterations_per_puzzle: int = 128
    starting_seeds: list[int] = field(
        default_factory=lambda: [3]
    )  # indices of selection of puzzles to evaluate with
    save_results: bool = True
    save_result_obj: bool = False  # if saving results, include the whole output
    # text from model for each iteration (which can get long)
    probsol: bool = True  # generate new problem+solution pairs from given
    # problems instead of just solutions to given problems
    # set eval_k >0 to evaluate pass@k of previous runs using this k, instead of
    # doing a new run
    eval_k: int = -1
    eval_timestamp: str = ""  # optionally provide timestamp of run to eval
    # pass@k, otherwise eval with latest run of every problem


def register_configstore() -> ConfigStore:
    """Register configs with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(group="env", name="sodarace", node=SodaraceEnvConfig)
    cs.store(group="env", name="image_evolution", node=ImageEnvConfig)
    cs.store(group="env", name="string_evolution", node=StringEnvConfig)
    cs.store(group="env", name="p3_probsol", node=P3ProbSolEnvConfig)
    cs.store(group="env", name="p3_problem", node=P3ProblemEnvConfig)
    cs.store(group="env", name="prompt_evolution", node=PromptEnvConfig)
    cs.store(group="env", name="qdaif", node=QDEnvConfig)
    cs.store(group="env", name="qd_bio_rna", node=QDBioRNAEnvConfig)
    cs.store(group="qd", name="mapelites", node=MAPElitesConfig)
    cs.store(group="qd", name="cvtmapelites", node=CVTMAPElitesConfig)
    cs.store(group="model", name="prompt", node=PromptModelConfig)
    cs.store(group="model", name="diff", node=DiffModelConfig)
    cs.store(group="model", name="bio_random", node=BioRandomModelConfig)
    cs.store(group="fitness_model", name="fitness_bio_ensemble", node=FitnessBioEnsembleConfig)
    cs.store(name="elmconfig", node=ELMConfig)
    cs.store(name="p3config", node=P3Config)
    return cs


CONFIGSTORE = register_configstore()
