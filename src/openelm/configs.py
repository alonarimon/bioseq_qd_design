from dataclasses import dataclass, field
from datetime import datetime
import os
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pathlib import Path

bioseq_base_dir = Path(__file__).resolve().parents[2]


@dataclass
class BaseConfig:
    output_dir: str = "logs/"


@dataclass
class ModelConfig(BaseConfig): #TODO: go over all and remove unused
    fp16: bool = True
    cuda: bool = True
    gpus: int = 1
    seed: Optional[int] = None
    deterministic: bool = False
    top_p: float = 0.95
    temp: float = 1.1
    gen_max_len: int = MISSING
    batch_size: int = 128
    model_name: str = MISSING
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
class BioRandomModelConfig(ModelConfig):
    model_name: str = "random"
    model_path: str = ""
    alphabet: list[int] = field(default_factory=lambda: [0, 1, 2, 3]) # [A, C, G, U]
    mutation_length: int = 5
    gen_max_len: int = 50


@dataclass 
class MutatorHelixConfig(ModelConfig):
    model_name: str = "helix"
    model_path: str = "" #TODO
    tokenizer_path: str = "" #TODO
    mutation_length: int = 5
    temp : float = 0.6
    top_k: int = 1
    logits_threshold: float = 0.8
    top_p: float = 0.0 # 0.0 = no top-p samplings
    batch_size: int = 128 #TODO!
    gen_max_len: int = 50

@dataclass
class FitnessBioEnsembleConfig(ModelConfig): 
    model_type: str = "bio_ensemble" 
    model_name: str = MISSING
    model_path: str = MISSING
    gen_max_len: int = MISSING # Length of the sequence to be evaluated
    ensemble_size: int = 4 # Number of scoring models to use for fitness evaluation #todo: 18
    beta: float = 2.0  # Penalty term factor
    alphabet_size: int = 4 # Size of the alphabet (e.g., 4 for nucleotides ACGU)
    batch_size: int = 128
    load_existing_models: bool = True # Whether to load existing models from the model path

    # retraining parameters
    validation_fraction: float = 0.1 # Fraction of data to use for validation during retraining
    use_conservative: bool = True # Whether to use conservative training
    epochs: int = 50 # Number of epochs for retraining

@dataclass
class FitnessUTREnsembleConfig(FitnessBioEnsembleConfig): 
    model_name: str = "ensemble"
    model_path: str = os.path.join("src", "openelm", "environments", "bioseq", "utr_fitness_function", "utr_scoring_ensemble", "scoring_models")
    gen_max_len: int = 50 # Length of the sequence to be evaluated

@dataclass
class FitnessTFBind10EnsembletConfig(FitnessBioEnsembleConfig): 
    model_name: str = "ensemble"
    model_path: str = os.path.join("src", "openelm", "environments", "bioseq", "tfbind_fitness_function", "tfbind_scoring_ensemble", "scoring_models", "2025-05-21_15-24-34") #TODO: without the date
    gen_max_len: int = 10 # Length of the sequence to be evaluated
    batch_size: int = 128
    use_conservative: bool = True  # Whether to use conservative training
    

@dataclass
class FitnessHelixMRNAConfig(ModelConfig):
    model_type: str = "helix_mrna"
    model_name: str = "helix"
    model_path: str = os.path.join(bioseq_base_dir, "logs", "helix_mrna_fine_tune", "exp_2025-05-12_16-25-14", "model")
    batch_size: int = 128
    gen_max_len: int = 50

@dataclass
class QDConfig(BaseConfig):
    init_steps: int = 1 
    total_steps: int =  35000 
    history_length: int = 1
    save_history: bool = False
    save_snapshot_interval: int = 100
    log_snapshot_dir: str = ""
    seed: Optional[int] = 42
    save_np_rng_state: bool = False
    load_np_rng_state: bool = False
    crossover: bool = False
    crossover_parents: int = 2
    eval_with_oracle: bool = True
    visualize_on_interval: bool = False  # Whether to visualize the QD archive on a regular interval
    number_of_final_solutions: int = 128
    save_logs_localy: bool = True


@dataclass
class MAPElitesConfig(QDConfig):
    qd_name: str = "mapelites"
    map_grid_size: tuple[int, ...] = field(default_factory=lambda: (5,))


@dataclass
class CVTMAPElitesConfig(QDConfig):
    qd_name: str = "cvtmapelites"
    n_niches: int = 2000
    cvt_samples: int = 10000
    init_from_offline_data: bool = True  # Whether to initialize the map from offline data


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
class QDEnvConfig(EnvConfig):
    env_name: str = "qdaif"
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            [0, 5],
            [0, 5],
        ]
    )

@dataclass
class QDBioEnvConfig(EnvConfig):
    env_name: str = "qd_bio_env" # todo: naming
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            [0, 1],
            [0, 1],
            [0, 1],
        ]
    )
    sequence_length: int = MISSING
    alphabet: list[int] = field(default_factory=lambda: [0, 1, 2, 3]) # [A, C, G, U]
    size_of_refs_collection: int =  16384 # Number of reference sequences to use for novelty evaluation and BD
    bd_type: str = "freq" #"freq": The phenotype is a vector of frequencies of the letters A, C, G (U can be inferred). "similarity_based": The phenotype is a vector of the similarity of the sequence to the offline ref data.
    normalize_bd: bool = True  # Whether to normalize the behavior descriptor according the offline data min-max
    distance_normalization_constant: float = MISSING  # Constant for distance normalization (for the similarity-based BD). -1 means constant will be automatically calculated from the offline data.
    task: str = MISSING # 'UTR-ResNet-v0-CUSTOM' or 'TFBind10-Exact-v1'
    retrain_fitness_model: bool = False # Whether to retrain the fitness model with the offline data

@dataclass
class QDBioTaskBasedEnvConfig(QDBioEnvConfig):
    env_name: str = "qd_bio_dna"
    distance_normalization_constant: float = -1  # Constant for distance normalization (for the similarity-based BD). -1 means constant will be automatically calculated from the offline data.
    task: str = 'TFBind10-Exact-v1'
    sequence_length: int = 10
    retrain_fitness_model: bool = True
    
    
@dataclass
class QDBioUTREnvConfig(QDBioEnvConfig):
    env_name: str = "qd_bio_utr"
    offline_data_dir: str = os.path.join("design-bench-detached", "design_bench_data", "utr", "oracle_data", "original_v0_minmax_orig", "sampled_offline_relabeled_data", "sampled_data_fraction_1_3_seed_42")
    offline_data_x_file: str = "x.npy"  # Name of the offline data X file
    offline_data_y_file: str = "y.npy"  # Name of the offline data Y file
    oracle_model_path: str = os.path.join("design-bench-detached", "design_bench_data", "utr", "oracle_data", "original_v0_minmax_orig")  # Path to the oracle model
    oracle_max_score: float = 0.7381 # Max score of the oracle model over UTR dataset
    oracle_min_score: float = 0.1885 # Min score of the oracle model over UTR dataset
    distance_normalization_constant: float = 14.3378899  # Constant for distance normalization (for the similarity-based BD). -1 means constant will be automatically calculated from the offline data.
    bd_min: list[float] = field(default_factory=lambda: []) # Min values for the BD, empty list means the min values will be calculated from the offline data
    bd_max: list[float] = field(default_factory=lambda:  []) # Max values for the BD, empty list means the max values will be calculated from the offline data
    task: str = 'UTR-ResNet-v0-CUSTOM'
    sequence_length: int = 50
    
    def __post_init__(self):
        path_fields = ['offline_data_dir', 'oracle_model_path']
        for field_name in path_fields:
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                if isinstance(value, str):
                    path = Path(value)
                    if not path.is_absolute():
                        setattr(self, field_name, os.path.join(bioseq_base_dir, value))


defaults_elm = [
    {"qd": "cvtmapelites"},
    {"env": "qd_bio_utr"},
    {"mutation_model": "mutator_helix"}, # can be "random" or "mutator_helix"
    {"fitness_model": "fitness_utr_ensemble"}, # can be "fitness_helix_mrna" or "fitness_utr_ensemble" or "fitness_TFBind10_ensemble"
    "_self_",
]


@dataclass
class ELMConfig(BaseConfig):
    hydra: Any = field(
        default_factory=lambda: {
            "run": {
                # Fallback if override_dirname is not set
                "dir": "logs/elm/${now:%y-%m-%d_%H-%M}"
            }
        }
    )
    mutation_model: Any = MISSING #TODO: move to env
    fitness_model: Any = MISSING #TODO: move to env
    qd: Any = MISSING
    env: Any = MISSING
    wandb_group: str = "run_elm"
    run_name: str = ""
    wandb_mode: str = "online"
  

@dataclass
class OneShotBioELMConfig(ELMConfig):
    defaults: list[Any] = field(default_factory=lambda: [
        {"mutation_model": "random"},
        {"fitness_model": "fitness_utr_ensemble"},
        {"qd": "cvtmapelites"},
        {"env": "qd_bio_utr"},
        "_self_",
    ])
    qd: Any = field(default_factory=lambda: CVTMAPElitesConfig(
        qd_name="cvtmapelites",
        n_niches=2000,
        cvt_samples=10000,
        init_steps=1,
        total_steps=100000,
        history_length=1,
        save_history=False,
        save_snapshot_interval=5000,
        log_snapshot_dir="",
        seed=42,
        save_np_rng_state=False,
        load_np_rng_state=False,
        crossover=False,
        crossover_parents=2,
        eval_with_oracle=True,
        number_of_final_solutions=128,
        init_from_offline_data=False,
    ))
    env: Any = field(default_factory=lambda: QDBioUTREnvConfig(
        env_name="qd_bio_utr",
        behavior_space=[
            [0, 1],
            [0, 1],
            [0, 1]
        ],
        sequence_length=50,
        alphabet=[0, 1, 2, 3], # [A, C, G, U]
        size_of_refs_collection=16384,
        offline_data_dir= os.path.join("design-bench-detached", "design_bench_data", "utr", "oracle_data", "original_v0_minmax_orig", "sampled_offline_relabeled_data", "sampled_data_fraction_1_3_seed_42"),
        offline_data_x_file= "x.npy",
        offline_data_y_file= "y.npy",
        oracle_model_path= os.path.join("design-bench-detached", "design_bench_data", "utr", "oracle_data", "original_v0_minmax_orig"), 
        bd_type="freq",
        normalize_bd=True,
        seed=42,
        distance_normalization_constant=14.3378899,
        task='UTR-ResNet-v0-CUSTOM',
        retrain_fitness_model=False 
    ))
    mutation_model: Any = field(default_factory=lambda: BioRandomModelConfig(
        model_name="random",
        model_path="",  
        alphabet=[0, 1, 2, 3], # [A, C, G, U]
        mutation_length=1,
        gen_max_len=50,
    ))


@dataclass
class OneShotSimilarityBDELMConfig(OneShotBioELMConfig):
    env: Any = field(default_factory=lambda: QDBioUTREnvConfig(
        env_name="qd_bio_utr",
        behavior_space=[
            [0, 1],
            [0, 1]
        ],
        sequence_length=50,
        alphabet=[0, 1, 2, 3], # [A, C, G, U]
        size_of_refs_collection=16384,
        offline_data_dir= os.path.join("design-bench-detached", "design_bench_data", "utr", "oracle_data", "original_v0_minmax_orig", "sampled_offline_relabeled_data", "sampled_data_fraction_1_3_seed_42"),
        offline_data_x_file= "x.npy",
        offline_data_y_file= "y.npy",
        oracle_model_path= os.path.join("design-bench-detached", "design_bench_data", "utr", "oracle_data", "original_v0_minmax_orig"),
        bd_type="similarity_based",
        normalize_bd=True,
        distance_normalization_constant=14.3378899,
        seed=42,
        task='UTR-ResNet-v0-CUSTOM',
        retrain_fitness_model= False 
    ))


def register_configstore() -> ConfigStore:
    """Register configs with Hydra's ConfigStore."""
    cs = ConfigStore.instance()

    cs.store(group="env", name="qdaif", node=QDEnvConfig)
    cs.store(group="env", name="qd_bio_env", node=QDBioEnvConfig)
    cs.store(group="env", name="qd_bio_utr", node=QDBioUTREnvConfig)
    cs.store(group="env", name="qd_bio_utr_similarity_based", node=QDBioUTREnvConfig)
    cs.store(group="env", name="qd_bio_dna", node=QDBioTaskBasedEnvConfig)
    cs.store(group="qd", name="mapelites", node=MAPElitesConfig)
    cs.store(group="qd", name="cvtmapelites", node=CVTMAPElitesConfig)
    cs.store(group="mutation_model", name="random", node=BioRandomModelConfig)
    cs.store(group="mutation_model", name="mutator_helix", node=MutatorHelixConfig)
    cs.store(group="fitness_model", name="fitness_utr_ensemble", node=FitnessUTREnsembleConfig)
    cs.store(group="fitness_model", name="fitness_TFBind10_ensemble", node=FitnessTFBind10EnsembletConfig)
    cs.store(group="fitness_model", name="fitness_helix_mrna", node=FitnessHelixMRNAConfig)
    cs.store(name="elmconfig", node=ELMConfig)
    cs.store(name="oneshot_bio_elmconfig", node=OneShotBioELMConfig)
    cs.store(name="oneshot_similarity_bd_elmconfig", node=OneShotSimilarityBDELMConfig)
    
    # from yaml files
    cs.store(name="train_TFBind10_ensemble", node=ELMConfig)
    cs.store(name="utr_task", node=ELMConfig)
    cs.store(name="utr_task_similarity_based", node=ELMConfig)
    cs.store(name="dna_task", node=ELMConfig)
    cs.store(name="debug", node=ELMConfig)


    return cs


CONFIGSTORE = register_configstore()
