import copy
import os
from typing import Any
import numpy as np
import logging

from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.disk_resource import DiskResource
from design_bench.oracles.tensorflow import ResNetOracle
from scipy.spatial.distance import squareform

from openelm.algorithms.map_elites import CVTMAPElites
from openelm.configs import ELMConfig, FitnessBioEnsembleConfig, FitnessHelixMRNAConfig, MAPElitesConfig, MutatorHelixConfig, QDBioRNAEnvConfig, CVTMAPElitesConfig, BioRandomModelConfig, QDConfig
from openelm.mutation_model import RandomSequenceMutator

ORACLE_NAME = "original_v0_minmax_orig"
DATASET_PATH = r"/design-bench-detached/design_bench_data/utr"

logger = logging.getLogger(__name__)


def cast_fitness_model(cfg: Any):
    if isinstance(cfg, FitnessBioEnsembleConfig) or isinstance(cfg, FitnessHelixMRNAConfig):
        return cfg
    if isinstance(cfg, dict):
        if cfg.get("model_type") == "bio_ensemble":
            return FitnessBioEnsembleConfig(**cfg)
        elif cfg.get("model_type") == "helix_mrna":
            return FitnessHelixMRNAConfig(**cfg)
    raise ValueError("Unknown fitness_model config")

def cast_mutation_model(cfg: Any):
    if isinstance(cfg, BioRandomModelConfig) or isinstance(cfg, MutatorHelixConfig):
        return cfg
    if isinstance(cfg, dict):
        if cfg.get("model_name") == "bio_random":
            return BioRandomModelConfig(**cfg)
        elif cfg.get("model_name") == "mutator_helix_mrna":
            return MutatorHelixConfig(**cfg)
    raise ValueError("Unknown mutation_model config")

def cast_qd_config(cfg: Any):
    if isinstance(cfg, QDConfig):
        return cfg
    if isinstance(cfg, dict):
        if cfg.get("qd_name") == "cvtmapelites":
            return CVTMAPElitesConfig(**cfg)
        elif cfg.get("qd_name") == "mapelites":
            return MAPElitesConfig(**cfg)
    raise ValueError("Unknown QD config")

def cast_env_config(cfg: Any):
    if isinstance(cfg, QDBioRNAEnvConfig):
        return cfg
    if isinstance(cfg, dict):
        if cfg.get("env_name") == "qd_bio_rna":
            return QDBioRNAEnvConfig(**cfg)
    raise ValueError("Unknown env config")

def cast_elm_config(cfg: Any):
    if isinstance(cfg, ELMConfig):
        return cfg
    if isinstance(cfg, dict):
        return ELMConfig(
            env=cast_env_config(cfg.get("env")),
            qd=cast_qd_config(cfg.get("qd")),
            fitness_model=cast_fitness_model(cfg.get("fitness_model")),
            mutation_model=cast_mutation_model(cfg.get("mutation_model")),
            run_name=cfg.get("run_name"),
            wandb_group=cfg.get("wandb_group"),
            wandb_mode=cfg.get("wandb_mode"),)

def load_oracle(dataset_path, oracle_name):
    oracle_data_path = os.path.join(dataset_path, "oracle_data")
    oracle_data_path = os.path.join(oracle_data_path, oracle_name)
    # Load validation split
    val_x = [DiskResource(os.path.join(oracle_data_path, 'oracle_train_split', "split-val-x-0.npy"))]
    val_y = [DiskResource(os.path.join(oracle_data_path, 'oracle_train_split', "split-val-y-0.npy"))]
    val_dataset = DiscreteDataset(val_x, val_y, num_classes=4)
    oracle_model_path = os.path.join(oracle_data_path, "oracle")

    # Load the saved oracle (fit=False ensures it loads from disk)
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

    return oracle

def loaf_ref_list(x_data_path, size_to_sample, seed=42):
    # Load the reference set from the offline data directory
    np.random.seed(seed)
    offline_data_x = np.load(x_data_path)
    random_indexes = np.random.choice(offline_data_x.shape[0], size=size_to_sample, replace=False)
    reference_set = offline_data_x[random_indexes]

    return reference_set

def get_conflicted_pairs(list_of_solutions, scondary_structures, distances, distances_ss, scores, save_dir):

    # show pairs who had high distance in the secondary structure but low in the primary structure
    # find the indexes of the pairs who had high distance in the secondary structure but low in the primary structure
    distance_matrix = squareform(distances)
    distance_matrix_ss = squareform(distances_ss)
    conflicted_pairs = []
    for i in range(len(list_of_solutions)):
        for j in range(i + 1, len(list_of_solutions)):
            if abs(distance_matrix[i][j] - distance_matrix_ss[i][j]) > 20:
                conflicted_pairs.append((i, j))
    # save the pairs to a file
    with open(os.path.join(save_dir, "conflicted_pairs_distances.txt"), "w") as f:
        for pair in conflicted_pairs:
            f.write(f"indexes {pair}: {distance_matrix[pair[0]][pair[1]]} [primarily] vs {distance_matrix_ss[pair[0]][pair[1]]} [secondary] \n")
            f.write(f"seq1: {str(list_of_solutions[pair[0]])}\n")
            f.write(f"seq2: {str(list_of_solutions[pair[1]])}\n")
            f.write(f"secondary structure 1: {scondary_structures[pair[0]]}\n")
            f.write(f"secondary structure 2: {scondary_structures[pair[1]]}\n")
            f.write(f"score1: {scores[pair[0]]}, score2: {scores[pair[1]]}\n")
            f.write("=" * 50 + "\n")

def downsample_solutions(genomes, k, save_dir, original_config: ELMConfig):
    logger.info(f"Downsampling {len(genomes)} solutions to {k} solutions")
    logger.info(f"Fitness model type: {type(original_config.fitness_model)}")  # Should be FitnessBioEnsembleConfig or FitnessHelixMRNAConfig
    logger.info(f"Mutation model type: {type(original_config.mutation_model)}")  # Should be RandomSequenceMutator or MutatorHelix

    downsampled_config = copy.deepcopy(original_config.qd)
    downsampled_config.cvt_samples = len(genomes)
    downsampled_config.n_niches = k
    downsampled_config.output_dir = os.path.join(save_dir, "downsampled_map")
    from openelm.environments.bioseq.bioseq import BioSeqEvolution
    logger.info(f"Downsampled QD config: {downsampled_config}")
    bioseq_env = BioSeqEvolution(config=original_config.env, 
                                 mutation_model_config=original_config.mutation_model,
                                 fitness_model_config=original_config.fitness_model,)
    downsampled_map = CVTMAPElites(
        env=bioseq_env,
        config=downsampled_config,
        data_to_init=genomes,
        name="downsampled_map",
    )
    phenotypes = [bioseq_env.to_phenotype(genotype) for genotype in genomes]
    # Insert solutions from original map into the new downsampled map
    for genotype, phenotype in zip(genomes, phenotypes):
        map_ix = downsampled_map.to_mapindex(phenotype)
        if map_ix is not None:
            fitness = downsampled_map.env.fitness(genotype)
            if fitness > downsampled_map.fitnesses[map_ix]:
                downsampled_map.fitnesses[map_ix] = fitness
                downsampled_map.genomes[map_ix] = genotype
                downsampled_map.nonzero[map_ix] = True

    downdampled_genomes = downsampled_map.genomes.array[downsampled_map.nonzero.array]
    # make sure we have enough solutions after down-sampling
    if len(downdampled_genomes) < k:
        logger.warning(f"Downsampled map has only {len(downdampled_genomes)} solutions, "
                       f"while {k} were requested. Sampling more solutions randomly from the map.")
        has_enough_solutions = (len(genomes) >= k)
        if not has_enough_solutions:
            logger.warning(
                f"Map has only {len(genomes)} solutions, while {k} were requested."
                f" Sampling more solutions randomly from the map.")
        for i in range((k - len(downdampled_genomes))):
            map_ix = np.random.choice(np.arange(len(genomes)), size=1)[0]
            while genomes[map_ix] in downdampled_genomes and has_enough_solutions:
                map_ix = np.random.choice(np.arange(len(genomes)), size=1)[0]
            downdampled_genomes = np.append(downdampled_genomes, genomes[map_ix])

    return downdampled_genomes
