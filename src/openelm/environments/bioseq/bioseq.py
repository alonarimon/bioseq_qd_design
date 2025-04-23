import logging
import os
from typing import Optional

import numpy as np
import torch
from scipy.special import softmax
from tqdm import tqdm
import Levenshtein

from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.disk_resource import DiskResource
from design_bench.oracles.tensorflow import ResNetOracle

from openelm.configs import QDEnvConfig, QDBioRNAEnvConfig
from openelm.environments.base import BaseEnvironment, Phenotype, Genotype
from openelm.mutation_model import MutationModel, get_model
from openelm.environments.bioseq.utr_fitness_function.fitness_model import get_fitness_model
from openelm.utils.evaluation import evaluate_solutions_set

logger = logging.getLogger()

MAP_INT_TO_LETTER = {
    0: "A",
    1: "C",
    2: "G",
    3: "U",
} # todo: check if this is correct


class RNAGenotype(Genotype):
    """
    A simple genotype class for RNA bioseq generation. (without llms)
    """

    def __init__(self, sequence: list[int]):
        self.sequence = sequence

    def to_phenotype(self) -> Optional[Phenotype]:
        """
        Convert the genotype to a phenotype.
        :return: Phenotype representation of the genotype.
        """
        raise NotImplementedError("Phenotype conversion is not implemented for RNAGenotype, expect to use the environment's to_phenotype method.")

    def __str__(self):
        """
        Convert the genotype to a string representation.
        """
        return "".join([MAP_INT_TO_LETTER[letter] for letter in self.sequence])


class RNAEvolution(BaseEnvironment[RNAGenotype]):
    def __init__(
            self,
            config: QDBioRNAEnvConfig,
            mutation_model: MutationModel #todo: not in use, GET CONFIG?
    ):
        """
        Args:
            config (QDEnvConfig): Configuration for the environment.
            mutation_model (MutationModel): Mutation model for mutating sequences.
        """
        super().__init__() #todo: check if this is needed
        print(f"Initializing RNAEvolution environment with config: {config}")
        self.config = config
        self.mutation_model = get_model(mutation_model.config) #todo: not in use
        self.fitness_function = get_fitness_model(config.fitness_model_config)

        self.batch_size = config.batch_size
        self.genotype_space = np.array(
            self.config.behavior_space).T  # todo: i think it should be renamed to behavior_space (in the base class)
        self.sequence_length = config.sequence_length
        self.alphabet = config.alphabet
        self.rng = np.random.default_rng(config.seed)
        self.offline_data_dir = config.offline_data_dir
        self.reference_set = self._load_ref_set()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        # set all random seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        if self.config.bd_type == "similarity_based":
            self.projection_matrix = np.random.uniform(low=0.0, high=1.0, size=(config.size_of_refs_collection, len(config.behavior_space))).astype(np.float32)  # Initialize W from N(0,1)
            logger.info(f"Projection matrix W shape: {self.projection_matrix.shape}")
            if config.distance_normalization_constant < 0:
                self.R_normalization_constant = self._get_r_distance_norm_const(subsample=True)  # todo: need to be done once without subsampling
            else:
                self.R_normalization_constant = config.distance_normalization_constant
            logger.info(f"R normalization constant: {self.R_normalization_constant}")

        self.bd_min = 0
        self.bd_max = 1
        if self.config.normalize_bd:
            subsample = self.config.bd_type == "similarity_based" # we subsample the offline data only for the similarity based bd, because it is expensive
            self.bd_min, self.bd_max = self.get_training_bd_stats(subsample=subsample)

        logger.info(f"Behavioral descriptor min: {self.bd_min}, max: {self.bd_max}")

        self.oracle = self._load_oracle()  # Load the oracle model from disk, for final evaluation on the solutions (not used in the optimization process)
        # todo: here they originally had 'del mutation_model'. see if it is needed (and if it does - delete it outside the constructor)

    def _get_r_distance_norm_const(self, subsample = False) -> float:
        """
        Calculate the normalization constant for the distance metric.
        :param subsample: if True, subsample the offline data to the size of the reference collection x 2.
        :return: Normalization constant.
        """
        dists = []
        offline_data = np.load(os.path.join(self.offline_data_dir, self.config.offline_data_x_file))
        if subsample:
            sampled_indexes = self.rng.choice(offline_data.shape[0], size=(self.config.size_of_refs_collection * 2), replace=False)
            offline_data = offline_data[sampled_indexes]
        for i in tqdm(range(len(offline_data)), desc="Calculating R normalization constant"):
            for j in range(i + 1, len(offline_data)):
                    dist = Levenshtein.distance(offline_data[i], offline_data[j])
                    dists.append(dist)

        R_normalization_constant = np.mean(dists) / 2
        return R_normalization_constant

    def _load_oracle(self):
        """
        Load the oracle model from disk.
        :return: Loaded oracle model.
        """
        # Load validation split
        val_x = [DiskResource(os.path.join(self.config.oracle_model_path, 'oracle_train_split', "split-val-x-0.npy"))]
        val_y = [DiskResource(os.path.join(self.config.oracle_model_path, 'oracle_train_split', "split-val-y-0.npy"))]
        val_dataset = DiscreteDataset(val_x, val_y, num_classes=4)
        oracle_model_path = os.path.join(self.config.oracle_model_path, "oracle")

        # Load the saved oracle (fit=False ensures it loads from disk)
        oracle = ResNetOracle(
            val_dataset,
            noise_std=0.0,
            fit=False,  # do not retrain
            is_absolute=True,
            disk_target=oracle_model_path

        )

        logger.info(
            f"Oracle params:\n"
            f"rank_correlation: {oracle.params['rank_correlation']}\n"
            f"model_kwargs: {oracle.params['model_kwargs']}\n"
            f"split_kwargs: {oracle.params['split_kwargs']}"
        )

        return oracle

    def get_training_bd_stats(self, subsample = False) -> tuple:
        """
        Get the training behavioral descriptor statistics.
        :param subsample: if True, subsample the offline data to the size of the reference collection.
        :return: tuple of min and max values for the behavioral descriptor space.
        """
        offline_data_x = np.load(os.path.join(self.offline_data_dir, self.config.offline_data_x_file))
        if subsample:
            # Subsample the offline data
            random_indexes = self.rng.choice(offline_data_x.shape[0], size=self.config.size_of_refs_collection, replace=False)
            offline_data_x = offline_data_x[random_indexes]

        offline_data_genotypes = [RNAGenotype(seq) for seq in offline_data_x]
        # Calculate the behavioral descriptor for each genotype
        bd_values = np.array([self.to_phenotype(genotype) for genotype in offline_data_genotypes])
        # Calculate the min and max values for each behavioral descriptor, for all dimensions
        min_bd = np.min(bd_values, axis=0)
        max_bd = np.max(bd_values, axis=0)

        return min_bd, max_bd

    def get_rng_state(self) -> Optional[np.random.Generator]:
        return self.rng

    def set_rng_state(self, rng_state: Optional[np.random.Generator]):
        self.rng = rng_state

    def initial_sequences(self) -> list[RNAGenotype]:
        """
        Generate a batch of initial sequences by randomly sample from the offline data.
        @return: list of RNAGenotype
        """
        offline_data_x = np.load(os.path.join(self.offline_data_dir, self.config.offline_data_x_file))
        seed = self.config.initial_population_sample_seed
        rng = np.random.default_rng(seed)  # Ensures reproducibility, not using the self.rng
        random_indexes = rng.choice(offline_data_x.shape[0], size=self.batch_size, replace=False)
        initial_sequences = offline_data_x[random_indexes]
        initial_genotypes = [RNAGenotype(seq) for seq in initial_sequences]
        logger.info(f"index of initial sequences:\n { random_indexes[:5]}")
        return initial_genotypes

    def _nucleotides_frequencies(self, x: RNAGenotype) -> np.ndarray:
        """
        Calculate the frequencies of nucleotides in a sequence.
        :return: numpy array with frequencies of the nucleotides 0, 1, 2
        """
        freq = np.zeros(3, dtype=float)
        for letter in x.sequence:
            if letter == 0:
                freq[0] += 1
            elif letter == 1:
                freq[1] += 1
            elif letter == 2:
                freq[2] += 1
        # Normalize frequencies
        freq /= len(x.sequence)

        return freq

    def _similarity_based_bd(self, x: RNAGenotype) -> np.ndarray:
        """
        Calculate the similarity-based behavioral descriptor.
        :param x: RNAGenotype
        :return: numpy array with the similarity-based behavioral descriptor
        """
        dists = np.array([-Levenshtein.distance(x.sequence, yi.sequence) for yi in self.reference_set])  # distances
        phi_n = softmax(dists)  # normalized distances via softmax
        dn = - np.dot(phi_n, dists) / self.R_normalization_constant
        bd = np.exp(dn) * (phi_n @ self.projection_matrix)
        return bd

    def to_phenotype(self, x: RNAGenotype) -> Phenotype:
        """
        Convert a genotype to a phenotype.
        :param x: genotype
        :return: phenotype
        """
        if self.config.bd_type == "nucleotides_frequencies":
            bd = self._nucleotides_frequencies(x)
        elif self.config.bd_type == "similarity_based":
            bd = self._similarity_based_bd(x)
        else:
            raise ValueError(f"Unknown bd_type: {self.config.bd_type}. Supported: nucleotides_frequencies")

        if self.config.normalize_bd:
            # normalize according to the min and max values and clip to [0, 1]
            bd = (bd - self.bd_min) / (self.bd_max - self.bd_min)

        return bd

    def _load_ref_set(self) -> list[RNAGenotype]:
        """
        Load the reference set from the offline data directory.
        :return: list of RNAGenotype
        """
        # Load the reference set from the offline data directory
        offline_data_x = np.load(os.path.join(self.offline_data_dir, self.config.offline_data_x_file))
        random_indexes = self.rng.choice(offline_data_x.shape[0], size=self.config.size_of_refs_collection, replace=False)
        reference_set = offline_data_x[random_indexes]
        reference_set = [RNAGenotype(seq) for seq in reference_set]
        return reference_set

    def _random_seq(self) -> list[int]:
        seq = [self.rng.choice(self.alphabet) for _ in range(self.sequence_length)]
        return seq

    def _mutate_seq(self, seq: list[int]) -> list[int]:
        """
        Mutate a sequence by randomly changing one letter.
        :param seq: sequence to mutate
        """ #todo: 1. change to use mutation model
        # todo: 2. the 50 steps limitation from the paper should be implemented here
        i = self.rng.integers(self.sequence_length)
        new_letter = self.rng.choice([a for a in self.alphabet if a != seq[i]])
        mutated_seq = seq.copy()
        mutated_seq[i] = new_letter
        return mutated_seq

    def random(self) -> list[RNAGenotype]:
        """
        Generate a batch of random genotypes.
        """
        return [RNAGenotype(self._random_seq()) for _ in range(self.batch_size)]

    def mutate(self, genomes: list[RNAGenotype]) -> list[RNAGenotype]:
        """
        Mutate a list of genomes by applying the mutation function to each genome.
        """
        return [RNAGenotype(self._mutate_seq(g.sequence)) for g in genomes] #todo: change to use mutation model

    def fitness(self, x: RNAGenotype) -> float:
        """
        Evaluate the fitness of the sequence using a list of scoring functions. (scoring ensemble)
        The fitness is the mean of the scores minus a penalty term.
        :param x: RNAGenotype
        :return: fitness score (float)
        """ # todo: make the fitness function work in batch mode
        fitness = self.fitness_function(x.sequence)
        return fitness

    def eval_with_oracle(self, genotypes=list[RNAGenotype], k=128) -> tuple:
        """
        Evaluate a list of genotypes using the oracle model.
        The oracle model is used to evaluate the solutions after the optimization process.
        (not used in the optimization process)
        :param genotypes: list of RNAGenotype
        :param k: number of top solutions to consider for evaluation (w.r.t. the oracle scores)
        :return: max, diversity, mean, and novelty scores for all solutions, and for the top k solutions.
        """
        N = len(genotypes)
        sequences = [genotype.sequence for genotype in genotypes]
        refs = [genotype.sequence for genotype in self.reference_set]
        list_of_solutions_np = np.array(sequences)

        # Evaluate the solutions using the oracle model #todo: switch to eval mode and use no_grad
        scores = self.oracle.predict(list_of_solutions_np).flatten()

        # calculate scores for all solutions
        max_all, diversity_all, mean_all, novelty_all = evaluate_solutions_set(sequences, refs, scores)
        # calculate scores for the top k solutions
        k = min(k, N)
        top_k_indexes = np.argsort(scores)[-k:].tolist()
        top_k_solutions = [sequences[i] for i in top_k_indexes]
        top_k_scores = [scores[i] for i in top_k_indexes]
        max_score_top_k, diversity_top_k, mean_top_k, novelty_top_k = evaluate_solutions_set(top_k_solutions, refs, top_k_scores)

        return (max_all, diversity_all, mean_all, novelty_all,
                max_score_top_k, diversity_top_k, mean_top_k, novelty_top_k)

