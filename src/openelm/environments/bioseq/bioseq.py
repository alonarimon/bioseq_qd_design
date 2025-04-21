import os
from typing import Optional

import numpy as np
import torch

from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.disk_resource import DiskResource
from design_bench.oracles.tensorflow import ResNetOracle
from openelm.configs import QDEnvConfig, QDBioRNAEnvConfig
from openelm.environments.base import BaseEnvironment
from openelm.mutation_model import MutationModel, get_model
from openelm.environments.bioseq.utr_fitness_function.fitness_model import get_fitness_model
from openelm.utils.evaluation import evaluate_solutions_set

MAP_INT_TO_LETTER = {
    0: "A",
    1: "C",
    2: "G",
    3: "U",
} # todo: check if this is correct


class RNAGenotype:
    """
    A simple genotype class for RNA bioseq generation. (without llms)
    """

    def __init__(self, sequence: list[int], min_bd: float, max_bd: float):
        self.sequence = sequence
        self.min_bd = min_bd
        self.max_bd = max_bd

    def _nucleotides_frequencies(self) -> np.ndarray:
        """
        Calculate the frequencies of nucleotides in a sequence.
        :return: numpy array with frequencies of the nucleotides 0, 1, 2
        """
        freq = np.zeros(3, dtype=float)
        for letter in self.sequence:
            if letter == 0:
                freq[0] += 1
            elif letter == 1:
                freq[1] += 1
            elif letter == 2:
                freq[2] += 1
        # Normalize frequencies
        freq /= len(self.sequence)
        # normalize according to the min and max values and clip to [0, 1]
        freq = (freq - self.min_bd) / (self.max_bd - self.min_bd)
        freq = np.clip(freq, 0, 1)
        return freq

    def to_phenotype(self, bd_type = "nucleotides_frequencies") -> Optional[np.ndarray]:
        """
        Convert sequence to a phenotype representation (behavioral descriptor) according to the specified type.
        :param bd_type: str (e.g. 'nucleotides_frequencies')
        return: np.ndarray with dtype float
        """
        if bd_type == "nucleotides_frequencies":
            return self._nucleotides_frequencies()
        else:
            raise ValueError(f"Unknown bd_type: {bd_type}. Supported: nucleotides_frequencies")

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
        print(f"Using device: {self.device}")
        # set all random seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        


        # self.projection_matrix = # todo: for similarity-based bd
        self.bd_type = config.bd_type # behavioral descriptor type (e.g. 'nucleotides_frequencies')
        if self.config.normalize_bd:
            self.bd_min, self.bd_max = self.get_training_bd_stats()  # Get the min and max values for the behavioral descriptor space
        else:
            self.bd_min = 0
            self.bd_max = 1
        self.oracle = self._load_oracle()  # Load the oracle model from disk, for final evaluation on the solutions (not used in the optimization process)
        # todo: here they originally had 'del mutation_model'. see if it is needed (and if it does - delete it outside the constructor)


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

        print("Oracle params:\n",
              "rank_correlation:", oracle.params["rank_correlation"],
              "\nmodel_kwargs:", oracle.params["model_kwargs"],
              "\nsplit_kwargs:", oracle.params["split_kwargs"])

        return oracle

    def get_training_bd_stats(self) -> tuple:
        """
        Get the training behavioral descriptor statistics.
        :return: tuple of min and max values for the behavioral descriptor space.
        """
        offline_data_x = np.load(os.path.join(self.offline_data_dir, self.config.offline_data_x_file))
        offline_data_genotypes = [RNAGenotype(seq, 0, 1) for seq in offline_data_x]
        # Calculate the behavioral descriptor for each genotype
        bd_values = np.array([genotype.to_phenotype(self.bd_type) for genotype in offline_data_genotypes])
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
        """ #todo: when comparing between methods, this should be the same seeds all the time
        offline_data_x = np.load(os.path.join(self.offline_data_dir, self.config.offline_data_x_file))
        random_indexes = self.rng.choice(offline_data_x.shape[0], size=self.batch_size, replace=False)
        initial_sequences = offline_data_x[random_indexes]
        initial_genotypes = [RNAGenotype(seq, min_bd=self.bd_min, max_bd=self.bd_max) for seq in initial_sequences]
        print("Initial sequences[:5]:\n", [str(s) for s in initial_genotypes[:5]])
        print("index of initial sequences:\n", random_indexes[:5])
        return initial_genotypes

    def _load_ref_set(self) -> list[RNAGenotype]:
        """
        Load the reference set from the offline data directory.
        :return: list of RNAGenotype
        """
        # Load the reference set from the offline data directory
        offline_data_x = np.load(os.path.join(self.offline_data_dir, self.config.offline_data_x_file))
        random_indexes = self.rng.choice(offline_data_x.shape[0], size=self.config.size_of_refs_collection, replace=False)
        reference_set = offline_data_x[random_indexes]
        reference_set = [RNAGenotype(seq,  min_bd=0, max_bd=1) for seq in reference_set]
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
        return [RNAGenotype(self._random_seq(), min_bd=self.bd_min, max_bd=self.bd_max) for _ in range(self.batch_size)]

    def mutate(self, genomes: list[RNAGenotype]) -> list[RNAGenotype]:
        """
        Mutate a list of genomes by applying the mutation function to each genome.
        """
        return [RNAGenotype(self._mutate_seq(g.sequence), min_bd=self.bd_min, max_bd=self.bd_max) for g in genomes] #todo: change to use mutation model

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

