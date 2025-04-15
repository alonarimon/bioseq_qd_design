import json
import os
from typing import Optional, Callable

import numpy as np
import torch
from langchain.schema import HumanMessage

from openelm.configs import QDEnvConfig, QDBioRNAEnvConfig
from openelm.environments.base import BaseEnvironment
from openelm.environments.bioseq.utr_fitness_function.fitness_model import FitnessScoringEnsemble
from openelm.environments.prompt.prompt import PromptGenotype
from openelm.mutation_model import MutationModel, get_model
from openelm.environments.bioseq.utr_fitness_function.scoring_model import ScoringNetwork  # import the class from Step 1

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

    def __init__(self, sequence: list[int]):
        self.sequence = sequence

    def _nucleotides_frequencies(self) -> np.ndarray:
        """
        Calculate the frequencies of nucleotides in a sequence.
        :param x: RNAGenotype
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

def trivial_scoring_function(genotype: RNAGenotype) -> float:
    """
    A trivial scoring function that returns a random score.
    This is just a placeholder and should be replaced with a real scoring function.
    """
    return np.random.rand()


class RNAEvolution(BaseEnvironment[RNAGenotype]):
    def __init__(
            self,
            config: QDBioRNAEnvConfig,
            mutation_model: MutationModel,
    ):
        """
        Args:
            config (QDEnvConfig): Configuration for the environment.
            mutation_model (MutationModel): Mutation model for mutating sequences.
        """
        print(f"Initializing RNAEvolution environment with config: {config}")
        self.config = config
        self.mutation_model = get_model(mutation_model.config)
        self.batch_size = config.batch_size
        self.genotype_space = np.array(
            self.config.behavior_space).T  # todo: i think it should be renamed to behavior_space (in the base class)
        self.sequence_length = config.sequence_length
        self.alphabet = config.alphabet
        self.rng = np.random.default_rng(config.seed)
        self.offline_data_dir = config.offline_data_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.fitness_function = FitnessScoringEnsemble(self.sequence_length,
                                                        len(self.alphabet),
                                                        config.scoring_model_path,
                                                        self.device,
                                                        config.fitness_ensemble_size,
                                                        config.beta)

        # self.reference_set = reference_set # todo: for similarity-based bd
        # self.projection_matrix = # todo: for similarity-based bd
        self.beta = config.beta # penalty term factor (in the fitness function)
        self.bd_type = config.bd_type # behavioral descriptor type (e.g. 'nucleotides_frequencies')
        # todo: here they originally had 'del mutation_model'. see if it is needed (and if it does - delete it outside the constructor)



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
        initial_genotypes = [RNAGenotype(seq) for seq in initial_sequences]
        return initial_genotypes

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
        """
        fitness, mean, std = self.fitness_function(x.sequence)
        return fitness
