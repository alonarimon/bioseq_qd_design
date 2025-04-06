import json
from typing import Optional, Callable

import numpy as np
from langchain.schema import HumanMessage

from openelm.configs import QDEnvConfig, QDBioRNAEnvConfig
from openelm.environments.base import BaseEnvironment
from openelm.environments.prompt.prompt import PromptGenotype
from openelm.mutation_model import MutationModel, get_model


class RNAGenotype():
    """
    A simple genotype class for RNA bioseq generation. (without llms)
    """

    def __init__(self, sequence: str):
        self.sequence = sequence

    def __str__(self):
        return self.sequence

    def _nucleotides_frequencies(self) -> np.ndarray:
        """
        Calculate the frequencies of nucleotides in a sequence.
        :param x: RNAGenotype
        :return: numpy array with frequencies of A, C, G
        """
        freq = np.zeros(3, dtype=float)
        for letter in self.sequence:
            if letter == "A":
                freq[0] += 1
            elif letter == "C":
                freq[1] += 1
            elif letter == "G":
                freq[2] += 1
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


class RNASolution():
    """
    A simple solution class for RNA bioseq generation. (without llms)
    """
    def __init__(self, genotype: RNAGenotype):
        self.genotype = genotype
        self.fitness_score = None
        self.behavioral_descriptor = None

    def __str__(self):
        return self.genotype.__str__()


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
        self.config = config
        self.mutation_model = get_model(mutation_model.config)
        self.batch_size = config.batch_size
        self.genotype_space = np.array(
            self.config.behavior_space).T  # todo: i think it should be renamed to behavior_space (in the base class)
        self.sequence_length = config.sequence_length
        self.alphabet = config.alphabet
        self.rng = np.random.default_rng(config.seed)

        self.scoring_functions = [trivial_scoring_function]
        # self.reference_set = reference_set #todo: for similarity-based bd
        # self.projection_matrix = #todo: for similarity-based bd
        self.beta = config.beta # penalty term factor (in the fitness function)
        self.bd_type = config.bd_type # behavioral descriptor type (e.g. 'nucleotides_frequencies')
        # todo: here they originally had 'del mutation_model'. see if it is needed (and if it does - delete it outside the constructor)

    def get_rng_state(self) -> Optional[np.random.Generator]:
        return self.rng

    def set_rng_state(self, rng_state: Optional[np.random.Generator]):
        self.rng = rng_state

    def _random_seq(self) -> str:
        return ''.join(self.rng.choice(self.alphabet, self.sequence_length))

    def _mutate_seq(self, seq: str) -> str:
        """
        Mutate a sequence by randomly changing one character.
        """
        i = self.rng.integers(self.sequence_length)
        new_char = self.rng.choice([a for a in self.alphabet if a != seq[i]])
        return seq[:i] + new_char + seq[i + 1:]

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
        scores = [f(x) for f in self.scoring_functions]
        mean = np.mean(scores)
        std = np.std(scores)
        return mean - self.beta * std

    def _nucleotides_frequencies(self, x: RNAGenotype) -> np.ndarray:
        """
        Calculate the frequencies of nucleotides in a sequence.
        :param x: RNAGenotype
        :return: numpy array with frequencies of A, C, G
        """ #todo: delete from here?
        freq = np.zeros(3, dtype=float)
        for letter in x.sequence:
            if letter == "A":
                freq[0] += 1
            elif letter == "C":
                freq[1] += 1
            elif letter == "G":
                freq[2] += 1
        freq /= len(x.sequence)
        return freq

    def to_phenotype(self, genotype: RNAGenotype) -> Optional[np.ndarray]:
        """ #todo: delete from here?
        Convert sequence to a phenotype representation (behavioral descriptor) according to the specified type.
        :param genotype: RNAGenotype
        return: np.ndarray with dtype float
        """
        if self.bd_type == "nucleotides_frequencies":
            return self._nucleotides_frequencies(genotype)
        else:
            raise ValueError(f"Unknown bd_type: {self.bd_type}. Supported: nucleotides_frequencies")
