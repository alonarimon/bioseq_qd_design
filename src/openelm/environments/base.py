import json
import math
import string
import sys
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Generic, Optional, TypeVar

import numpy as np
import requests
import logging

from openelm.configs import EnvConfig, StringEnvConfig
from openelm.environments.utils import NULL_SEED, get_image_target
from openelm.utils.code_eval import pool_exec_processes

if (
    (sys.version_info >= (3, 9, 14) and sys.version_info <= (3, 10))
    or (sys.version_info >= (3, 10, 7) and sys.version_info <= (3, 11))
    or sys.version_info >= (3, 11)
):
    # remove length limitation for int->str conversion
    # (model sometimes outputs really long ints)
    sys.set_int_max_str_digits(0)

Phenotype = Optional[np.ndarray]

logger = logging.getLogger(__name__)

def ackley(x: np.ndarray) -> np.ndarray:
    d = x.shape[-1]
    a = 5
    b = 0.1

    o1 = -a * np.exp(-b * np.sqrt(np.sum(x**2, axis=1) / d))
    o2 = -np.exp(np.sum(np.cos(math.tau * x) / d, axis=1))

    return -(a + math.exp(1) + o1 + o2)


def numpy_to_ascii_art(arr):
    """Convert a numpy array with dimensions (width, height, channels) to ascii art."""
    art_chars = " .:-=#"
    im = np.sum(arr, axis=-1)  # we can't do colors
    idx = np.round(np.interp(im, (im.min(), im.max()), (0, len(art_chars) - 1))).astype(
        "int"
    )
    chars = np.choose(idx, art_chars)
    ascii_art = "\n".join(["".join(x) for x in chars])
    return ascii_art


class Genotype(ABC):
    def __str__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def to_phenotype(self) -> Optional[Phenotype]:
        raise NotImplementedError


GenoType = TypeVar("GenoType", bound=Genotype)


class BaseEnvironment(ABC, Generic[GenoType]):
    def __init__(self) -> None:
        self.genotype_space: np.ndarray
        self.batch_size: int
        self.config: EnvConfig

    @abstractmethod
    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        raise NotImplementedError

    @abstractmethod
    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        raise NotImplementedError

    def initial_sequences(self) -> list[GenoType]:
        """Generate initial sequences."""
        return self.random()

    def to_phenotype(self, x: GenoType) -> Phenotype:
        """Convert genotype to phenotype."""
        return x.to_phenotype()

    @abstractmethod
    def random(self) -> list[GenoType]:
        raise NotImplementedError

    @abstractmethod
    def mutate(self, x: list[GenoType]) -> list[GenoType]:
        raise NotImplementedError

    @abstractmethod
    def fitness(self, x: GenoType) -> float:
        raise NotImplementedError

    def fitness_batch(self, x: list[GenoType]) -> list[float]:
        """
        Evaluate the fitness of a batch of genotypes.
        :param x: List of genotypes to evaluate.
        :return: List of fitness scores for the genotypes.
        """
        logger.warning("not implemented fitness_batch for this env, using fitness instead")
        return [self.fitness(genotype) for genotype in x]

    @abstractmethod
    def eval_with_oracle(self, genotypes: list[GenoType], downsampled_genotypes: list[GenoType] = None, k=128, save_dir: str | Path = None) -> dict:
        """
        Evaluate the fitness of a list of genotypes using an oracle.
        :param genotypes: List of genotypes to evaluate.
        :param downsampled_genotypes: List of downsampled genotypes to evaluate. (optional)
        :param k: The number of top genotypes to consider. (w.r.t. oracle fitness)
        :param save_dir: Directory to save the results.
        :return: A dictionary containing the scores of the genotypes set.
        """ #todo: move this to the map-elites level
        raise NotImplementedError("Oracle evaluation is not implemented for this environment.")

    @property
    def max_fitness(self) -> int:
        return 0

    @property
    # [starts, endings) of search intervals
    def behavior_space(self) -> np.ndarray:
        return self.genotype_space

    @property
    def behavior_ndim(self) -> int:
        return self.behavior_space.shape[1]




class ArrayGenotype(Genotype, np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __str__(self) -> str:
        return f'({", ".join(map(str, np.asarray(self)))})'

    def to_phenotype(self) -> Phenotype:
        return np.asarray(self)


# find all local maxima of a multimodal function
class FunctionOptim(BaseEnvironment[ArrayGenotype]):
    def __init__(self, ndim=2, seed=None):
        self.genotype_ndim = ndim
        self.genotype_space = np.repeat([[-4, 4]], self.genotype_ndim, axis=0).T
        self.batch_size: int = 1
        self.rng = np.random.default_rng(seed)

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        return self.rng

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        self.rng = rng_state

    def random(self) -> list[ArrayGenotype]:
        return [
            ArrayGenotype(self.rng.uniform(*self.genotype_space))
            for _ in range(self.batch_size)
        ]

    def mutate(self, x: list[ArrayGenotype]) -> list[ArrayGenotype]:
        for i in range(self.batch_size):
            ix = self.rng.integers(self.genotype_ndim)
            x[i][ix] = x[i][ix] + self.rng.uniform(-1, 1)
        return x

    def fitness(self, x: ArrayGenotype) -> float:
        return ackley(x[None])[0]


class StringArrayGenotype(ArrayGenotype):
    def __str__(self) -> str:
        x: np.ndarray = np.round(self)
        return "".join(
            string.ascii_letters[ix]
            for ix in np.clip(x.astype(int), 0, len(string.ascii_letters) - 1)
        )

    def to_phenotype(self) -> Phenotype:
        return np.asarray(self)


class MatchString(BaseEnvironment[StringArrayGenotype]):
    # find a string by mutating one character at a time

    def __init__(self, config: StringEnvConfig):
        self.alphabet = string.ascii_letters

        self.config: StringEnvConfig = config
        self.batch_size = self.config.batch_size
        self.target = np.array([self.alphabet.index(ch) for ch in self.config.target])
        self.genotype_ndim = self.target.shape[0]
        self.genotype_space = np.repeat(
            [[0, len(self.alphabet)]], self.genotype_ndim, axis=0
        ).T
        self.rng = np.random.default_rng(self.config.seed)

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        return self.rng

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        self.rng = rng_state

    def random(self) -> list[StringArrayGenotype]:
        return [
            StringArrayGenotype(self.rng.uniform(*self.genotype_space))
            for _ in range(self.batch_size)
        ]

    def mutate(self, genomes: list[StringArrayGenotype]) -> list[StringArrayGenotype]:
        x = deepcopy(genomes)
        for i in range(self.batch_size):
            ix = self.rng.integers(self.genotype_ndim)
            x[i][ix] = x[i][ix] + self.rng.uniform(-1, 1)
        return x

    def fitness(self, x: StringArrayGenotype) -> float:
        return -np.abs(x.to_phenotype() - self.target).sum()
