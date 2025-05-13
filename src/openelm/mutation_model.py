import functools
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional

import numpy as np


from openelm.codegen import model_setup, set_seed, truncate
from openelm.configs import BioRandomModelConfig, ModelConfig
from openelm.utils.diff_eval import apply_diff, split_diff


def get_model(config: ModelConfig): #todo: move from this file
    if config.model_type == "bio_random":
        return RandomSequenceMutator(config=config) #todo ?
    else:
        raise NotImplementedError


class MutationModel(ABC):
    """Base model class for all mutation models."""

    def __init__(self) -> None:
        self.config: ModelConfig

    @abstractmethod
    def mutate(self, seq: list) -> list:
        """Mutate a sequence."""
        raise NotImplementedError

class RandomSequenceMutator(MutationModel):
    """
    A simple random sequence mutator for bioseq generation. (without llms)
    """
    def __init__(self, config: BioRandomModelConfig):
        super().__init__()
        self.config = config
        self.alphabet = config.alphabet
        self.rng = np.random.default_rng(config.seed)
    
    def set_rng_state(self, rng_state: Optional[np.random.Generator]):
        self.rng = rng_state

    def mutate(self, seq: list) -> list:
        """
        Mutate a sequence by randomly changing one letter.
        :param seq: sequence to mutate
        """ 
        # TODO: the 50 steps limitation from the paper should be implemented here
        i = self.rng.integers(len(seq))
        new_letter = self.rng.choice([a for a in self.alphabet if a != seq[i]])
        mutated_seq = seq.copy()
        mutated_seq[i] = new_letter
        return mutated_seq
    