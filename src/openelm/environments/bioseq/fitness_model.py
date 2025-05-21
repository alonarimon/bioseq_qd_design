from abc import ABC, abstractmethod
from typing import Generic

import torch

from openelm.configs import ModelConfig, FitnessHelixMRNAConfig, FitnessBioEnsembleConfig
from openelm.environments.base import GenoType


class FitnessModel(ABC, torch.nn.Module, Generic[GenoType]):
    """
    Base class for fitness models.
    """
    def __init__(self, config):
        """
        Initializes the fitness model with the given configuration.
        :param config: Configuration object containing the model parameters.
        """
        super().__init__()
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() and config.cuda else "cpu"

    @abstractmethod
    def __call__(self, genotypes: list[GenoType]) -> list[float]:
        """
        Abstract method to process a batch of sequences and return scores.
        :param genotypes: Input genotypes to be scored.
        :return: Scores for the input sequences.
        """
        pass




def get_fitness_model(config: ModelConfig):
    """
    Factory function to create a fitness model based on the provided configuration.
    :param config: Configuration object containing the model parameters.
    :return: An instance of the appropriate fitness model class.
    """
    if config.model_type == "helix_mrna":
        from openelm.environments.bioseq.utr_fitness_function.helix_mrna.helix_mrna_fitness_function import HelixMRNAFitnessFunction
        if not isinstance(config, FitnessHelixMRNAConfig):
            raise ValueError("Expected FitnessHelixMRNAConfig for helix_mrna model type.")
        return HelixMRNAFitnessFunction(config)
    elif config.model_type == "bio_ensemble":
        from openelm.environments.bioseq.scoring_ensemble.fitness_scoring_ensemble import FitnessScoringEnsemble
        if not isinstance(config, FitnessBioEnsembleConfig):
            raise ValueError("Expected FitnessBioEnsembleConfig for bio_ensemble model type.")
        return FitnessScoringEnsemble(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")