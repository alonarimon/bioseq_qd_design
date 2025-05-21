from datetime import datetime
import os
import numpy as np
import torch
import logging

from openelm.configs import FitnessBioEnsembleConfig
from openelm.environments.bioseq.bioseq import RNAGenotype
from openelm.environments.bioseq.fitness_model import FitnessModel
from openelm.environments.bioseq.scoring_ensemble.scoring_model import ScoringNetwork
from openelm.environments.bioseq.scoring_ensemble.preprocess import sequence_nuc_to_one_hot, log_interpolated_one_hot
from openelm.environments.bioseq.scoring_ensemble.train_scoring_models import train_scoring_models

logger = logging.getLogger(__name__)

def load_scoring_ensemble(seq_len, K, model_dir, device="cuda", ensemble_size=1):
    """
    Loads all .pt (or .pth) models from `model_dir` into a list.
    """
    ensemble = torch.nn.ModuleList()
    model_fnames = [f for f in os.listdir(model_dir) if f.endswith(".pt") or f.endswith(".pth")]
    model_fnames = sorted(model_fnames)[:ensemble_size]  # limit to ensemble_size models

    for fname in model_fnames:
        path = os.path.join(model_dir, fname)
        # 1) create a fresh instance
        model = ScoringNetwork(seq_len, K)
        # 2) load state dict
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        ensemble.append(model)
        print(f"Loaded model from {path}.")

    return ensemble


class FitnessScoringEnsemble(FitnessModel[RNAGenotype]):
    """
    A wrapper class for a scoring ensemble that allows for batch processing of sequences.
    """
    def __init__(self, config: FitnessBioEnsembleConfig):
        """
        Initializes the scoring ensemble.
        :param config: Configuration object containing the model parameters.
        """
        super().__init__(config)
        self.models_dir = config.model_path
        self.scoring_ensemble = None
        if config.load_existing_models:
            self.scoring_ensemble = load_scoring_ensemble(config.gen_max_len, config.alphabet_size,
                self.models_dir, self.device, config.ensemble_size)
        self.beta = config.beta
    
    def retrain(self, data_x: np.ndarray, data_y: np.ndarray):
        """
        Create a new scoring ensemble modules.
        """
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.models_dir = os.path.join(self.config.model_path, time)
        train_scoring_models(
            data_x, data_y,
            disk_target_data=self.models_dir,
            validation_fraction=self.config.validation_fraction,
            save_dir=self.models_dir,
            seq_len=self.config.gen_max_len,
            alphabet_size=self.config.alphabet_size,
            ensemble_size=self.config.ensemble_size,
            use_conservative=self.config.use_conservative,
            device=self.device,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size
        )
        self.scoring_ensemble = load_scoring_ensemble(
            self.config.gen_max_len, self.config.alphabet_size,
            self.models_dir, self.device, self.config.ensemble_size
        )    

        

    def __call__(self, genotypes: list[RNAGenotype]) -> list[float]:
        """
        Process a batch of sequences and return scores.
        :param genotypes: Input genotypes to be scored.
        :return: Scores for the input sequences.
        """
        if self.scoring_ensemble is None:
            raise ValueError("Scoring ensemble not initialized. Call retrain() first.")
    
        if len(genotypes) > self.config.batch_size:
            raise ValueError(f"Batch size {len(genotypes)} exceeds the configured batch size {self.config.batch_size}.")

        # preprocess sequence
        sequences = [genotype.sequence for genotype in genotypes] # todo: maybe not efficient?
        torch_seq = torch.tensor(sequences, dtype=torch.int64)
        one_hot = sequence_nuc_to_one_hot(torch_seq)
        log_x = log_interpolated_one_hot(one_hot).to(self.device)

        with torch.no_grad():
            # Get scores from each model in the ensemble
            scores = np.array([model(log_x).detach().cpu().numpy() for model in self.scoring_ensemble])

        # mean and std of scores
        mean = np.mean(scores, axis=0)
        std = np.std(scores, axis=0)
        
        return list(mean - self.beta * std)