import os

import numpy as np
import torch

from openelm.environments.bioseq.utr_fitness_function.scoring_model import ScoringNetwork
from openelm.environments.bioseq.utr_fitness_function.preprocess import sequence_nuc_to_one_hot, log_interpolated_one_hot


def load_scoring_ensemble(seq_len, K, model_dir, device="cuda", ensemble_size=1):
    """
    Loads all .pt (or .pth) models from `model_dir` into a list.
    """
    ensemble = []
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

class FitnessScoringEnsemble:
    """
    A wrapper class for a scoring ensemble that allows for batch processing of sequences.
    """
    def __init__(self, seq_len, K, model_dir, device="cuda", ensemble_size=1, beta=0.0):
        """
        Initializes the scoring ensemble.
        :param seq_len: Length of the sequences.
        :param K: alphabet size
        :param model_dir: Directory containing the scoring models.
        :param device: Device to load the models on (e.g., "cpu" or "cuda").
        :param ensemble_size: Number of models to load from the directory.
        :param beta: Penalty term factor
        """
        self.scoring_ensemble = load_scoring_ensemble(seq_len, K, model_dir, device, ensemble_size)
        self.device = device
        self.beta = beta

    def __call__(self, sequence):
        """
        Process a batch of sequences and return their scores.
        """
        # preprocess sequence
        torch_seq = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        one_hot = sequence_nuc_to_one_hot(torch_seq)
        log_x = log_interpolated_one_hot(one_hot).to(self.device)

        # Get scores from each model in the ensemble
        scores = [model(log_x).detach().cpu().numpy() for model in self.scoring_ensemble]

        # mean and std of scores
        scores = np.array(scores)
        mean = np.mean(scores, axis=0)
        std = np.std(scores, axis=0)

        return mean - self.beta * std, mean, std