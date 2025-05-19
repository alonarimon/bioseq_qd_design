# todo: cite the original repo https://github.com/rail-berkeley/design-baselines

import torch
import numpy as np
import os

# Dictionary returning one-hot encoding of nucleotides.
NUC_D = {0: [1, 0, 0, 0], 1: [0, 1, 0, 0],
         2: [0, 0, 1, 0], 3: [0, 0, 0, 1]}

def sequence_nuc_to_one_hot(sequences: torch.Tensor):
    """
    Convert an integers sequences to a one-hot encoded tensor.
    @param sequences: Nucleotide sequences as integers. shape (batch_size, seq_len)
    @return: One-hot encoded tensor of shape (batch_size, seq_len, 4)
    """
    one_hot = torch.zeros(sequences.shape[0], sequences.shape[1], 4, dtype=torch.float32)
    one_hot = one_hot.scatter(2, sequences.unsqueeze(2), 1.0)
    return one_hot

def log_interpolated_one_hot(one_hot, C=0.6):
    """
    Convert one-hot encoded vectors to log probabilities using a relaxed softmax.
    @param one_hot: Tensor of shape (batch_size, sequence_length, K)
    @param C: Temperature parameter for the relaxed softmax.
    """
    K = one_hot.shape[-1]  # one_hot shape: (batch_size, sequence_length, K)
    relaxed = C * one_hot + (1.0 - C) / K
    log_relaxed = torch.log(relaxed)
    # remove the last coordinate to reduce linear dependence #todo: check if this is correct
    return log_relaxed[..., :-1]

def load_sampled_data(data_dir:str):
    """
    Load sampled data from the specified directory.
    @param data_dir: Directory containing the sampled data files.
    @return: Tuple of sampled X and Y data.
    """
    sampled_x = np.load(os.path.join(data_dir, "sampled_x.npy"))
    sampled_y = np.load(os.path.join(data_dir, "sampled_y.npy"))
    return sampled_x, sampled_y


