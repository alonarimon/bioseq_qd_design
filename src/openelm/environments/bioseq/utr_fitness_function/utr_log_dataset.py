import torch
from torch.utils.data import Dataset

from preprocess import sequence_nuc_to_one_hot, log_interpolated_one_hot


class UTRLogDataset(Dataset):
    def __init__(self, int_sequences, float_scores):
        # do everything once
        one_hot = sequence_nuc_to_one_hot(torch.tensor(int_sequences))  # (N, L, K)
        self.data_x = log_interpolated_one_hot(one_hot)                 # (N, L, K-1)
        self.data_y = torch.tensor(float_scores).float()
        # store in memory (or CPU tensor, or GPU if it fits)

    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]
