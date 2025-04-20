# todo: cite the original repo https://github.com/rail-berkeley/design-baselines
import torch
from torch import nn


class ScoringNetwork(nn.Module):
    def __init__(self, seq_len: int, K: int):
        """
        seq_len: length of the sequence
        K: size of the alphabet (4 for RNA, 20 for proteins)
        """
        super().__init__()
        input_dim = seq_len * (K - 1)  # after discarding last coord
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1)
        )

    def forward(self, log_x: torch.Tensor) -> torch.Tensor:
        """
        log_x shape: (B, seq_len, K-1) or (B, input_dim)
        """
        if log_x.dim() == 3:
            # flatten
            B, seq_len, new_K = log_x.shape
            log_x = log_x.reshape(B, seq_len * new_K)
        return self.net(log_x).squeeze(-1)  # shape: (B,)
