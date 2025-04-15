# todo: cite the original repo https://github.com/rail-berkeley/design-baselines

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from scoring_model import ScoringNetwork
from preprocess import log_interpolated_one_hot, sequence_nuc_to_one_hot

def optimize_conservatism(x_input: torch.Tensor,
                          model: nn.Module,
                          steps: int = 50,
                          lr: float = 0.05,
                          entropy_coeff: float = 0.9) -> torch.Tensor:
    """
    Performs gradient ascent on the model's output + entropy bonus,
    starting from x_input. Mimics the design-baselines approach:
      - Shuffle x to compute an "entropy" term = mean( (x - x_shuffled)^2 )
      - Score = model(x) + entropy_coeff * entropy
      - do x <- x + lr * grad(Score, x)
    """
    x_opt = x_input.clone().detach()  # shape (B, dim)
    x_opt.requires_grad_(True)

    for _ in range(steps):
        # We'll shuffle batch along dim=0 for the entropy part
        idx = torch.randperm(x_opt.size(0), device=x_opt.device)
        x_shuffled = x_opt[idx]

        # forward
        score = model(x_opt)
        # compute an L2 distance as "entropy"
        ent = (x_opt - x_shuffled).pow(2).mean()

        loss = score.mean() + entropy_coeff * ent

        # gradient
        grad = torch.autograd.grad(loss, x_opt)[0]
        with torch.no_grad():
            x_opt = x_opt + lr * grad
        x_opt.requires_grad_(True)

    return x_opt.detach()


class COMTrainer:
    """
    Trainer that can do standard MSE or a conservative objective approach (COM).
    """

    def __init__(self,
                 model: ScoringNetwork,
                 device: torch.device,
                 lr: float = 3e-4,
                 alpha_init: float = 1.0,
                 alpha_lr: float = 1e-2,
                 overestimation_limit: float = 0.5,
                 particle_steps: int = 50,
                 particle_lr: float = 0.05,
                 entropy_coeff: float = 0.9,
                 noise_std: float = 0.0,
                 use_conservative: bool = True
                 ):
        """
        Args:
            model: PyTorch scoring model
            lr: learning rate for scoring model
            alpha_init: initial alpha (for COM)
            alpha_lr: learning rate for alpha
            overestimation_limit: how strongly we penalize overestimation
            particle_steps: steps for gradient ascent "negative sample" search
            particle_lr: learning rate for that gradient ascent
            entropy_coeff: coefficient for the entropy bonus
            noise_std: optional Gaussian noise for data augmentation
            use_conservative: if False, we do a plain MSE
        """
        self.model = model.to(device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.device = device

        # alpha stuff
        self.log_alpha = torch.tensor(np.log(alpha_init), dtype=torch.float32,
                                      requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.use_conservative = use_conservative
        self.overestimation_limit = overestimation_limit
        self.particle_steps = particle_steps
        self.particle_lr = particle_lr
        self.entropy_coeff = entropy_coeff
        self.noise_std = noise_std

    def train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> dict:
        """One training step. x_batch shape (B, seq_len, K) or flattened
           y_batch shape (B,)
        """
        B = x_batch.size(0)
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        # Optionally, add noise
        if self.noise_std > 0:
            x_batch = x_batch + self.noise_std * torch.randn_like(x_batch)

        # forward pass
        pred_pos = self.model(x_batch)  # shape (B,)
        mse_loss = nn.functional.mse_loss(pred_pos, y_batch)

        # gather stats
        with torch.no_grad():
            # rank correlation optional
            # e.g. spearmanr => returns correlation, pvalue
            corr, _ = spearmanr(pred_pos.cpu().numpy(), y_batch.cpu().numpy())

        stats = {
            "train/mse": mse_loss.item(),
            "train/rank_corr": float(corr),
        }

        # If not using conservative objective, just do an MSE step
        if not self.use_conservative:
            self.opt.zero_grad()
            mse_loss.backward()
            self.opt.step()
            return stats

        # otherwise, do the COM approach
        # 1) find negative samples
        x_neg = optimize_conservatism(
            x_batch, self.model,
            steps=self.particle_steps,
            lr=self.particle_lr,
            entropy_coeff=self.entropy_coeff
        )
        with torch.no_grad():
            pred_neg = self.model(x_neg)
            overestimation = (pred_neg - pred_pos).detach()  # shape (B,)

        # 2) build total loss
        # model_loss = MSE + alpha * overestimation
        # alpha_loss = alpha * (overestimation_limit - overestimation)
        alpha = self.log_alpha.exp()
        model_loss = mse_loss + (alpha * overestimation).mean()
        alpha_loss = (alpha * self.overestimation_limit - alpha * overestimation).mean()

        stats["train/overestimation"] = overestimation.mean().item()
        stats["train/alpha"] = alpha.item()

        # 3) compute grads wrt model
        self.opt.zero_grad()
        model_loss.backward(retain_graph=True)  # keep for alpha step
        self.opt.step()

        # 4) compute grad wrt alpha
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        return stats

def train_scoring_function(
    x_npy_path: str,
    y_npy_path: str,
    seq_len: int,
    K: int,
    use_conservative: bool = True,
    device: str = "cuda"
):
    # 1) Load data
    X = np.load(x_npy_path)  # shape (N, seq_len) or (N, seq_len, K) if onehot
    Y = np.load(y_npy_path)  # shape (N,)
    # If Y is shape (N,1), flatten:
    if Y.ndim == 2:
        Y = Y[:, 0]

    # 2) Construct DataLoader
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 3) Build model and trainer
    model = ScoringNetwork(seq_len, K)
    trainer = COMTrainer(
        model=model,
        lr=3e-4,
        alpha_init=0.1,
        alpha_lr=1e-2,
        overestimation_limit=0.5,
        particle_steps=50,
        particle_lr=0.05,
        entropy_coeff=0.0,  # or 0.9 if you prefer
        noise_std=0.0,
        use_conservative=use_conservative,
        device=device
    )

    # 4) Train
    for epoch in range(50):
        epoch_stats = {"train/mse": [], "train/rank_corr": [],
                       "train/overestimation": [], "train/alpha": []}
        for x_batch, y_batch in loader:
            stats = trainer.train_step(x_batch, y_batch)
            for k, v in stats.items():
                if k in epoch_stats:
                    epoch_stats[k].append(v)

        # log epoch summary
        print(f"Epoch {epoch+1}: ", end="")
        for k, arr in epoch_stats.items():
            if arr:  # non-empty
                print(f"{k}={np.mean(arr):.4f}  ", end="")
        print()

    # 5) Return final model
    return model

if __name__ == '__main__':
    # Example usage
    x_npy_path = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\OpenELM_GenomicQD\design-bench_forked\design_bench_data\utr\sampled_data_fraction_1_3_seed_42\sampled_x.npy"
    y_npy_path = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\OpenELM_GenomicQD\design-bench_forked\design_bench_data\utr\sampled_data_fraction_1_3_seed_42\sampled_y.npy"
    seq_len = 50
    K = 4
    model = train_scoring_function(x_npy_path, y_npy_path, seq_len, K)