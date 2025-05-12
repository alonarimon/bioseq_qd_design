import json
import sys
from dataclasses import asdict
from datetime import datetime
import numpy as np
import torch
import torch.nn.modules.loss as torch_loss
import os
import logging
import wandb

from helical.models.helix_mrna import HelixmRNAFineTuningModel, HelixmRNAConfig
from openelm.environments.bioseq.utr_fitness_function.helix_mrna.configs import HelixFineTuneConfig
from openelm.utils.plots import plot_learning_curves

# todo: make better solution for adding the project root to sys.path
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../../../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root: {project_root}")
print("PYTHONPATH (sys.path):", sys.path[:3])


# === Load Config ===
config = HelixFineTuneConfig()
device = config.device if torch.cuda.is_available() else "cpu"
helix_config = HelixmRNAConfig(batch_size=config.batch_size, device=device, max_length=config.max_length,
                               val_batch_size=config.val_batch_size)

# === Create experiment folder ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_dir = os.path.join(config.save_base_dir, f"exp_{timestamp}")
os.makedirs(exp_dir, exist_ok=True)

# === Set up logging ===
log_file_path = os.path.join(exp_dir, "fine_tuning.log")

logger = logging.getLogger()  # root logger
logger.setLevel(logging.INFO)
logger.handlers = [] # Clear existing handlers

file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"Logging initialized. Logs will be saved to {log_file_path}")

# === Initialize Weights and Biases ===
if config.wandb:
    wandb.init(
        project="bioseq_qd_design",
        group="fine_tune_helix-mRNA",
        name=f"e_{config.epochs}_b_{config.batch_size}_seed_{config.seed}_val_f_{config.val_fraction}_{timestamp}",
        config=asdict(config),
        dir=exp_dir,
        reinit=True,
    )
    wandb.config.update(asdict(config))
    logger.info(f"WandB initialized. Run ID: {wandb.run.id}")

# === Save config to experiment folder ===
with open(os.path.join(exp_dir, "config.json"), "w") as f:
    json.dump(asdict(config), f, indent=4)

# === Load data ===
x = np.load(os.path.join(config.data_dir, "x.npy"))
y = np.load(os.path.join(config.data_dir, "y.npy")).flatten()
# shuffle the data
indices = np.arange(len(x))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]
# split the data into train and validation sets
train_size = int(len(x) * (1 - config.val_fraction))
x_train = x[:train_size]
y_train = y[:train_size]
x_val = x[train_size:]
y_val = y[train_size:]
logger.info(f"X_train shape: {x_train.shape}, Y_train shape: {y_train.shape}")
logger.info(f"X_val shape: {x_val.shape}, Y_val shape: {y_val.shape}")

if config.debug:
    x_train = x_train[:100]
    y_train = y_train[:100]
    x_val = x_val[:100]
    y_val = y_val[:100]

alphabet = config.alphabet


def numbers_seq_to_alphabet(seq):
    """Convert a sequence of numbers to a string of letters."""
    return "".join([alphabet[i] for i in seq])


input_sequences_train = [numbers_seq_to_alphabet(seq) for seq in x_train]
labels_train = y_train.tolist()
input_sequences_val = [numbers_seq_to_alphabet(seq) for seq in x_val]
labels_val = y_val.tolist()

# === Model setup & training ===
model = HelixmRNAFineTuningModel(
    helix_mrna_config=helix_config,
    fine_tuning_head="regression",
    output_size=config.output_size
)
train_dataset = model.process_data(input_sequences_train)
if config.val_fraction > 0:
    val_dataset = model.process_data(input_sequences_val)
else:
    val_dataset = None

if config.loss == "mse":
    loss = torch_loss.MSELoss()
else:
    raise ValueError(f"Loss function {config.loss} not supported.")

train_losses, val_losses = model.train_fine_tune(train_dataset=train_dataset, train_labels=labels_train,
                                                 validation_dataset=val_dataset, validation_labels=labels_val,
                                                 epochs=config.epochs,
                                                 loss_function=loss, return_loss=True,
                                                 save_dir=exp_dir)

model.save_model(os.path.join(exp_dir, "model"))

# === Save training curve plot ===
learning_stats = {
    "train_loss": train_losses,
    "val_loss": val_losses,  # todo : when validation is implemented
}
plot_learning_curves(learning_stats, save_dir=exp_dir,
                     model_id=config.epochs,
                     title="Helix mRNA Fine-tuning", x_label="Epochs", y_label="MSE Loss")

logger.info("Training complete and model saved.")
logger.info(f"train losses: {train_losses}")
logger.info(f"val losses: {val_losses}")