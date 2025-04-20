import json
from dataclasses import asdict
from datetime import datetime
import numpy as np
import torch
import torch.nn.modules.loss as torch_loss
import os

from helical.models.helix_mrna import HelixmRNAFineTuningModel, HelixmRNAConfig
from openelm.environments.bioseq.utr_fitness_function.helix_mrna.configs import HelixFineTuneConfig

# === Load Config ===
config = HelixFineTuneConfig()
device = config.device if torch.cuda.is_available() else "cpu"
helix_config = HelixmRNAConfig(batch_size=config.batch_size, device=device, max_length=config.max_length)

# === Create experiment folder ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exp_dir = os.path.join(config.save_base_dir, f"exp_{timestamp}")
os.makedirs(exp_dir, exist_ok=True)

# === Save config to experiment folder ===
with open(os.path.join(exp_dir, "config.json"), "w") as f:
    json.dump(asdict(config), f, indent=4)

# === Load data ===
x = np.load(os.path.join(config.data_dir, "x.npy"))
y = np.load(os.path.join(config.data_dir, "y.npy")).flatten()
alphabet = config.alphabet

def numbers_seq_to_alphabet(seq):
    """Convert a sequence of numbers to a string of letters."""
    return "".join([alphabet[i] for i in seq])

input_sequences = [numbers_seq_to_alphabet(seq) for seq in x]
labels = y.tolist()

# === Model setup & training ===
model = HelixmRNAFineTuningModel(
    helix_mrna_config=helix_config,
    fine_tuning_head="regression",
    output_size=config.output_size
)
train_dataset = model.process_data(input_sequences)
if config.loss == "mse":
    loss = torch_loss.MSELoss()
else:
    raise ValueError(f"Loss function {config.loss} not supported.")

model.train(train_dataset=train_dataset, train_labels=labels, epochs=config.epochs, loss_function=loss)
model.save_model(os.path.join(exp_dir, "model"))

train_outputs = model.get_outputs(train_dataset)
np.save(os.path.join(exp_dir, "train_outputs.npy"), train_outputs)
print("Output shape:", train_outputs.shape)
