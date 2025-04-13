# todo: cite the original repo https://github.com/rail-berkeley/design-baselines

import torch
import numpy as np
import os

UTR_DATA_DIR = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\OpenELM_GenomicQD\design-bench_forked\design_bench_data\utr"  # todo: not absolute path
# Dictionary returning one-hot encoding of nucleotides.
NUC_D = {0: [1, 0, 0, 0], 1: [0, 1, 0, 0],
         2: [0, 0, 1, 0], 3: [0, 0, 0, 1]}

def sequence_nuc_to_one_hot(sequence):
    """
    Convert an integers sequence to a one-hot encoded tensor.
    @param sequence: Nucleotide sequence as integers. shape (batch_size, seq_len)
    @return: One-hot encoded tensor of shape (batch_size, seq_len, 4)
    """
    # Create an empty tensor to hold the one-hot encoded vectors
    one_hot = torch.zeros(sequence.shape[0], sequence.shape[1], 4, dtype=torch.float32)

    # Iterate through the sequence and fill the one-hot tensor
    for i in range(sequence.shape[0]):
        for j in range(sequence.shape[1]):
            key = int(sequence[i, j])
            one_hot[i, j] = torch.tensor(NUC_D[key])

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

def preprocess_and_save_npz(int_sequences, out_file):
    """
    Preprocess the integer sequences and save them as a compressed npz file.
    @param int_sequences: Integer sequences to be preprocessed. shape (N, seq_len)
    @param out_file: Output file path for the compressed npz file.
    """
    # shape: (N, seq_len, 4) after one-hot
    one_hot = sequence_nuc_to_one_hot(torch.tensor(int_sequences))
    # shape: (N, seq_len, 3) after log interpolation
    log_encoded = log_interpolated_one_hot(one_hot).numpy()
    # Save
    np.savez_compressed(out_file, X=log_encoded)
    print(f"Saved preprocessed data to {out_file}")


def load_preprocessed_npz(in_file):
    data = np.load(in_file)
    return data["X"]  # shape: (N, seq_len, 3)
