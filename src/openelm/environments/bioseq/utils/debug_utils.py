import os
import numpy as np


from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.disk_resource import DiskResource
from design_bench.oracles.tensorflow import ResNetOracle
from scipy.spatial.distance import squareform

ORACLE_NAME = "original_v0_minmax_orig"
DATASET_PATH = r"/design-bench-detached/design_bench_data/utr"

def load_oracle(dataset_path, oracle_name):
    oracle_data_path = os.path.join(dataset_path, "oracle_data")
    oracle_data_path = os.path.join(oracle_data_path, oracle_name)
    # Load validation split
    val_x = [DiskResource(os.path.join(oracle_data_path, 'oracle_train_split', "split-val-x-0.npy"))]
    val_y = [DiskResource(os.path.join(oracle_data_path, 'oracle_train_split', "split-val-y-0.npy"))]
    val_dataset = DiscreteDataset(val_x, val_y, num_classes=4)
    oracle_model_path = os.path.join(oracle_data_path, "oracle")

    # Load the saved oracle (fit=False ensures it loads from disk)
    oracle = ResNetOracle(
        val_dataset,
        noise_std=0.0,
        fit=False,  # do not retrain
        is_absolute=True,
        disk_target=oracle_model_path
    )

    print("Oracle params:\n",
          "rank_correlation:", oracle.params["rank_correlation"],
          "\nmodel_kwargs:", oracle.params["model_kwargs"],
          "\nsplit_kwargs:", oracle.params["split_kwargs"])

    return oracle

def loaf_ref_list(x_data_path, size_to_sample, seed=42):
    # Load the reference set from the offline data directory
    np.random.seed(seed)
    offline_data_x = np.load(x_data_path)
    random_indexes = np.random.choice(offline_data_x.shape[0], size=size_to_sample, replace=False)
    reference_set = offline_data_x[random_indexes]

    return reference_set

def get_conflicted_pairs(list_of_solutions, scondary_structures, distances, distances_ss, scores, save_dir):

    # show pairs who had high distance in the secondary structure but low in the primary structure
    # find the indexes of the pairs who had high distance in the secondary structure but low in the primary structure
    distance_matrix = squareform(distances)
    distance_matrix_ss = squareform(distances_ss)
    conflicted_pairs = []
    for i in range(len(list_of_solutions)):
        for j in range(i + 1, len(list_of_solutions)):
            if abs(distance_matrix[i][j] - distance_matrix_ss[i][j]) > 20:
                conflicted_pairs.append((i, j))
    # save the pairs to a file
    with open(os.path.join(save_dir, "conflicted_pairs_distances.txt"), "w") as f:
        for pair in conflicted_pairs:
            f.write(f"indexes {pair}: {distance_matrix[pair[0]][pair[1]]} [primarily] vs {distance_matrix_ss[pair[0]][pair[1]]} [secondary] \n")
            f.write(f"seq1: {str(list_of_solutions[pair[0]])}\n")
            f.write(f"seq2: {str(list_of_solutions[pair[1]])}\n")
            f.write(f"secondary structure 1: {scondary_structures[pair[0]]}\n")
            f.write(f"secondary structure 2: {scondary_structures[pair[1]]}\n")
            f.write(f"score1: {scores[pair[0]]}, score2: {scores[pair[1]]}\n")
            f.write("=" * 50 + "\n")




