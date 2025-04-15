# todo: downsampling
#todo: novelty (by defining x_ref)
import os
import pickle

import Levenshtein
import numpy as np

from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.disk_resource import DiskResource
from design_bench.oracles.tensorflow import ResNetOracle

ORACLE_NAME = "original_v0_minmax_orig"
DATASET_PATH = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\OpenELM_GenomicQD\design-bench_forked\design_bench_data\utr"


def evaluate_final_solutions(list_of_solutions, oracle_model):
    """
    Evaluate the final solutions using the oracle model.
    :param list_of_solutions: List of solutions to evaluate. (genotypes)
    :param oracle_model: The oracle model used for evaluation.
    :return:  max, diversity, mean, and novelty scores for the solutions. And a list of scores.
    * the scores are defined as in appendix part E of -
    Jérémie DO, Flajolet A, Marginean A, Cully A, Pierrot T.
    Quality-diversity for one-shot biological sequence design.
    """
    N = len(list_of_solutions)
    print(f"Evaluating {N} solutions...")

    list_of_solutions_np = np.array(list_of_solutions)

    # Evaluate the solutions using the oracle model
    scores = oracle_model.predict(list_of_solutions_np)

    # Calculate max, diversity, mean, and novelty scores
    max_score = max(scores)
    diversity_score = 1 / (N * (N - 1)) * sum(
        [sum([Levenshtein.distance(s1, s2) for s2 in list_of_solutions]) for s1 in list_of_solutions]
    )
    mean_score = sum(scores) / N
    # todo : novelty_score = sum([min([oracle_model.distance(s, ref) for ref in refs_list]) for s in list_of_solutions]) / N

    return max_score, diversity_score, mean_score , scores #todo: novelty_score

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


if __name__ == '__main__':
    # load maps from pkl file
    exp_logs_dir = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\OpenELM_GenomicQD\logs\elm\25-04-15_11-59"
    maps_pkl_file = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\OpenELM_GenomicQD\logs\elm\25-04-15_11-59\step_999\maps.pkl"
    with open(maps_pkl_file, "rb") as f:
        maps = pickle.load(f)
    genomes = maps["genomes"]
    non_zero_seq = [g.sequence for g in genomes if g != 0]
    print(f"Loaded {len(genomes)} genomes from the maps.")
    print(f"Number of non-zero genomes: {len(non_zero_seq)}")
    # load oracle model
    oracle = load_oracle(DATASET_PATH, ORACLE_NAME)
    # evaluate the genomes
    max_score, diversity_score, mean_score, scores = evaluate_final_solutions(non_zero_seq, oracle)
    print(f"Max score: {max_score}, Diversity score: {diversity_score}, Mean score: {mean_score}")
    # save results
    results = {
        "max_score": max_score,
        "diversity_score": diversity_score,
        "mean_score": mean_score,
        "scores": scores
    }
    results_file_path = os.path.join(exp_logs_dir, "oracle_evaluation_all.pkl")
    with open(results_file_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Results saved to {results_file_path}")

