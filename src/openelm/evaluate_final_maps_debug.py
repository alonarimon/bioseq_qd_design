# todo: downsampling
#todo: novelty (by defining x_ref)
import json
import os
import pickle

import Levenshtein
import numpy as np

from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.disk_resource import DiskResource
from design_bench.oracles.tensorflow import ResNetOracle
from openelm.environments.bioseq.bioseq import RNAGenotype

ORACLE_NAME = "original_v0_minmax_orig"
DATASET_PATH = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\bioseq_qd_design\design-bench-detached\design_bench_data\utr"


def evaluate_final_solutions(list_of_solutions, oracle_model, ref_solutions=None):
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
    list_of_solutions_np = np.array(list_of_solutions)

    # Evaluate the solutions using the oracle model
    scores = oracle_model.predict(list_of_solutions_np).flatten() #todo: use gpu for evaluation, and add top_k

    # Calculate max, diversity, mean, and novelty scores
    max_score = float(max(scores))
    diversity_score = 1 / (N * (N - 1)) * sum(
        [sum([Levenshtein.distance(s1, s2) for s2 in list_of_solutions]) for s1 in list_of_solutions]
    )
    mean_score = float(sum(scores) / N)
    novelty_score = -1
    if ref_solutions is not None:
        novelty_score = sum([min([Levenshtein.distance(s, ref) for ref in ref_solutions]) for s in list_of_solutions]) / N

    return max_score, diversity_score, mean_score, novelty_score, scores

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

def loaf_ref_list(x_data_path, size_to_sample):
    # Load the reference set from the offline data directory
    offline_data_x = np.load(x_data_path)
    random_indexes = np.random.choice(offline_data_x.shape[0], size=size_to_sample, replace=False)
    reference_set = offline_data_x[random_indexes]

    return reference_set

if __name__ == '__main__':
    # load maps from pkl file
    exp_logs_dir = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\bioseq_qd_design\logs\elm\25-04-22_22-23\step_10000"
    maps_pkl_file = os.path.join(exp_logs_dir, "maps.pkl")
    with open(maps_pkl_file, "rb") as f:
        maps = pickle.load(f)
    genomes = maps["genomes"]
    non_zero_seq = [g.sequence for g in genomes if g != 0]
    print(f"Loaded {len(genomes)} genomes from the maps.")
    print(f"Number of non-zero genomes: {len(non_zero_seq)}")
    # load oracle model
    oracle = load_oracle(DATASET_PATH, ORACLE_NAME)

    offline_data_path_x = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\bioseq_qd_design\design-bench-detached\design_bench_data\utr\oracle_data\original_v0_minmax_orig\sampled_offline_relabeled_data\sampled_data_fraction_1_3_seed_42"
    ref_list = loaf_ref_list(os.path.join(offline_data_path_x, "x.npy"), 16384)

    # evaluate the genomes
    max_score, diversity_score, mean_score, novelty_score, scores = evaluate_final_solutions(non_zero_seq, oracle, ref_list)
    print(f"Max score: {max_score}, Diversity score: {diversity_score}, Mean score: {mean_score}, Novelty score: {novelty_score}")
    max_score_sequence = non_zero_seq[np.argmax(scores)]
    max_rna = RNAGenotype(max_score_sequence)
    print(f"Max score sequence (RNA): {max_rna}")

    # evaluate top 128 sequences (w.r.t. oracle)
    top_128_indexes = np.argsort(scores)[-128:].tolist()
    top_128_sequences = [non_zero_seq[i] for i in top_128_indexes]
    top_128_scores = [scores[i] for i in top_128_indexes]
    max_score_top, diversity_score_top, mean_score_top, novelty_score_top, scores_top = evaluate_final_solutions(top_128_sequences, oracle, ref_list)
    print(f"Top 128 Max score: {max_score_top}, Diversity score: {diversity_score_top}, Mean score: {mean_score_top}, Novelty score: {novelty_score_top}")

    # save results
    results = {
        "max_score": max_score,
        "diversity_score": diversity_score,
        "mean_score": mean_score,
        "novelty_score": novelty_score,
        "max_score_sequence": str(max_rna),
        "max_score_top": max_score_top,
        "diversity_score_top": diversity_score_top,
        "mean_score_top": mean_score_top,
        "novelty_score_top": novelty_score_top
    }
    results_file_path = os.path.join(exp_logs_dir, "oracle_evaluation_all.json")
    with open(results_file_path, "w") as f:
        json.dump(results, f)

    print(f"Results saved to {results_file_path}")

