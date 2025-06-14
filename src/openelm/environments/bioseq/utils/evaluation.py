import json
import os
import pickle
from pathlib import Path
import logging

from omegaconf import OmegaConf
from rapidfuzz.distance import Levenshtein
import numpy as np
import yaml

import wandb
from scipy.spatial.distance import pdist, cosine
import tensorflow as tf
import RNA

from openelm.environments.bioseq.genotypes import RNAGenotype
from openelm.environments.bioseq.utils.debug_utils import cast_elm_config, loaf_ref_list, load_oracle, downsample_solutions
from openelm.utils.plots import plot_distance_histograms, plot_distance_histograms_only_k

logger = logging.getLogger(__name__)

def predict_secondary_structure(seq: str) -> str:
    structure, mfe = RNA.fold(seq)
    return structure  # dot-bracket notation


def extract_oracle_embeddings(list_of_sequences, oracle, layer_name='reshape'):
    """
    Extract embeddings from the oracle's hidden layer.
    """
    model = oracle.params["model"]
    embedding_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    x_input = oracle.dataset_to_oracle_x(np.array(list_of_sequences))
    embeddings = embedding_model.predict(x_input, verbose=0)
    return embeddings


def evaluate_solutions_set(oracle, solutions: list[RNAGenotype],
                           ref_solutions: list[RNAGenotype],
                           downsampled_solutions: list[RNAGenotype],
                           min_score: float,
                           max_score: float,
                           k: int = 128,
                           plot: bool = False, save_path: str = None,
                           use_oracle_embeddings: bool = True):
    """
    Evaluate the solutions using the oracle scores.
    :param oracle: The oracle to use for evaluation.
    :param solutions: List of solutions to evaluate.
    :param ref_solutions: List of reference solutions.
    :param downsampled_solutions: List of down-sampled solutions
    :param k: Number of top solutions to consider.
    :param plot: Whether to plot the results.
    :param save_path: Path to save the results.
    :return: Dictionary of all metrics.
    """
    logger.info(f"Evaluating {len(solutions)} genomes and {len(downsampled_solutions)} down-sampled genomes against the oracle and reference set.")


    sequences = [genotype.sequence for genotype in solutions]
    sequences_np = np.array(sequences)
    scores = oracle.predict(sequences_np).flatten()
    scores = (scores - min_score) / (max_score - min_score) # normalize the scores

    results_all = calc_all_metrics(
        scores, solutions, ref_solutions, oracle,  use_oracle_embeddings=use_oracle_embeddings
    )

    # Sort the scores and get the top k solutions
    sorted_indices = np.argsort(scores)[::-1]
    top_k_indices = sorted_indices[:k]
    top_k_solutions = [solutions[i] for i in top_k_indices]
    top_k_scores = [scores[i] for i in top_k_indices]

    # Calculate metrics for the top k solutions
    results_top_k = calc_all_metrics(
        top_k_scores, top_k_solutions, ref_solutions, oracle, use_oracle_embeddings=use_oracle_embeddings
    )

    # Calculate metrics for the downsampled solutions if provided
    downsampled_scores = oracle.predict(np.array([genotype.sequence for genotype in downsampled_solutions])).flatten()
    downsampled_scores = (downsampled_scores - min_score) / (max_score - min_score)  # normalize the scores
    results_downsampled = calc_all_metrics(
        downsampled_scores, downsampled_solutions, ref_solutions, oracle, use_oracle_embeddings=use_oracle_embeddings
    )

    results = {
        "all_solutions/": {
            "max_score": results_all["max_score"],
            "mean_score": results_all["mean_score"],
            "diversity_score_first_order": results_all["diversity_score_first_order"],
            "novelty_score_first_order": results_all["novelty_score_first_order"],
            "diversity_score_second_order": results_all["diversity_score_second_order"],
            "novelty_score_second_order": results_all["novelty_score_second_order"],
        },
        "top_k_solutions/": {
            "max_score": results_top_k["max_score"],
            "mean_score": results_top_k["mean_score"],
            "diversity_score_first_order": results_top_k["diversity_score_first_order"],
            "novelty_score_first_order": results_top_k["novelty_score_first_order"],
            "diversity_score_second_order": results_top_k["diversity_score_second_order"],
            "novelty_score_second_order": results_top_k["novelty_score_second_order"],
        },
        "downsampled_solutions/": {
            "max_score": results_downsampled["max_score"],
            "mean_score": results_downsampled["mean_score"],
            "diversity_score_first_order": results_downsampled["diversity_score_first_order"],
            "novelty_score_first_order": results_downsampled["novelty_score_first_order"],
            "diversity_score_second_order": results_downsampled["diversity_score_second_order"],
            "novelty_score_second_order": results_downsampled["novelty_score_second_order"],
        }
    }
    if use_oracle_embeddings:
        results["all_solutions/"]["diversity_score_embed"] = results_all["diversity_score_embed"]
        results["all_solutions/"]["novelty_score_embed"] = results_all["novelty_score_embed"]
        results["top_k_solutions/"]["diversity_score_embed"] = results_top_k["diversity_score_embed"]
        results["top_k_solutions/"]["novelty_score_embed"] = results_top_k["novelty_score_embed"]
        results["downsampled_solutions/"]["diversity_score_embed"] = results_downsampled["diversity_score_embed"]
        results["downsampled_solutions/"]["novelty_score_embed"] = results_downsampled["novelty_score_embed"]

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        results_file_path = os.path.join(save_path, "oracle_evaluation.json")
        with open(results_file_path, "w") as f:
            json.dump(results, f, indent=4)
        # save all the distance to a pickel file
        distances_file_path = os.path.join(save_path, "results.pkl")
        with open(distances_file_path, "wb") as f:
            pickle.dump({
                'results_all': results_all,
                'results_top_k': results_top_k,
                'results_downsampled': results_downsampled}, f)

    if plot:
        for key in ["internal_distances_first_order", "ref_distances_first_order",
                     "internal_distances_second_order", "ref_distances_second_order",
                     "internal_distances_embed", "novelty_distances_embed"]:
            if key in results_all:
                plot_distance_histograms(all_distances=results_all[key],
                                        topk_distances=results_top_k[key],
                                        downsampled_distances=results_downsampled[key],
                                        title=key.replace("_", " ").title(),
                                        save_path=os.path.join(save_path, f"{key}.png"))
                plot_distance_histograms_only_k(topk_distances=results_top_k[key],
                                                downsampled_distances=results_downsampled[key],
                                                title=key.replace("_", " ").title(),
                                                save_path=os.path.join(save_path, f"{key}_only_k.png"))

    return results


def calc_novelty_diversity_levenstein(solutions: list, ref_solutions: list):
    """
    Calculate the diversity and novelty of a set of solutions using Levenshtein distance.
    :param solutions: List of solutions to evaluate.
    :param ref_solutions: List of reference solutions.
    :param scores: List of scores for the solutions.
    :param secondary: Whether to use secondary-structure solutions.
    :return: max, diversity, mean, and novelty scores
    """
    N = len(solutions)
    internal_distances = [Levenshtein.distance(solutions[i], solutions[j]) for i in range(N) for j in range(i + 1, N)]
    diversity_score = np.mean(internal_distances)

    novelty_score = -1
    if ref_solutions is not None:
        ref_distances = [min([Levenshtein.distance(s, ref) for ref in ref_solutions]) for s in solutions]
        novelty_score = np.mean(ref_distances)

    return diversity_score, novelty_score, internal_distances, ref_distances


def calc_novelty_diversity_embeddings(oracle_embeddings: list, oracle_embeddings_ref: list):
    """
    Calculate the diversity and novelty of a set of solutions using oracle embeddings.
    :param oracle_embeddings: List of oracle embeddings to evaluate.
    :param oracle_embeddings_ref: List of reference oracle embeddings.
    :return: diversity and novelty scores (using cosine distance)
    """
    internal_distances = pdist(oracle_embeddings, metric='cosine')
    diversity_score_embed = np.mean(internal_distances)
    novelty_distances = []
    for emb in oracle_embeddings:
        distances = [cosine(emb, ref_emb) for ref_emb in oracle_embeddings_ref]
        novelty_distances.append(np.min(distances))
    novelty_score_embed = np.mean(novelty_distances)

    return diversity_score_embed, novelty_score_embed, internal_distances, novelty_distances


def calc_all_metrics(
        scores: list[float],
        solutions: list[RNAGenotype],
        ref_solutions: list[RNAGenotype],
        oracle,
        use_oracle_embeddings: bool = True,
):
    """
    Calculate all metrics for the given solutions and reference solutions.
    :param oracle: The oracle to use for evaluation.
    :param solutions: List of solutions to evaluate.
    :param ref_solutions: List of reference solutions.
    :return: Dictionary of all metrics.
    """
    # Calculate max, diversity, mean, and novelty scores
    N = len(solutions)
    max_score = float(max(scores))
    mean_score = float(sum(scores) / N)
    sequences = [genotype.sequence for genotype in solutions]
    refs_seq = [genotype.sequence for genotype in ref_solutions]

    # first order levenshtein diversity and novelty
    diversity_score_first_order, novelty_score_first_order, internal_distances_first_order, ref_distances_first_order \
        = calc_novelty_diversity_levenstein(sequences, refs_seq)
    # secondary structure diversity and novelty
    secondary_structure_solutions = [predict_secondary_structure(str(g)) for g in solutions]
    secondary_structure_ref_solutions = [predict_secondary_structure(str(g)) for g in ref_solutions]
    diversity_score_second_order, novelty_score_second_order, internal_distances_second_order, \
        ref_distances_second_order = calc_novelty_diversity_levenstein(secondary_structure_solutions,
                                                                       secondary_structure_ref_solutions)
    results = {
        "max_score": max_score,
        "mean_score": mean_score,
        "diversity_score_first_order": diversity_score_first_order,
        "novelty_score_first_order": novelty_score_first_order,
        "internal_distances_first_order": internal_distances_first_order,
        "ref_distances_first_order": ref_distances_first_order,
        "diversity_score_second_order": diversity_score_second_order,
        "novelty_score_second_order": novelty_score_second_order,
        "internal_distances_second_order": internal_distances_second_order,
        "ref_distances_second_order": ref_distances_second_order,
    }

    # oracle embedding diversity and novelty
    if use_oracle_embeddings:
        oracle_embeddings = extract_oracle_embeddings(sequences, oracle)
        oracle_embeddings_ref = extract_oracle_embeddings(refs_seq, oracle)
        diversity_score_embed, novelty_score_embed, internal_distances_embed, novelty_distances_embed = \
            calc_novelty_diversity_embeddings(oracle_embeddings, oracle_embeddings_ref)
        results["diversity_score_embed"] = diversity_score_embed
        results["novelty_score_embed"] = novelty_score_embed
        results["internal_distances_embed"] = internal_distances_embed
        results["ref_distances_embed"] = novelty_distances_embed


    return results

if __name__ == '__main__':

    # Example debug usage
    # load maps from pkl file

    ONLY_LAST_STEP = True  # Set to True to evaluate only the last step

    bioseq_base_dir = Path(__file__).resolve().parents[5]
 
    dirs= [
        'logs/elm/25-05-31_21-08',
        'logs/elm/25-05-31_22-17',
        'logs/elm/25-05-31_23-06',
        'logs/elm/25-05-31_23-14',
        'logs/elm/25-05-31_23-39',
        'logs/elm/25-06-01_01-13',
        'logs/elm/25-06-01_01-46',
        'logs/elm/25-06-01_01-52',
        'logs/elm/25-06-01_02-50',
        'logs/elm/25-06-01_04-06',
        'logs/elm/25-06-01_04-50',
        'logs/elm/25-06-01_05-03',
        'logs/elm/25-06-01_05-23',
        'logs/elm/25-06-01_07-05',
        'logs/elm/25-06-01_07-38',
        'logs/elm/25-06-01_08-37',
        'logs/elm/25-06-01_10-00',
        'logs/elm/25-06-01_10-38',
        'logs/elm/25-06-01_11-10',
        'logs/elm/25-06-01_12-52',
        'logs/elm/25-06-01_13-45',
        'logs/elm/25-06-01_15-05',
        'logs/elm/25-06-01_15-37',
        'logs/elm/25-06-01_17-19',
        'logs/elm/25-06-01_18-12',

    ]
    
    # Define a mapping for the wandb naming conventions, to support old and new naming conventions
    naming_for_wandb = {
        "mutator_helix_mrna": "helix",
        "fitness_helix_mrna": "helix",
        "helix": "helix",
        "random": "random",
        "bio_random": "random",
        "ensemble": "ensemble",
        "fitness_bio_ensemble": "ensemble",
        "fitness_utr_ensemble": "ensemble",
        "utr_ensemble": "ensemble",
        "nucleotides_frequencies": "freq",
        "freq": "freq",
    }

    for dir in dirs:
        exp_logs_dir = os.path.join(bioseq_base_dir, dir)
        config_file = os.path.join(exp_logs_dir, ".hydra", "config.yaml")
        config_hydra = OmegaConf.load(config_file)
        config_dict = OmegaConf.to_container(config_hydra, resolve=True)
        elm_original_config = cast_elm_config(config_dict)

      
        run_group = f"{elm_original_config.wandb_group}_{elm_original_config.env.task}_evaluation"
        run_name = f"BD {naming_for_wandb[elm_original_config.env.bd_type]} FITNESS {naming_for_wandb[elm_original_config.fitness_model.model_name]} MUTATOR {naming_for_wandb[elm_original_config.mutation_model.model_name]}"
        wandb.init(
            project="bioseq_qd_design",
            group=run_group,
            name=run_name,
            config=config_dict,
        )
        wandb.config.update(config_dict)
        
        # log all the png files in the logs directory
        for file in os.listdir(exp_logs_dir):
            if file.endswith(".png"):
                file_path = os.path.join(exp_logs_dir, file)
                wandb.log({file: wandb.Image(file_path)})
    
        all_steps_dirs = [d for d in os.listdir(exp_logs_dir) if os.path.isdir(os.path.join(exp_logs_dir, d))and d.startswith("step_")]
        all_steps_dirs.sort(key=lambda x: int(x.split("_")[1]))
        print(f"Found {len(all_steps_dirs)} steps directories: {all_steps_dirs}")
        
        # load fitness_history pkl file
        fitness_history_pkl_file = os.path.join(exp_logs_dir, all_steps_dirs[-1], "MAPElites_fitness_history.pkl")
        fitness_history = pickle.load(open(fitness_history_pkl_file, "rb"))
    
        # log the fitness history -  uncomment if needed
        # for i in range(len(fitness_history['max'])):
        #     wandb.log({
        #         "fitness max": fitness_history['max'][i],
        #         "fitness mean": fitness_history['mean'][i],
        #         "fitness min": fitness_history['min'][i],
        #         "qd score": fitness_history['qd_score'][i],
        #         "niches filled": fitness_history['niches_filled'][i],
        #         "step": i
        #     })

        # load oracle model
        ORACLE_NAME = "original_v0_minmax_orig"
        DATASET_PATH = bioseq_base_dir / "design-bench-detached" / "design_bench_data" / "utr"
        oracle = load_oracle(DATASET_PATH, ORACLE_NAME)
        model = oracle.params["model"]  # access the Keras model
        
        embedding_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer('reshape').output  # layer 21
        )
        offline_data_path = DATASET_PATH / "oracle_data" / ORACLE_NAME / "sampled_offline_relabeled_data" / "sampled_data_fraction_1_3_seed_42"
        ref_list = loaf_ref_list(os.path.join(offline_data_path, "x.npy"), 16384, seed=elm_original_config.env.seed)
        full_data_y_path = DATASET_PATH / "oracle_data" / ORACLE_NAME / "relabelled_y.npy"
        full_data_y = np.load(full_data_y_path)
        max_score = np.max(full_data_y)
        min_score = np.min(full_data_y)
        logger.info(f"offline data max score: {max_score}, min score: {min_score}")
        ref_genotypes = [RNAGenotype(seq) for seq in ref_list]
        
        if ONLY_LAST_STEP:
            all_steps_dirs = [all_steps_dirs[-1]]
        for step_dir in all_steps_dirs:
            print(f"Evaluating step: {step_dir}")
            # load the maps pkl file
            step_dir_path = os.path.join(exp_logs_dir, step_dir)
            maps_pkl_file = os.path.join(step_dir_path, "MAPElites_maps.pkl")
            with open(maps_pkl_file, "rb") as f:
                maps = pickle.load(f)
            genomes = maps["genomes"]
            non_zero_genoms = [g for g in genomes if g != 0]
            print(f"Loaded {len(genomes)} genomes from the maps.")
            print(f"Number of non-zero genomes: {len(non_zero_genoms)}")
            
            save_dir = os.path.join(step_dir_path, "oracle_nonrmalised_post_evaluation")
            
            k = np.min([128, len(non_zero_genoms)])  # downsample to k=128 or less if not enough genomes
            downsampled_genoms = downsample_solutions(genomes=non_zero_genoms, k=k, save_dir=save_dir, original_config=elm_original_config)
            logging.info(f"Evaluating {len(non_zero_genoms)} genomes and {len(downsampled_genoms)} down-sampled genomes against the oracle and reference set.")

            results = evaluate_solutions_set(oracle=oracle,
                                    solutions=non_zero_genoms,
                                    ref_solutions=ref_genotypes,
                                    downsampled_solutions=downsampled_genoms,
                                    min_score=min_score,
                                    max_score=max_score,
                                    k=k, plot=False, save_path=save_dir)
            step = int(step_dir.split("_")[1])
            wandb.log({"step": step, "results": results, "k": k})
        wandb.finish()