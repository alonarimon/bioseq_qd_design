import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.special import softmax
from tqdm import tqdm
import Levenshtein
import tensorflow as tf

import design_bench
from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.disk_resource import DiskResource
from design_bench.oracles.tensorflow import ResNetOracle


from openelm.configs import ModelConfig, QDBioEnvConfig, QDBioTaskBasedEnvConfig, QDBioUTREnvConfig
from openelm.environments.base import BaseEnvironment, Phenotype
from openelm.environments.bioseq.genotypes import BioSeqGenotype, RNAGenotype, DNAGenotype
from openelm.mutation_model import get_mutation_model
from openelm.environments.bioseq.fitness_model import get_fitness_model
from openelm.environments.bioseq.utils.evaluation import evaluate_solutions_set

logger = logging.getLogger(__name__)


class BioSeqEvolution(BaseEnvironment[BioSeqGenotype]):
    def __init__(
            self,
            config: QDBioEnvConfig,
            mutation_model_config: ModelConfig, #todo: not in use, GET CONFIG?
            fitness_model_config: ModelConfig,
    ):
        """
        Args:
            config (QDBioEnvConfig): Configuration for the environment.
            mutation_model (MutationModel): Mutation model for mutating sequences.
        """
        super().__init__() #todo: check if this is needed
        logger.info(f"Initializing BioSeqEvolution environment with config: {config}")
        self.config = config
        
        
        # models
        self.mutation_model = get_mutation_model(mutation_model_config) 
        self.fitness_function = get_fitness_model(fitness_model_config)
        

        self.batch_size = config.batch_size
        self.genotype_space = np.array(
            self.config.behavior_space).T  # todo: i think it should be renamed to behavior_space (in the base class)
        self.sequence_length = config.sequence_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # set all random seeds for reproducibility
        self.rng = np.random.default_rng(config.seed)
        self.mutation_model.set_rng_state(self.rng)
        tf.random.set_seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # initialise task specific variables
        if self.config.task == 'TFBind10-Exact-v1': #TODO: also the utr resnet from the task framework
            assert isinstance(self.config, QDBioTaskBasedEnvConfig), "TFBind10-Exact-v1 task requires QDBioTaskBasedEnvConfig"
            self.task = design_bench.make(self.config.task, relabel=False)
            self.offline_data_x = self.task.dataset.x
            self.offline_data_y = self.task.dataset.y
            self.min_output = self.task.dataset_min_output
            self.max_output = self.task.dataset_max_output
            if self.min_output == -np.inf:
                logger.warning("min_output is -inf, setting it according to the offline data")
                self.min_output = np.min(self.offline_data_y)
            if self.max_output == np.inf:
                logger.warning("max_output is +inf, setting it according to the offline data")
                self.max_output = np.max(self.offline_data_y)
            
            logger.info(f"offline dataset size: {self.task.dataset_size}")
            self.offline_data_x_gen = np.array([DNAGenotype(seq) for seq in self.offline_data_x])

        elif self.config.task == 'UTR-ResNet-v0-CUSTOM':
            assert isinstance(self.config, QDBioUTREnvConfig), "UTR-ResNet-v0-CUSTOM task requires QDBioUTREnvConfig"
            self.task = None
            # Load the reference set from the offline data directory
            self.offline_data_x = np.load(os.path.join(config.offline_data_dir, self.config.offline_data_x_file))
            self.offline_data_y = np.load(os.path.join(config.offline_data_dir, self.config.offline_data_y_file))
            self.offline_data_x_gen = np.array([RNAGenotype(seq) for seq in self.offline_data_x])
            self.min_output = config.oracle_min_score
            self.max_output = config.oracle_max_score
        else:
            raise ValueError(f"Unknown task: {self.config.task}. Supported: TFBind10-Exact-v1, UTR-ResNet-v0-CUSTOM")
        
        if self.config.retrain_fitness_model:
            logger.info("Retraining fitness model...")
            self.fitness_function.retrain(self.offline_data_x, self.offline_data_y)
        
        self.reference_set = self._load_ref_set()
        self.oracle = self._load_oracle()  # Load the oracle model from disk, for final evaluation on the solutions (not used in the optimization process)


        if self.config.bd_type == "similarity_based":
            self.projection_matrix = self.rng.uniform(low=0.0, high=1.0, size=(config.size_of_refs_collection, len(config.behavior_space))).astype(np.float32)  # Initialize W from N(0,1)
            logger.info(f"Projection matrix W: {self.projection_matrix}")
            logger.info(f"Projection matrix W shape: {self.projection_matrix.shape}")
            if config.distance_normalization_constant < 0:
                self.R_normalization_constant = self._get_r_distance_norm_const(subsample=True)  # todo: need to be done once without subsampling
            else:
                self.R_normalization_constant = config.distance_normalization_constant
            logger.info(f"R normalization constant: {self.R_normalization_constant}")
        

        self.bd_min = 0
        self.bd_max = 1
        if self.config.normalize_bd:
            if len(self.config.bd_min) > 0 and len(self.config.bd_max) > 0:
                # If bd_min and bd_max are provided in the config, use them
                self.bd_min = np.array(self.config.bd_min)
                self.bd_max = np.array(self.config.bd_max)
                logger.info(f"Using pre-defined behavioral descriptor min: {self.bd_min}, max: {self.bd_max}")
            else:
                subsample = self.config.bd_type == "similarity_based" # we subsample the offline data only for the similarity based bd, because it is expensive
                self.bd_min, self.bd_max = self.get_training_bd_stats(subsample=subsample)
                logger.info(f"Behavioral descriptor min: {self.bd_min}, max: {self.bd_max}")


        if self.batch_size > self.fitness_function.config.batch_size:
            logger.warning(f"Environment batch size {self.batch_size} exceeds the fitness model batch size {self.fitness_function.config.batch_size}.")
    
    
    
    def _get_r_distance_norm_const(self, subsample = False) -> float:
        """
        Calculate the normalization constant for the distance metric.
        :param subsample: if True, subsample the offline data to the size of the reference collection x 2.
        :return: Normalization constant.
        """
        dists = []
        offline_data = self.offline_data_x_gen
        if subsample:
            sampled_indexes = self.rng.choice(self.offline_data_x.shape[0], size=(self.config.size_of_refs_collection * 2), replace=False)
            offline_data = self.offline_data_x[sampled_indexes]
        for i in tqdm(range(len(offline_data)), desc="Calculating R normalization constant"):
            for j in range(i + 1, len(offline_data)):
                    dist = Levenshtein.distance(offline_data[i], offline_data[j])
                    dists.append(dist)

        R_normalization_constant = np.mean(dists) / 2
        return R_normalization_constant

    def _load_oracle(self):
        """
        Load the oracle model from disk.
        :return: Loaded oracle model.
        """
        if self.config.task == 'TFBind10-Exact-v1':
            oracle = self.task.oracle
        elif self.config.task == 'UTR-ResNet-v0-CUSTOM':
            # Load validation split
            val_x = [DiskResource(os.path.join(self.config.oracle_model_path, 'oracle_train_split', "split-val-x-0.npy"))]
            val_y = [DiskResource(os.path.join(self.config.oracle_model_path, 'oracle_train_split', "split-val-y-0.npy"))]
            val_dataset = DiscreteDataset(val_x, val_y, num_classes=4)
            oracle_model_path = os.path.join(self.config.oracle_model_path, "oracle")

            # Load the saved oracle (fit=False ensures it loads from disk)
            oracle = ResNetOracle(
                val_dataset,
                noise_std=0.0,
                fit=False,  # do not retrain
                is_absolute=True,
                disk_target=oracle_model_path)
            
            logger.info(
                f"Oracle params:\n"
                f"rank_correlation: {oracle.params['rank_correlation']}\n"
                f"model_kwargs: {oracle.params['model_kwargs']}\n"
                f"split_kwargs: {oracle.params['split_kwargs']}"
        )
        else:
            raise ValueError(f"Unknown task: {self.config.task}. Supported: TFBind10-Exact-v1, UTR-ResNet-v0-CUSTOM")

        return oracle

    def get_training_bd_stats(self, subsample = False) -> tuple:
        """
        Get the training behavioral descriptor statistics.
        :param subsample: if True, subsample the offline data to the size of the reference collection.
        :return: tuple of min and max values for the behavioral descriptor space.
        """
        offline_data_x_gen = self.offline_data_x_gen
        if subsample:
            # Subsample the offline data
            logger.info(f"Subsampling the offline data to {self.config.size_of_refs_collection} genotypes, to calculate the behavioral descriptor statistics.")
            random_indexes = self.rng.choice(self.offline_data_x.shape[0], size=self.config.size_of_refs_collection, replace=False)
            offline_data_x_gen = self.offline_data_x_gen[random_indexes]

        # Calculate the behavioral descriptor for each genotype
        bd_values = np.array([self.to_phenotype(genotype) for genotype in offline_data_x_gen])
        # Calculate the min and max values for each behavioral descriptor, for all dimensions
        min_bd = np.min(bd_values, axis=0)
        max_bd = np.max(bd_values, axis=0)

        return min_bd, max_bd

    def get_rng_state(self) -> Optional[np.random.Generator]:
        return self.rng

    def set_rng_state(self, rng_state: Optional[np.random.Generator]):
        self.rng = rng_state

    def initial_sequences(self) -> list[BioSeqGenotype]:
        """
        Generate a batch of initial sequences by randomly sample from the offline data.
        @return: list of RNAGenotype
        """
        rng = np.random.default_rng(self.config.seed)  # Ensures reproducibility, not using the self.rng
        random_indexes = rng.choice(self.offline_data_x.shape[0], size=self.batch_size, replace=False)
        initial_sequences = self.offline_data_x[random_indexes]
        if self.config.task == 'TFBind10-Exact-v1':
            initial_genotypes = [DNAGenotype(seq) for seq in initial_sequences]
        elif self.config.task == 'UTR-ResNet-v0-CUSTOM':
            initial_genotypes = [RNAGenotype(seq) for seq in initial_sequences]
        else:
            raise ValueError(f"Unknown task: {self.config.task}. Supported: TFBind10-Exact-v1, UTR-ResNet-v0-CUSTOM")
        logger.info(f"index of initial sequences:\n { random_indexes[:5]}")
        return initial_genotypes

    def _nucleotides_frequencies(self, x: BioSeqGenotype) -> np.ndarray:
        """
        Calculate the frequencies of nucleotides in a sequence.
        :return: numpy array with frequencies of the nucleotides 0, 1, 2
        """
        freq = np.zeros(3, dtype=float)
        for letter in x.sequence:
            if letter == 0:
                freq[0] += 1
            elif letter == 1:
                freq[1] += 1
            elif letter == 2:
                freq[2] += 1
        # Normalize frequencies
        freq /= len(x.sequence)

        return freq

    def _similarity_based_bd(self, x: BioSeqGenotype) -> np.ndarray:
        """
        Calculate the similarity-based behavioral descriptor.
        :param x: RNAGenotype
        :return: numpy array with the similarity-based behavioral descriptor
        """
        dists = np.array([-Levenshtein.distance(x.sequence, yi.sequence) for yi in self.reference_set])  # distances
        phi_n = softmax(dists)  # normalized distances via softmax
        dn = - np.dot(phi_n, dists) / self.R_normalization_constant
        bd = np.exp(dn) * (phi_n @ self.projection_matrix)
        return bd

    def to_phenotype(self, x: BioSeqGenotype) -> Phenotype:
        """
        Convert a genotype to a phenotype.
        :param x: genotype
        :return: phenotype
        """
        if self.config.bd_type == "freq" or self.config.bd_type == "nucleotides_frequencies":
            bd = self._nucleotides_frequencies(x)
        elif self.config.bd_type == "similarity_based":
            bd = self._similarity_based_bd(x)
        else:
            raise ValueError(f"Unknown bd_type: {self.config.bd_type}. Supported: freq, similarity_based")

        if self.config.normalize_bd:
            # normalize according to the min and max values and clip to [0, 1]
            bd = (bd - self.bd_min) / (self.bd_max - self.bd_min)

        return bd

    def _load_ref_set(self) -> np.array:
        """
        Load the reference set from the offline data directory.
        :return: numpy array of the reference set
        """
        random_indexes = self.rng.choice(self.offline_data_x.shape[0], size=self.config.size_of_refs_collection, replace=False)
        logger.info(f"index of reference set:\n { random_indexes[:5]}")
        reference_set = self.offline_data_x_gen[random_indexes]
        return reference_set.tolist() #todo: work with np arrays instead of lists

    def _random_seq(self) -> list[int]:
        seq = [self.rng.choice(self.config.alphabet) for _ in range(self.config.sequence_length)]
        return seq

    def random(self) -> list[BioSeqGenotype]:
        """
        Generate a batch of random genotypes.
        """
        if self.config.task == 'TFBind10-Exact-v1':
            return [DNAGenotype(self._random_seq()) for _ in range(self.batch_size)]
        elif self.config.task == 'UTR-ResNet-v0-CUSTOM':
            return [RNAGenotype(self._random_seq()) for _ in range(self.batch_size)]
        else:
            raise ValueError(f"Unknown task: {self.config.task}. Supported: TFBind10-Exact-v1, UTR-ResNet-v0-CUSTOM")

    def mutate(self, genomes: list[BioSeqGenotype]) -> list[BioSeqGenotype]:
        """
        Mutate a list of genomes by applying the mutation function to each genome.
        """     # TODO: the 50 steps limitation from the paper should be implemented here
        return self.mutation_model.mutate(genomes)

    def fitness(self, x: BioSeqGenotype) -> float:
        """
        Evaluate the fitness of the sequence using a list of scoring functions. (scoring ensemble)
        :param x: BioSeqGenotype
        :return: fitness score (float)
        """ # todo: make the fitness function work in batch mode
        fitness = self.fitness_function([x])
        fitness = fitness[0]
        return fitness

    def fitness_batch(self, genotypes: list[BioSeqGenotype]) -> list[float]:
        """
        Evaluate the fitness of a batch of sequences using a list of scoring functions. (scoring ensemble)
        :param genotypes: list of BioSeqGenotype
        :return: list of fitness scores (float)
        """
        number_of_internal_batches = len(genotypes) / self.fitness_function.config.batch_size
        number_of_internal_batches = int(np.ceil(number_of_internal_batches))  # round up
        outputs = []
        for i in range(number_of_internal_batches):
            # get the batch
            batch_start = i * self.fitness_function.config.batch_size
            batch_end = (i + 1) * self.fitness_function.config.batch_size if i < number_of_internal_batches - 1 else len(genotypes)
            batch_genotypes = genotypes[batch_start:batch_end]
            fitness = self.fitness_function(batch_genotypes)
            outputs.append(fitness)

        outputs = np.concatenate(outputs, axis=0).tolist()
        return outputs

    def eval_with_oracle(self, genotypes: list[BioSeqGenotype], downsampled_genotypes: list[BioSeqGenotype] = None, k=128, save_dir: str | Path = None) -> dict:
        """
        Evaluate a list of genotypes using the oracle model.
        The oracle model is used to evaluate the solutions after the optimization process.
        (not used in the optimization process)
        :param genotypes: list of RNAGenotype
        :param k: number of top solutions to consider for evaluation (w.r.t. the oracle scores)
        :param save_dir: directory to save the evaluation results and plots
        :return: max, diversity, mean, and novelty scores for all solutions, and for the top k solutions.
        """
        results_normalized = evaluate_solutions_set(
            solutions=genotypes,
            downsampled_solutions=downsampled_genotypes,
            ref_solutions=self.reference_set,
            oracle=self.oracle,
            max_score=self.max_output,
            min_score=self.min_output,
            k=k,
            plot=(save_dir is not None),
            save_path=os.path.join(save_dir, "normalized") if save_dir is not None else None,
            use_oracle_embeddings=(self.config.task == 'UTR-ResNet-v0-CUSTOM')
        )
        results_unnormalized = evaluate_solutions_set(
            solutions=genotypes,
            downsampled_solutions=downsampled_genotypes,
            ref_solutions=self.reference_set,
            oracle=self.oracle,
            max_score=1.0,
            min_score=0.0,
            k=k,
            plot=(save_dir is not None),
            save_path=os.path.join(save_dir, "unnormalized") if save_dir is not None else None,
            use_oracle_embeddings=(self.config.task == 'UTR-ResNet-v0-CUSTOM')
        )
        results = {
            "normalized/": results_normalized,
            "unnormalized/": results_unnormalized
        }
        return results

