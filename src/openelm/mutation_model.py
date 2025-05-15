import functools
import logging
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Generic, Optional

from helical.models.helix_mrna_generator.model.hg38_char_tokenizer import CharTokenizer
from helical.models.helix_mrna_generator.model.modelling_helix_mrna import HelixmRNAForCausalLM
from helical.models.helix_mrna_generator.sequence_gen import mutate_sequence
import numpy as np
import torch


from openelm.codegen import model_setup, set_seed, truncate
from openelm.configs import BioRandomModelConfig, ModelConfig, MutatorHelixConfig
from openelm.environments.base import GenoType
from openelm.environments.bioseq.genotypes import BioSeqGenotype, DNAGenotype, RNAGenotype
from openelm.utils.diff_eval import apply_diff, split_diff

logger = logging.getLogger(__name__)


def get_mutation_model(config: ModelConfig): #todo: move from this file
    if config.model_name == "bio_random":
        return RandomSequenceMutator(config=config) #todo ?
    elif config.model_name == "mutator_helix_mrna":
        return HelixMRNASequenceMutator(config=config)
    else:
        raise NotImplementedError


class MutationModel(ABC, Generic[GenoType]):
    """Base model class for all mutation models."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.device = torch.device("cuda") if config.cuda and torch.cuda.is_available() else torch.device("cpu")
        logger.info(f"initialized mutation model on {self.device}")
    
    def set_rng_state(self, rng_state: Optional[np.random.Generator]):
        self.rng = rng_state

    @abstractmethod
    def mutate(self, sequences: list[GenoType]) -> list[GenoType]:
        """
        Mutate a list of sequences.
        :param sequences: list of sequences to mutate
        :return: list of mutated sequences
        """
        raise NotImplementedError

class RandomSequenceMutator(MutationModel[BioSeqGenotype]):
    """
    A simple random sequence mutator for bioseq generation. (without llms)
    """
    def __init__(self, config: BioRandomModelConfig):
        super().__init__(config)
        self.alphabet = config.alphabet

    def _mutate(self, seq: BioSeqGenotype) -> BioSeqGenotype:
        """
        Mutate a sequence by randomly changing one letter.
        :param seq: sequence to mutate
        """ 
        # TODO: the 50 steps limitation from the paper should be implemented here
        mutation_position = self.rng.integers(self.config.gen_max_len - self.config.mutation_length + 1)
        mutated_seq = seq.sequence.copy()
        for i in range(mutation_position, mutation_position + self.config.mutation_length):
            new_letter = self.rng.choice([a for a in self.alphabet if a != seq.sequence[i]])
            mutated_seq[i] = new_letter
        if isinstance(seq, RNAGenotype):
            mutated_gen = RNAGenotype(mutated_seq)
        elif isinstance(seq, DNAGenotype):
            mutated_gen = DNAGenotype(mutated_seq)
        return mutated_gen
    
    def mutate(self, sequences: list[BioSeqGenotype]) ->  list[BioSeqGenotype]:
        """
        Mutate a list of sequences.
        :param sequences: list of sequences to mutate
        """
        return [self._mutate(seq) for seq in sequences]
    
class HelixMRNASequenceMutator(MutationModel[RNAGenotype]):
    """
    A sequence mutator for helix mRNA generation.
    """
    def __init__(self, config: MutatorHelixConfig):
        super().__init__(config)
        self.model = HelixmRNAForCausalLM.from_pretrained(
                "helical-ai/helix-mRNA",
                attn_implementation="flash_attention_2",
                ).to(self.device) # model in evaluation mode by default

        self.tokenizer = CharTokenizer(
            model_max_length=12288,
            padding_side="right")
        logger.info(f"loaded helix mRNA mutator model with config: {config}")
        logger.info(torch.cuda.get_device_name())
        logger.info(torch.get_autocast_gpu_dtype())  # Should show bfloat16 or float16
    

    #TODO: use scores?
    def mutate(self, sequences: list[RNAGenotype]) -> list[RNAGenotype]:
        mutation_position = self.rng.integers(self.config.gen_max_len - self.config.mutation_length + 1)
        str_sequences = [str(seq) for seq in sequences]
        mutated, scores = mutate_sequence(model=self.model,
                                        device=self.device,
                                        tokenizer=self.tokenizer,
                                        mutation_position=mutation_position,
                                        original_seq=str_sequences,
                                        mutation_length=self.config.mutation_length,
                                        softmax_temperature=self.config.temp,
                                        top_k=self.config.top_k,
                                        logits_threshold=self.config.logits_threshold,
                                        top_p=self.config.top_p)
        
        return [RNAGenotype(seq) for seq in mutated] 