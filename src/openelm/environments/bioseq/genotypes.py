from typing import Optional
from abc import ABC, abstractmethod

import numpy
from openelm.environments import Genotype
from openelm.environments.base import Phenotype



class BioSeqGenotype(Genotype, ABC):
    """
    A simple genotype class for bioseq generation. (without llms)
    """
    def __init__(self, sequence): #TODO: maybe not efficient?
        """
        Initialize the genotype with a sequence.
        :param sequence: The sequence to initialize the genotype with. Can be a string or a list of integers.
        """
        if isinstance(sequence, str):
            self.sequence = [self.map_letter_to_int(letter) for letter in sequence]
            self.sequence_str = sequence
        elif isinstance(sequence, list) or isinstance(sequence, numpy.ndarray):
            self.sequence_str = "".join([self.map_int_to_letter(letter) for letter in sequence])
            self.sequence = sequence
        else:
            raise ValueError(f"Invalid sequence type: {type(sequence)}. Expected str or list[int].")

    def to_phenotype(self) -> Optional[Phenotype]:
        """
        Convert the genotype to a phenotype.
        :return: Phenotype representation of the genotype.
        """
        raise NotImplementedError("Phenotype conversion is not implemented for BioSeqGenotype, expect to use the environment's to_phenotype method.")

    def __str__(self):
        """
        Convert the genotype to a string representation.
        """
        return self.sequence_str

    @abstractmethod
    def map_letter_to_int(self, letter: str) -> int:
        """
        Map a letter to its corresponding integer representation.
        :param letter: The letter to map.
        :return: The integer representation of the letter.
        """
        raise NotImplementedError("Not Implemented")
    
    @abstractmethod
    def map_int_to_letter(self, integer: int) -> str:
        """
        Map an integer to its corresponding letter representation.
        :param integer: The integer to map.
        :return: The letter representation of the integer.
        """
        raise NotImplementedError("Not Implemented")

    

MAP_INT_TO_LETTER_RNA = {
    0: "A",
    1: "C",
    2: "G",
    3: "U",
} # todo: check if this is correct

MAP_LETTER_TO_INT_RNA = {v: k for k, v in MAP_INT_TO_LETTER_RNA.items()}

class RNAGenotype(BioSeqGenotype):
    """
    A simple genotype class for RNA bioseq generation. (without llms)
    """

    def __init__(self, sequence):
        super().__init__(sequence)

    def to_phenotype(self) -> Optional[Phenotype]:
        """
        Convert the genotype to a phenotype.
        :return: Phenotype representation of the genotype.
        """
        raise NotImplementedError("Phenotype conversion is not implemented for RNAGenotype, expect to use the environment's to_phenotype method.")    

    def map_letter_to_int(self, letter: str) -> int:
        """
        Map a letter to its corresponding integer representation.
        :param letter: The letter to map.
        :return: The integer representation of the letter.
        """
        return MAP_LETTER_TO_INT_RNA[letter]
    
    def map_int_to_letter(self, integer: int) -> str:
        """
        Map an integer to its corresponding letter representation.
        :param integer: The integer to map.
        :return: The letter representation of the integer.
        """
        return MAP_INT_TO_LETTER_RNA[integer]

MAP_INT_TO_LETTER_DNA = {
    0: "A",
    1: "C",
    2: "G",
    3: "T",
}

MAP_LETTER_TO_INT_DNA = {v: k for k, v in MAP_INT_TO_LETTER_DNA.items()}

class DNAGenotype(BioSeqGenotype):
    """
    A simple genotype class for RNA bioseq generation. (without llms)
    """

    def __init__(self, sequence):
        super().__init__(sequence)

    def to_phenotype(self) -> Optional[Phenotype]:
        """
        Convert the genotype to a phenotype.
        :return: Phenotype representation of the genotype.
        """
        raise NotImplementedError("Phenotype conversion is not implemented for RNAGenotype, expect to use the environment's to_phenotype method.")

    def map_letter_to_int(self, letter: str) -> int:
        """
        Map a letter to its corresponding integer representation.
        :param letter: The letter to map.
        :return: The integer representation of the letter.
        """
        return MAP_LETTER_TO_INT_DNA[letter]
    
    def map_int_to_letter(self, integer: int) -> str:
        """
        Map an integer to its corresponding letter representation.
        :param integer: The integer to map.
        :return: The letter representation of the integer.
        """
        return MAP_INT_TO_LETTER_DNA[integer]

