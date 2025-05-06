from typing import Optional
from abc import ABC, abstractmethod
from openelm.environments import Genotype
from openelm.environments.base import Phenotype



class BioSeqGenotype(Genotype, ABC):
    """
    A simple genotype class for bioseq generation. (without llms)
    """

    def __init__(self, sequence: list[int]):
        self.sequence = sequence

    def to_phenotype(self) -> Optional[Phenotype]:
        """
        Convert the genotype to a phenotype.
        :return: Phenotype representation of the genotype.
        """
        raise NotImplementedError("Phenotype conversion is not implemented for BioSeqGenotype, expect to use the environment's to_phenotype method.")

    @abstractmethod
    def __str__(self):
        """
        Convert the genotype to a string representation.
        """
        raise NotImplementedError("Not Implemented")

MAP_INT_TO_LETTER_RNA = {
    0: "A",
    1: "C",
    2: "G",
    3: "U",
} # todo: check if this is correct

class RNAGenotype(BioSeqGenotype):
    """
    A simple genotype class for RNA bioseq generation. (without llms)
    """

    def __init__(self, sequence: list[int]):
        super().__init__(sequence)

    def to_phenotype(self) -> Optional[Phenotype]:
        """
        Convert the genotype to a phenotype.
        :return: Phenotype representation of the genotype.
        """
        raise NotImplementedError("Phenotype conversion is not implemented for RNAGenotype, expect to use the environment's to_phenotype method.")

    def __str__(self):
        """
        Convert the genotype to a string representation.
        """
        return "".join([MAP_INT_TO_LETTER_RNA[letter] for letter in self.sequence])


MAP_INT_TO_LETTER_DNA = {
    0: "A",
    1: "C",
    2: "G",
    3: "T",
}

class DNAGenotype(BioSeqGenotype):
    """
    A simple genotype class for RNA bioseq generation. (without llms)
    """

    def __init__(self, sequence: list[int]):
        super().__init__(sequence)

    def to_phenotype(self) -> Optional[Phenotype]:
        """
        Convert the genotype to a phenotype.
        :return: Phenotype representation of the genotype.
        """
        raise NotImplementedError("Phenotype conversion is not implemented for RNAGenotype, expect to use the environment's to_phenotype method.")

    def __str__(self):
        """
        Convert the genotype to a string representation.
        """
        return "".join([MAP_INT_TO_LETTER_DNA[letter] for letter in self.sequence])

