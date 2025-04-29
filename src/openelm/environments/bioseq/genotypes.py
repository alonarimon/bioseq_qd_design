from typing import Optional

from openelm.environments import Genotype
from openelm.environments.base import Phenotype

MAP_INT_TO_LETTER = {
    0: "A",
    1: "C",
    2: "G",
    3: "U",
} # todo: check if this is correct


class RNAGenotype(Genotype):
    """
    A simple genotype class for RNA bioseq generation. (without llms)
    """

    def __init__(self, sequence: list[int]):
        self.sequence = sequence

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
        return "".join([MAP_INT_TO_LETTER[letter] for letter in self.sequence])

