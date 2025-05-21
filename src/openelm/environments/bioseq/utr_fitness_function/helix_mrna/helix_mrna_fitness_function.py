import os

import torch

from helical.models.helix_mrna import HelixmRNAFineTuningModel, HelixmRNAConfig
from openelm.configs import FitnessHelixMRNAConfig
from openelm.environments.bioseq.bioseq import RNAGenotype
from openelm.environments.bioseq.fitness_model import FitnessModel


class HelixMRNAFitnessFunction(FitnessModel[RNAGenotype]):
    """
    A wrapper class for a fine-tuned helix_mrna fitness function.
    """
    def __init__(self, config: FitnessHelixMRNAConfig):
        """
        Initializes the helix_mrna fitness function.
        :param config: Configuration object containing the model parameters.
        """
        super().__init__(config)
        self.model = self.load_model()  # Load your model here
        self.model.to(self.device)
        self.model.eval()


    def load_model(self):
        """
        Loads the fine-tuned helix_mrna model from saved files.
        :param model_path: Directory containing base_model.pt, head.pt, config.pt
        """
        model_path = self.config.model_path
        # check that model_path exists and contains the required files
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        required_files = ["base_model.pt", "head.pt", "config.pt"]
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                raise FileNotFoundError(f"Required file {file} not found in {model_path}.")

        # Load the model
        config_dict = torch.load(os.path.join(model_path, "config.pt"))
        print(f"Loaded config: {config_dict}")
        helix_config = HelixmRNAConfig(batch_size=config_dict["batch_size"],
                                      device=self.device,
                                      max_length=config_dict["input_size"],
                                      val_batch_size=config_dict["val_batch_size"],
                                       nproc=config_dict["nproc"])
        helix_model = HelixmRNAFineTuningModel(
            helix_mrna_config=helix_config,
            fine_tuning_head="regression",
            output_size=1
        )
        helix_model.load_model(model_path)
        helix_model.eval()
        return helix_model

    def __call__(self, genotypes: list[RNAGenotype]) -> list[float]:
        """
        Process a batch of sequences and return scores.
        :param genotypes: Input genotypes to be scored.
        :return: Scores for the input sequences.
        """ #todo: move to tensors / numpy and not lists?
        if len(genotypes) > self.config.batch_size:
            raise ValueError(f"Batch size {len(genotypes)} exceeds the configured batch size {self.batch_size}.")

        sequences_str = [str(g) for g in genotypes]#todo: not efficient,  - better to work only with strings
        input_dataset = self.model.process_data(sequences_str)
        with torch.no_grad():
            output = self.model.get_outputs(input_dataset)
        if len(genotypes) == 1:
            output = [output]
        else:
            output = output.squeeze().tolist()
        return output


