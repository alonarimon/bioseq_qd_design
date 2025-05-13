from typing import Any, Optional, Type

from hydra.core.hydra_config import HydraConfig

from openelm.configs import ELMConfig
from openelm.environments.base import BaseEnvironment


def load_env(env_name: str) -> Type[BaseEnvironment]:
    if env_name == "sodarace":
        from openelm.environments.sodaracer.sodarace import Sodarace

        return Sodarace
    elif env_name == "image_evolution":
        from openelm.environments.base import ImageOptim

        return ImageOptim
    elif env_name == "match_string":
        from openelm.environments.base import MatchString

        return MatchString
    elif env_name == "function_optim":
        from openelm.environments.base import FunctionOptim

        return FunctionOptim
    elif env_name == "p3_probsol":
        from openelm.environments.p3.p3 import P3ProbSol

        return P3ProbSol
    elif env_name == "p3_problem":
        from openelm.environments.p3.p3 import P3Problem

        return P3Problem
    elif env_name == "prompt_evolution":
        from openelm.environments.prompt.prompt import PromptEvolution

        return PromptEvolution
    elif env_name == "qdaif":
        from openelm.environments.poetry import PoetryEvolution

        return PoetryEvolution
    elif env_name == "qd_bio_rna":
        from openelm.environments.bioseq.bioseq import BioSeqEvolution

        return BioSeqEvolution
    else:
        raise ValueError(f"Unknown environment {env_name}")


def load_algorithm(algorithm_name: str) -> Any:
    if algorithm_name == "mapelites":
        from openelm.algorithms.map_elites import MAPElites

        return MAPElites
    elif algorithm_name == "cvtmapelites":
        from openelm.algorithms.map_elites import CVTMAPElites

        return CVTMAPElites


class ELM:
    def __init__(self, config: ELMConfig, env: Optional[Any] = None) -> None:
        """
        The main class of ELM.

        This class will load a diff model, an environment, and a QD algorithm
        from the passed config.

        Args:
            config: The config containing the diff model, environment, and QD algorithm.
            env (Optional): An optional environment to pass in. Defaults to None.
        """
        self.config: ELMConfig = config
        hydra_conf = HydraConfig.instance()
        if hydra_conf.cfg is not None:
            self.config.qd.output_dir = HydraConfig.get().runtime.output_dir
        
        if env is None:
            self.environment = load_env(self.config.env.env_name)(
                config=self.config.env,
                mutation_model_config=self.config.mutation_model,
                fitness_model_config=self.config.fitness_model,
            )
        elif isinstance(env, type(BaseEnvironment)):
            self.environment = env(
                config=self.config.env,
                mutation_model_config=self.config.mutation_model,
                fitness_model_config=self.config.fitness_model,
            )
        elif isinstance(env, BaseEnvironment):
            self.environment = env
        else:
            raise ValueError(f"Unknown environment {env.__name__}")
        
        self.qd_algorithm = load_algorithm(self.config.qd.qd_name)(
            env=self.environment,
            config=self.config.qd,
        )

    def run(
        self, init_steps: Optional[int] = None, total_steps: Optional[int] = None
    ) -> str:
        """
        Run the ELM algorithm to evolve the population in the environment.

        Args:
            init_steps: The number of steps to run the initialisation phase.
            total_steps: The number of steps to run the QD algorithm in total,
            including init_steps.

        Returns:
            str: A string representing the maximum fitness genotype. The
            `qd_algorithm` class attribute will be updated.
        """
        if init_steps is None:
            init_steps = self.config.qd.init_steps
        if total_steps is None:
            total_steps = self.config.qd.total_steps
        return self.qd_algorithm.search(init_steps=init_steps, total_steps=total_steps)
