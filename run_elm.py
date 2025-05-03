"""
This module gives an example of how to run the main ELM class.

It uses the hydra library to load the config from the config dataclasses in
configs.py.

This config file demonstrates an example of running ELM with the Sodarace
environment, a 2D physics-based environment in which robots specified by
Python dictionaries are evolved over.

"""
import logging
import os
from datetime import datetime

import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from openelm import ELM

logging.getLogger("helical").setLevel(logging.WARNING)

@hydra.main(
    config_name="elmconfig",
    version_base="1.2",
)
def main(config):

    # Set up wandb
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    config_dict = OmegaConf.to_container(config, resolve=True)
    wandb.init(project="bioseq_qd_design", group="run_elm", name=current_time, config=config_dict)
    wandb.config.update(config_dict)

    config.output_dir = HydraConfig.get().runtime.output_dir
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(config))
    print("-----------------  End -----------------")
    config = OmegaConf.to_object(config)

    elm = ELM(config)
    print(
        "Best Individual: ",
        elm.run(init_steps=config.qd.init_steps, total_steps=config.qd.total_steps),
    )
    artifact = wandb.Artifact('run_logs', type='log')
    artifact.add_file(os.path.join(config.output_dir, "run_elm.log"))
    wandb.log_artifact(artifact)
    wandb.finish()

if __name__ == "__main__":
    main()

