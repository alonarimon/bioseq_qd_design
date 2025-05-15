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
import subprocess
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0)) 

from openelm import ELM

logging.getLogger("helical").setLevel(logging.WARNING)

@hydra.main(
    config_name="elmconfig",
    version_base="1.2",
)
def main(config):

    # Set up wandb
    config_dict = OmegaConf.to_container(config, resolve=True)
    wandb_group = config.wandb_group
    run_name = config.run_name
    if config.env.env_name == "qd_bio_rna":
        run_group = f"{wandb_group}_{config.env.task}"
        run_name = f"{run_name}_{config.env.bd_type}_{config.fitness_model.model_name}_{config.mutation_model.model_name}"
    
    wandb.init(
        project="bioseq_qd_design", 
        name=run_name, 
        group=run_group, 
        config=config_dict,
        mode=config.wandb_mode,
        dir="/workdir/wandb",)  
    wandb.config.update(config_dict)
    wandb.config["gpu_name"] = torch.cuda.get_device_name(0)

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

