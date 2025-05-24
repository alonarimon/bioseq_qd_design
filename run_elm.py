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

def setup_logging(log_file):
   

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    file_handler.setFormatter(formatter)

    # Attach handler to root
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # Optionally add handler to all existing loggers
    # This can be useful in case some modules have created their loggers early
    for name in logging.root.manager.loggerDict:
        if name not in ["wandb", "helical", "dill"]:
            logging.getLogger(name).addHandler(file_handler)
            logging.getLogger(name).setLevel(logging.INFO)
            logging.getLogger(name).propagate = True

    # Set the logging level for the helical library to WARNING
    logging.getLogger("helical").setLevel(logging.WARNING)

@hydra.main(
    config_name="elmconfig",
    version_base="1.2",
)
def main(config):

    # Set up wandb
    config_dict = OmegaConf.to_container(config, resolve=True)

    run_group = f"{config.wandb_group}_{config.env.task}"
    if config.run_name != "":
        run_name = f"{config.run_name} BD {config.env.bd_type} FITNESS {config.fitness_model.model_name} MUTATOR {config.mutation_model.model_name}"
    else:
        run_name = f"BD {config.env.bd_type} FITNESS {config.fitness_model.model_name} MUTATOR {config.mutation_model.model_name}"    
    
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
    # Set logging to logs file
    os.makedirs(config.output_dir, exist_ok=True)
    log_file = os.path.join(config.output_dir, "run_elm.log")
    setup_logging(log_file)
    logging.info("Starting ELM run")
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

