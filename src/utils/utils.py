import random
import numpy as np
import torch
import os
import json
import wandb
from matplotlib.figure import Figure
import PIL.Image
import sys
from safetensors import safe_open
import io
from omegaconf import OmegaConf
import pickle
import pandas as pd
from loguru import logger

def is_rank0():
    return os.environ.get('RANK', '0') == '0'
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # Disabled because RuntimeError: "fill_empty_deterministic_" not implemented for 'ComplexHalf'

def init_rnd_seeds(worker_id):
    """
    Sets random seed of a worker based on the global seed.
    """
    seed = torch.initial_seed() % 2**30
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

def create_folder_if_not_exists(path : str):
    """
        Creates a folder if it does not exist
    """
    if not os.path.exists(path):
        os.makedirs(path)


def compute_gradient_statistics(model, iteration):
    d = {"iteration": iteration}
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and ("bias" not in n):
            mean_grad = p.grad.abs().mean()
            max_grad = p.grad.abs().max()

            d[f"{n}_mean_grad"] = mean_grad
            d[f"{n}_max_grad"] = max_grad

    return d


def log_fig_to_wandb(fig : Figure, name : str):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = PIL.Image.open(buf)
    img = wandb.Image(img)
    wandb.log({name: img})

def save_config(conf):
    """
    Saves the config file for a given checkpoint name as a pickeled dictionnary.
    """
    with open(f'{conf.experiment.dir}/{conf.experiment.name}/config.pkl', 'wb') as f:
        pickle.dump(conf, f)


def load_config(conf):
    """
    Loads a config file for given  name. If no file is found, returns an empty dictionary.
    """
    assert os.path.exists(f'{conf.experiment.dir}/{conf.experiment.name}/config.pkl'), f"No config file found for {conf.experiment.name}."

    with open(f'{conf.experiment.dir}/{conf.experiment.name}/config.pkl', 'rb') as f:
        config = pickle.load(f)
    return config

def load_checkpoint_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device='cpu') as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors

def setup_wandb_and_config(conf, tags=None):
    """
    Sets up wandb and saves the config file.
    """
    assert conf.experiment.wandb_mode in ["online", "offline", "disabled"], f"Received {conf.experiment.wandb_mode} for wandb_mode. Must be one of ['online', 'offline', 'disabled']"
    wandb_run_id_path = f'{conf.experiment.dir}/{conf.experiment.name}/wandb_run_id.txt'
    if conf.training.resume and os.path.exists(wandb_run_id_path) and \
            len(os.listdir(f'{conf.experiment.dir}/{conf.experiment.name}/checkpoints')) > 0:
        
        with open(wandb_run_id_path, 'r') as f:
            conf.experiment.wandb_run_id = f.read().strip()
    

        conf.resume_from_checkpoint = True

        if is_rank0():
            wandb.init(project=conf.experiment.wandb_project, name=conf.experiment.name, entity=conf.experiment.wandb_entity,
                        mode=conf.experiment.wandb_mode, dir=f'{conf.experiment.dir}/{conf.experiment.name}/wandb', id=conf.experiment.wandb_run_id, resume="must",
                        config=conf)
        
            logger.info(f'Resuming wandb run {conf.experiment.wandb_run_id}')
    else:
        conf.resume_from_checkpoint = False
        if is_rank0():
            wandb.init(project=conf.experiment.wandb_project, name=conf.experiment.name, entity=conf.experiment.wandb_entity,
                        mode=conf.experiment.wandb_mode, dir=f'{conf.experiment.dir}/{conf.experiment.name}/wandb',
                        tags=tags, config=conf)
            with open(wandb_run_id_path, 'w') as f:
                f.write(wandb.run.id)
            conf.experiment.wandb_run_id = wandb.run.id if conf.experiment.wandb_mode != "disabled" else "debug-run"
            save_config(conf)
            logger.info(f'Starting fresh wandb run {conf.experiment.wandb_run_id} with name {conf.experiment.name}')
    return conf



def get_mean_std(file_path, log1p):
    normalization = pd.read_csv(file_path)
    if log1p: 
        log1p_mean = np.array(normalization["log1p_mean"])[:, np.newaxis, np.newaxis]
        log1p_std = np.array(normalization["log1p_std"])[:, np.newaxis, np.newaxis]
        return log1p_mean, log1p_std
    else: 
        mean = np.array(normalization["mean"])[:, np.newaxis, np.newaxis]
        std = np.array(normalization["std"])[:, np.newaxis, np.newaxis]
        return mean, std
    