import wandb

from utils.paths import  path_wandb_out

def init_wandb(experiment_name, config, dryrun):
    wandb.init(project=experiment_name, 
           mode="disabled" if dryrun else "online",
           dir=path_wandb_out,
           config=config)