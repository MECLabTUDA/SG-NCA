from datetime import datetime
import os
from data.cholec import CholecDataset
from data.cataracts import CataractsDataset
import torch
from evaluation.dice import compute_dice
from evaluation.eval_framewise import eval_framewise
from models.load_continual_nca import load_continual_nca
from models.OctreeNCA2D import OctreeNCA2D
import tqdm
import torch.nn.functional as F
from models.cholec80 import get_cholec_model
from training_utils import training
from training_utils.trainer import IncrementalNCATrainer, RelationClassifierTrainer, UNetTrainer, BaseTrainer
from training_utils.trainer import UNetTrainer
from utils.paths import path_results, path_wandb_out
from unet import UNet2D
import wandb

from utils.wandb_utils import init_wandb

def train_nca():
    #trainer = IncrementalNCATrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CholecDataset/dashing-wildflower-38", load_best=True) # NCA (smallest)
    trainer = IncrementalNCATrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CataractsDataset/charmed-mountain-80", load_best=True) # NCA (smallest)
    trainer = IncrementalNCATrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CataractsDataset/earthy-eon-103", load_best=True) # NCA (smallest)
    trainer.relation_config["num_frames"] = 8
    trainer.relation_config["time_span"] = 1.0
    trainer.relation_config["method"] = "direct_prediction"
    trainer.relation_config["feature_extraction"]["from_all_levels"] = True
    trainer.relation_config["feature_extraction"]["expand_nca"] = False
    trainer.relation_config["feature_extraction"]["area_threshold"] = 150
    trainer.relation_config["feature_extraction"]["frame_encoding"] = None # each frame will be encoded to this dimension
    trainer.relation_config["feature_extraction"]["clip_encoding"] = None # the whole clip will be encoded to this dimension (after stacking all frames)

    trainer.initial_lr = 1e-4
    return trainer

def baseline():
    #trainer = UNetTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CholecDataset/spring-star-14", load_best=False) # SegFormer
    #trainer = UNetTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CholecDataset/devoted-glade-13", load_best=False) # UNet
    #trainer = UNetTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CholecDataset/dutiful-resonance-15", load_best=False) # Swin UNet
    #trainer = UNetTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CholecDataset/dark-silence-77", load_best=False) # tiny UNet
    
    #trainer = UNetTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CataractsDataset/snowy-vortex-18", load_best=False) # SegFormer
    #trainer = UNetTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CholecDataset/devoted-glade-13", load_best=False) # UNet
    #trainer = UNetTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CataractsDataset/jumping-galaxy-22", load_best=False) # Swin UNet
    trainer = UNetTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CataractsDataset/fresh-dream-108", load_best=True) # tiny UNet
    trainer.relation_config["num_frames"] = 8
    trainer.relation_config["time_span"] = 1.0
    return trainer

def setup_direct(trainer):
    trainer.relation_config["method"] = "direct_prediction"
    trainer.relation_config["feature_extraction"]["area_threshold"] = 150
    trainer.relation_config["feature_extraction"]["frame_encoding"] = 32 # each frame will be encoded to this dimension
    trainer.relation_config["feature_extraction"]["clip_encoding"] = 128 # the whole clip will be encoded to this dimension (after stacking all frames)
    trainer.relation_config["training"] = {}
    trainer.relation_config["training"]["balanced_sampling"] = False
    trainer.relation_config["training"]["class_weighted_bce"] = False
    trainer.relation_config["training"]["pair_wise_ranking_loss"] = False
    return trainer

def setup_motif(trainer):
    trainer.relation_config["method"] = "motif"
    trainer.relation_config["feature_extraction"]["area_threshold"] = 150
    trainer.relation_config["feature_extraction"]["frame_encoding"] = None # each frame will be encoded to this dimension
    trainer.relation_config["feature_extraction"]["clip_encoding"] = None # the whole clip will be encoded to this dimension (after stacking all frames)
    return trainer

def setup_tiny_motif(trainer):
    trainer = setup_motif(trainer)
    trainer.relation_config["feature_extraction"]["motif_hidden_dim"] = 128
    return trainer

trainer = baseline()
trainer = setup_tiny_motif(trainer)
trainer.print_relation_predictor()

#print(trainer.relation_config)
#exit()

trainer.train_relation_predictor(dryrun=False)
trainer.save_checkpoint()
trainer.eval_relation_predictor("val")
trainer.eval_relation_predictor("test")
exit()

