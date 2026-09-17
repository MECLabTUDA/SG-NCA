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



#trainer = UNetTrainer(CholecDataset, "tiny_unet", train_w_softmax=True, classes=[1,2,3,4,5,6,7,8,9,10])
trainer = UNetTrainer(CataractsDataset, "tiny_unet", train_w_softmax=True, classes=list(range(1, 18)))
#trainer.max_num_epochs = 3
trainer.train(dryrun=False)
trainer.save_checkpoint()
trainer.eval("val", "dices")

trainer = UNetTrainer.load_checkpoint(trainer.get_save_path(), load_best=True)
trainer.eval("val", "dices_best")
exit()


trainer = BaseTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CataractsDataset/snowy-vortex-18") # SurgicalSegFormer
#trainer = BaseTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CholecDataset/spring-star-14") # SurgicalSegFormer






#trainer = BaseTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CataractsDataset/vivid-aardvark-19") # NCA
#trainer.initial_lr = 1e-4



#trainer = BaseTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CataractsDataset/colorful-blaze-8") # UNet
trainer.train_relation_predictor(dryrun=True)
trainer.save_checkpoint()
#trainer = BaseTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CataractsDataset/peach-planet-26") 
#trainer = BaseTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CataractsDataset/dummy-nqx6e3tm")
print(trainer.eval_relation_predictor("val"))
exit()


#trainer = IncrementalNCATrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CataractsDataset/vivid-aardvark-19")
#trainer.train_relation_predictor(dryrun=False)
#trainer.save_checkpoint()
#print(trainer.eval_relation_predictor("val"))
#exit()
#
#trainer = UNetTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CholecDataset/spring-star-14")
#trainer.train_relation_predictor(dryrun=True)
#trainer.save_checkpoint()
#print(trainer.eval_relation_predictor("val"))
#exit()

trainer = UNetTrainer(CholecDataset, "swinunet", train_w_softmax=True, classes=[1,2,3,4,5,6,7,8,9,10])
#trainer = UNetTrainer(CataractsDataset, "swinunet", train_w_softmax=True, classes=list(range(1, 18)))
trainer.train(dryrun=False)
trainer.save_checkpoint()
trainer.eval("val")
exit()
#trainer = BaseTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CataractsDataset/colorful-blaze-8/")
#trainer.eval("val")

#trainer = IncrementalNCATrainer(CataractsDataset)
#trainer.train([1,2,3,5,7], dryrun=False)
#trainer.save_checkpoint()
#trainer.train([4,6,14], dryrun=False)
#trainer.save_checkpoint()

#trainer = IncrementalNCATrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CataractsDataset/fallen-sun-10/")
#trainer.train([8,12,16], dryrun=False)
#trainer.save_checkpoint()
#trainer.train([9,10], dryrun=False)
#trainer.save_checkpoint()
#trainer.train([15], dryrun=False)
#trainer.save_checkpoint()
#trainer.train([13], dryrun=False)
#trainer.save_checkpoint()
trainer = IncrementalNCATrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CataractsDataset/vague-wildflower-15/")
trainer.train([11], dryrun=False)
trainer.save_checkpoint()
trainer.train([17], dryrun=False)
trainer.save_checkpoint()
trainer.eval("val")


#trainer = BaseTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CholecDataset/treasured-resonance-5")
#assert isinstance(trainer, IncrementalNCATrainer)
##trainer.eval("val")
#trainer.train_relation_predictor(dryrun=False)
#trainer.save_checkpoint()

#trainer = BaseTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CholecDataset/wild-fire-10")
#assert isinstance(trainer, RelationClassifierTrainer)
#trainer.eval("val")
#print(trainer.eval_relation_predictor("val"))



#trainer = UNetTrainer(CholecDataset, "segformer", train_w_softmax=True, classes=[1,2,3,4,5,6,7,8,9,10])
#trainer.train(dryrun=False)
#trainer.save_checkpoint()
#trainer.eval("val")

#trainer = IncrementalNCATrainer(CholecDataset)
#trainer.train([2,4,5,6,9], dryrun=False)
#trainer.save_checkpoint()
#trainer.train([1,3,10], dryrun=False)
#trainer.save_checkpoint()
#trainer.train([7], dryrun=False)
#trainer.save_checkpoint()
#trainer.train([8], dryrun=False)
#trainer.save_checkpoint()



#trainer = BaseTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CataractsDataset/dummy-3ca3zt92")
#trainer.train([4,5,6], dryrun=True)
#trainer.save_checkpoint()

#trainer = BaseTrainer.load_checkpoint("/local/scratch/clmn1/videoNCA/CataractsDataset/vibrant-brook-7")
#trainer.eval("val")