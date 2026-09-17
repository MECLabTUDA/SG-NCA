import itertools
import os
from typing import List
import einops
from matplotlib import cm
from pyparsing import Optional
import sklearn.metrics
import torch
import tqdm
import numpy as np
from data.cholec import CholecDataset
from data.cataracts import CataractsDataset
from data.cholec_scene_graph import CholecSceneGraphDataset
from data.sampler import BalancedSampler
from evaluation.eval_framewise import eval_framewise
from evaluation.scene_graph_builder import SceneGraphBuilder
from evaluation.segmentation_visualizer import SegmentationVisualizer
from models.LargeOctreeNCA2D import LargeOctreeNCA2D
from models.cholec80 import get_cholec_model
from models.motif import MotifRelationClassifier
from models.surg_swin_unet import SurgicalSwinUNet
from models.surg_unet import SurgicalUNet
from models.surgical_mask2former import SurgicalMask2Former
from models.surgical_segformer import SurgicalSegFormer

from models.relation_classifier import RelationClassifier
from models.tiny_surgical_unet import TinySurgicalUNet
from models.transformer import TransformerRelationClassifier
from training_utils import training
from training_utils.scenegraph_evaluator import SceneGraphEvaluator
from training_utils.util import deep_merge
from utils.wandb_utils import init_wandb
import wandb
from utils.paths import path_results
import json
import torch.nn.functional as F
import pandas as pd
from collections import defaultdict


class BaseTrainer:
    def __init__(self, dataset_class):
        self.max_num_epochs = 30
        self.initial_lr = 1e-3
        self.batches_per_epoch = 10_000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_class = dataset_class
        self.network = None
        self.run_name = None

    def get_save_path(self):
        return os.path.join(
            path_results,
            self.dataset_class.__name__,
            self.run_name,
        )

    def get_config(self) -> dict:
        return {
            "run_name": self.run_name,
            "dataset": self.dataset_class.__name__,
            "trainer": self.__class__.__name__,
            "max_num_epochs": self.max_num_epochs,
            "initial_lr": self.initial_lr,
            "batches_per_epoch": self.batches_per_epoch,
            "model": self.network.__class__.__name__,
        }
    def parse_config(self, config: dict):
        self.run_name = config["run_name"]
        self.max_num_epochs = config["max_num_epochs"]
        self.initial_lr = config["initial_lr"]
        self.batches_per_epoch = config["batches_per_epoch"]

    def init_wandb(self, dryrun, relation_classification=False):
        init_wandb(
            experiment_name=f"surgical-nca.{self.dataset_class.__name__}",
            config=self.get_config(),
            dryrun=dryrun,
        )
        self.run_name = wandb.run.name
        if relation_classification:
            wandb.watch(self.relation_classifier, log="all", log_freq=100)
        else:
            wandb.watch(self.network, log="all", log_freq=100)
        

    def train(self, dryrun):
        raise NotImplementedError

    def eval(self, split: str, file_name="dices"):
        raise NotImplementedError

    def save_checkpoint(self, is_best=False):
        path = self.get_save_path()
        os.makedirs(path, exist_ok=True)
        json.dump(self.get_config(), open(os.path.join(path, "config.json"), "w"), indent=4)
        torch.save(self.network.state_dict(), os.path.join(path, "model.pth"))
        if is_best:
            torch.save(self.network.state_dict(), os.path.join(path, "model_best.pth"))
            with open(os.path.join(path, "best_info.txt"), "w") as f:
                f.write(f"{self.current_epoch}\n{self.best_val_dice}")

    @staticmethod
    def load_checkpoint(path, load_best=False) -> 'RelationClassifierTrainer':
        configs = json.load(open(os.path.join(path, "config.json"), "r"))
        dataset_class = {"CataractsDataset": CataractsDataset, "CholecDataset": CholecDataset}[configs["dataset"]]
        trainer_class = {"UNetTrainer": UNetTrainer, "IncrementalNCATrainer": IncrementalNCATrainer}[configs["trainer"]]
        trainer = trainer_class.load_checkpoint_internal(path, configs, dataset_class, load_best=load_best)
        trainer.dataset_class = dataset_class
        # Keep checkpoint loading portable. RelationClassifierTrainer.parse_config
        # previously rebuilt the path from the machine-local results directory,
        # which made copied checkpoints impossible to export on another machine.
        trainer._checkpoint_path = path
        trainer.parse_config(configs)
        del trainer._checkpoint_path
        return trainer
    
    def get_hard_map(self, frame_logits: torch.Tensor, cls: int) -> torch.Tensor:
        raise NotImplementedError
    
    def best_dice_callback(self, best_dice, epoch):
        self.best_val_dice = best_dice
        self.current_epoch = epoch
        self.save_checkpoint(is_best=True)


class RelationClassifierTrainer(BaseTrainer):
    def __init__(self, dataset_class):
        super().__init__(dataset_class)
        #self.temporal_window = 2
        self.is_relation_trained = False

        self.relation_config = {
            "num_frames": 8, # number of frames next to the main frame
            "time_span": 1.0, # in seconds, how far back to look for context frames
            "method": "direct_prediction", # "direct_prediction", "motif", "transformer"
            "feature_extraction": {
                "from_all_levels": True, # only to be used for NCA. Standard: False
                "area_threshold": 1,
                "fixed_mask": False,
                "feature_adaptation": False,   # int or False
                "additional_features": ["segmentation_size", "bbox"],
                "expand_nca": False,
                "motif_hidden_dim": 512,
            },
            "training":{
                "balanced_sampling": False,
                "class_weighted_bce": False,
                "pairwise_ranking_loss": False,
            }
        }

    def save_checkpoint(self, is_best=False):
        super().save_checkpoint(is_best=is_best)
        if not self.is_relation_trained:
            return
        path = self.get_save_path()
        torch.save(self.relation_classifier.state_dict(), os.path.join(path, "relation_classifier.pth"))

    def get_config(self):
        config = super().get_config()
        config["is_relation_trained"] = self.is_relation_trained
        config["relation_config"] = self.relation_config
        return config
    
    def parse_config(self, config):
        super().parse_config(config)
        self.is_relation_trained = config.get("is_relation_trained", False)
        # self.relation_config = config.get("relation_config", self.relation_config)
        self.relation_config = deep_merge(self.relation_config, config.get("relation_config", {}))
        if self.is_relation_trained:
            path = getattr(self, "_checkpoint_path", self.get_save_path())
            self._init_relation_classifier()
            self.relation_classifier.load_state_dict(torch.load(os.path.join(path, "relation_classifier.pth"),
                                                               map_location=self.device, weights_only=True))


    def _init_relation_classifier(self):
        additional_feature_dims = {
            "segmentation_size": 1,
            "bbox": 4,
        }
        if self.network.__class__ == LargeOctreeNCA2D:
            num_features = self.network.get_num_features(return_states_all_levels=self.relation_config["feature_extraction"].get("from_all_levels", False))
        else:
            num_features = self.network.get_num_features()

        num_features =  num_features + sum(
            additional_feature_dims[feat] for feat in self.relation_config["feature_extraction"]["additional_features"]
        )
        num_verbs = len(self.dataset_class.SCENE_GRAPH_DATASET.VERBS)-1
        if self.relation_config["method"] in ["direct_prediction"]:
            self.relation_classifier = RelationClassifier(feature_dim=num_features, 
                                                        num_frames=self.relation_config["num_frames"], 
                                                        num_verbs=num_verbs,
                                                        method=self.relation_config["method"],
                                                        frame_encoding=self.relation_config["feature_extraction"]["frame_encoding"],
                                                        clip_encoding=self.relation_config["feature_extraction"]["clip_encoding"]).to(self.device)
        elif self.relation_config["method"] == "motif":
            self.relation_classifier = MotifRelationClassifier(feature_dim=num_features, 
                                                               num_frames=self.relation_config["num_frames"],
                                                                num_classes=num_verbs, hidden_dim=self.relation_config["feature_extraction"]["motif_hidden_dim"]).to(self.device)
        elif self.relation_config["method"] == "transformer":
            self.relation_classifier = TransformerRelationClassifier(feature_dim=num_features, 
                                                               num_frames=self.relation_config["num_frames"],
                                                                num_classes=num_verbs).to(self.device)
        else:
            raise ValueError(f"Unknown relation prediction method {self.relation_config['method']}")

    @staticmethod
    def _collate_fn(batch: list):
        ids = [s['id'] for s in batch]
        frames = [s["frames"] for s in batch]
        # frames: list of (T, 3, H, W)
        frames = torch.stack(frames, dim=0)  # (B, T, 3, H, W)
        
        relations_dict = [s["relations_dict"] for s in batch]
        return {
            "id": ids,
            "frames": frames,
            "relations_dict": relations_dict,
        }

    def masked_avg_pool(self, feat_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # feat_map: (T, C, H, W)
        # mask: binary mask (T, H, W)
        assert feat_map.dim() == 4
        assert mask.dim() == 3 and mask.shape[0] == feat_map.shape[0]
        if self.relation_config["feature_extraction"]["fixed_mask"]:
            mask = mask[-1]
            masked_feat = feat_map * mask
            sum_feat = masked_feat.sum(dim=[2,3])  # (T, C)
            num_pixels = mask.sum() + 1e-6
            avg_feat = sum_feat / num_pixels  # (T, C)
        else:
            masked_feat = feat_map * mask.unsqueeze(1)  # (T, C, H, W)
            sum_feat = masked_feat.sum(dim=[2,3])  # (T, C)
            num_pixels = mask.sum(dim=[1,2]) + 1e-6  # (T,)
            avg_feat = sum_feat / num_pixels.unsqueeze(1)  # (T, C)

        return avg_feat


    def object_present(self, hard_map: torch.Tensor) -> bool:
        assert hard_map.dim() == 2
        assert hard_map.dtype == torch.bool
        # hard_map: (H, W)
        return hard_map.sum() > self.relation_config["feature_extraction"]["area_threshold"]

    
    def extract_features(self, states, hard_mask):
        # states: (T, C, H, W)
        # hard_map: binary mask (T, H, W)
        feature = self.masked_avg_pool(states, hard_mask) #(T C)
        additional_feature_vectors = []
        for additional_feature in self.relation_config["feature_extraction"]["additional_features"]:
            if additional_feature == "segmentation_size":
                # compute size of segmentation as additional feature
                sizes = hard_mask.sum(dim=[1,2], keepdim=True).float().squeeze(1)  # (T, 1)
                sizes = sizes / (hard_mask.shape[1] * hard_mask.shape[2])  # normalize to [0, 1]
                additional_feature_vectors.append(sizes)
            elif additional_feature == "bbox":
                # compute bounding box of segmentation as additional feature
                bboxes = []
                T = hard_mask.shape[0]
                for t in range(T):
                    ys, xs = torch.where(hard_mask[t])
                    if ys.numel() == 0:
                        # no object present
                        bbox = torch.tensor([0, 0, 0, 0], device=hard_mask.device).float()
                    else:
                        y_min = ys.min().float() / hard_mask.shape[1]
                        y_max = ys.max().float() / hard_mask.shape[1]
                        x_min = xs.min().float() / hard_mask.shape[2]
                        x_max = xs.max().float() / hard_mask.shape[2]
                        bbox = torch.tensor([x_min, y_min, x_max, y_max], device=hard_mask.device).float()
                    bboxes.append(bbox)
                bboxes = torch.stack(bboxes, dim=0)  # (T, 4)
                additional_feature_vectors.append(bboxes)
            else:
                raise ValueError(f"Unknown additional feature {additional_feature}")
        if additional_feature_vectors:
            additional_features = torch.cat(additional_feature_vectors, dim=1)  # (T, sum of additional feature dims)
            feature = torch.cat([feature, additional_features], dim=1)  # (T, C + sum of additional feature dims)
        return feature

    def get_seg_channel_of_class(self, cls: int):
        #TODO remove from this class
        raise NotImplementedError

    def infer_segmentation_and_graph(self, frames: torch.Tensor):
        assert frames.dim() == 5  # (B, T, C, H, W)
        assert frames.shape[0] == 1  # B=1 for inference 
        result = self.predict_relations(frames)
        return result
        sg_builder = SceneGraphBuilder()
        scene_graph = sg_builder.visualize_scene_graph(self, frame_logits, relation_logits, pair_proposals)
        segmentation_img = SegmentationVisualizer().visualize_segmentation(self, frame_logits[:, -1])
        return segmentation_img, scene_graph

    def build_object_features(self, frame_logits: torch.Tensor, frame_states: torch.Tensor) -> List[dict]:
        raise NotImplementedError

    def compute_features(self, frames: torch.Tensor):
        T = frames.shape[1]
        B = frames.shape[0]
        frames = einops.rearrange(frames, "B T C H W -> (B T) C H W").to(self.device)
        with torch.no_grad():
            network_output = self.network.compute_features(frames)
            frame_logits = network_output['logits']
            frame_states = network_output['features']

        frame_logits = einops.rearrange(frame_logits, "(B T) C H W -> B T C H W", B=B)
        frame_states = einops.rearrange(frame_states, "(B T) C H W -> B T C H W", B=B)
        return frame_logits, frame_states

    def predict_relations(self, frames: torch.Tensor=None, frame_logits: torch.Tensor=None, frame_states: torch.Tensor=None, relations_dict=None):
        if frame_logits is None or frame_states is None:
            assert relations_dict is not None # train mode
            assert frames.ndim == 5, f"got shape {frames.shape}"  # (B, T, C, H, W)
            frame_logits, frame_states = self.compute_features(frames)
        else:
            assert frame_logits.ndim == 5, f"got shape {frame_logits.shape}"  # (B, T, C, H, W)
            assert frame_states.ndim == 5, f"got shape {frame_states.shape}"  # (B, T, C, H, W)
            # nothing to do here
        B = frame_logits.shape[0]

        object_features_dict = self.build_object_features(frame_logits, frame_states)
        
        #sort by keys to have consistent ordering
        object_indices_list = [sorted(object_features_dict[b]) for b in range(B)]
        object_features_list = [torch.stack([object_features_dict[b][k] for k in object_indices_list[b]]) for b in range(B)]


        pair_proposals = []
        pair_proposals_original = []
        num_pairs = [0] * B
        ground_truth_flat = []
        for b in range(B):
            pair_proposals.append([])
            pair_proposals_original.append([])
            for tool, tissue in self.dataset_class.SCENE_GRAPH_DATASET.POSSIBLE_PAIRS:
                if tool not in object_features_dict[b] or tissue not in object_features_dict[b]:
                    continue
                reorded_tool_idx = object_indices_list[b].index(tool)
                reorded_tissue_idx = object_indices_list[b].index(tissue)
                if relations_dict is not None: 
                    if (tool, tissue) in relations_dict[b]:
                        gt = relations_dict[b][(tool, tissue)]
                    else:
                        gt = torch.zeros(len(self.dataset_class.SCENE_GRAPH_DATASET.VERBS)-1, dtype=torch.float32)
                    ground_truth_flat.append(gt)
                pair_proposals[-1].append((reorded_tool_idx, reorded_tissue_idx))
                pair_proposals_original[-1].append((tool, tissue))
                num_pairs[b] += 1



        return_dict = {
            "frame_logits": frame_logits,
            "pair_proposals": pair_proposals, # use these to build valid pairs
            "pair_proposals_original": [torch.tensor(pair_proposals_original[b], device=self.device) for b in range(B)], # use these to extract ground truths / verbs
            "num_pairs": num_pairs,
        }
        if sum(num_pairs) == 0:
            # no valid pairs in this batch
            return return_dict
        

        ground_truth_flat = torch.stack(ground_truth_flat).to(self.device) if relations_dict is not None else None

        relation_logits = self.relation_classifier(object_features_list, pair_proposals) # (sum(num_pairs), num_verbs)

        return_dict["relation_logits"] = relation_logits
        return_dict["ground_truth_verb"] = ground_truth_flat

        return return_dict

    @torch.no_grad()
    def _online_eval(self, frames: torch.Tensor, relations_dict: dict, prediction: dict[str, torch.Tensor]):
        self.sg_evaluator.record_batch(prediction, relations_dict)

    
    def pair_wise_ranking_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        margin = 1.0
        topk_neg = 2 * self._infer_K()  # hard negative mining

        ranking_losses = []

        # convert logits to scores (no sigmoid needed)
        scores = logits.detach()  # detach optional for stability

        for c in range(logits.shape[1]):  # loop over predicate classes
            class_scores = logits[:, c]
            class_targets = targets[:, c]

            pos_mask = class_targets > 0
            neg_mask = class_targets == 0

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            pos_scores = class_scores[pos_mask]
            neg_scores = class_scores[neg_mask]

            # Hard negative mining: select top-K highest scoring negatives
            k = min(topk_neg, neg_scores.numel())
            topk_neg_scores, _ = torch.topk(neg_scores, k)

            # Compute pairwise loss
            # Broadcast: (P, 1) vs (1, K)
            diff = margin - pos_scores.unsqueeze(1) + topk_neg_scores.unsqueeze(0)
            ranking_loss = torch.clamp(diff, min=0)

            ranking_losses.append(ranking_loss.mean())

        if len(ranking_losses) > 0:
            ranking_loss = torch.stack(ranking_losses).mean()
        else:
            ranking_loss = torch.tensor(0.0, device=logits.device)

        return ranking_loss


    def _train_step(self, frames: torch.Tensor, relations_dict: dict, online_eval: bool = False):
        prediction = self.predict_relations(frames, relations_dict=relations_dict)
        loss = None
        if "relation_logits" in prediction:
            loss = F.binary_cross_entropy_with_logits(prediction["relation_logits"], prediction["ground_truth_verb"], reduction='none')
            if self.relation_config["training"]["class_weighted_bce"]:
                class_counts = (prediction["ground_truth_verb"] > 0).sum(dim=0)
                loss_per_class = loss.sum(dim=0) / class_counts.clamp(min=1)
                loss = loss_per_class.mean()
            else:
                loss = loss.mean()
            
            if self.relation_config["training"]["pairwise_ranking_loss"]:
                lambda_rank = 0.5
                loss = loss + lambda_rank * self.pair_wise_ranking_loss(prediction["relation_logits"], prediction["ground_truth_verb"])


        if online_eval:
            if "relation_logits" not in prediction:
                num_verbs = len(self.dataset_class.SCENE_GRAPH_DATASET.VERBS)-1
                prediction["relation_logits"] = torch.zeros((0, num_verbs), device=self.device)
            self._online_eval(frames, relations_dict, prediction)

        return loss

    def print_relation_predictor(self):
        if not hasattr(self, "relation_classifier") or self.relation_classifier is None:
            self._init_relation_classifier()
        print(self.relation_classifier)
        print(f"Number parameters: {sum(p.numel() for p in self.relation_classifier.parameters()):,}")

    def create_relation_optimizer(self):
        optimizer = torch.optim.Adam(self.relation_classifier.parameters(), lr=self.initial_lr)
        return optimizer

    def train_relation_predictor(self, dryrun):
        self.network.to(self.device)
        self.network.eval()
        if not hasattr(self, "relation_classifier") or self.relation_classifier is None:
            self._init_relation_classifier()
        self.relation_classifier.to(self.device)
        self.init_wandb(dryrun, relation_classification=True)

        print('-' * 10, f"Start training {self.relation_classifier.__class__.__name__}", '-' * 10)

        optimizer = self.create_relation_optimizer()
        for epoch in range(3 if dryrun else 3):
            self.relation_classifier.train()

            d = self.dataset_class.SCENE_GRAPH_DATASET(split="train", num_frames=self.relation_config["num_frames"], time_span=self.relation_config["time_span"])
            if self.relation_config["training"]["balanced_sampling"]:
                loader = torch.utils.data.DataLoader(d, batch_size=4, num_workers=4, collate_fn=self._collate_fn,
                                                    sampler=BalancedSampler(d))
            else:
                loader = torch.utils.data.DataLoader(d, batch_size=4, num_workers=4, collate_fn=self._collate_fn,
                                                     shuffle=True)
            loading_bar = tqdm.tqdm(loader, desc=f"Epoch {epoch+1}/{3}")
            losses = []
            for i, data in enumerate(loading_bar):
                optimizer.zero_grad()
                loss = self._train_step(data["frames"], data["relations_dict"])
                if loss is None:
                    continue
                loss.backward()
                optimizer.step()
                loading_bar.set_postfix(loss=f"{loss.item():.4f}")
                losses.append(loss.item())
                if i % 1000 == 999:
                    wandb.log({
                        "train/relation_loss": np.mean(losses),
                        "train/lr": optimizer.param_groups[0]['lr'],
                    }, step=epoch * len(loader) + i)
                    losses.clear()
                if dryrun and i > 100:
                    break
            if epoch % 5 == 4:
                self._eval_relation_predictor_internal(split="val")
                relation_results = self.sg_evaluator.get_results_dict()
                self.sg_evaluator.print_results()
                wandb_results = {
                    "val/Recall@K": relation_results["Recall@K"],
                    "val/mRecall@K": relation_results["mRecall@K"],
                    "val/mAP@K": relation_results["mAP@K"],
                }
                for metric in ["recall_per_verb", "ap@K_per_verb"]:
                    for verb, value in relation_results[metric].items():
                        wandb_results[f"val/{metric}_{verb}"] = value
                wandb.log(wandb_results, step=(epoch+1) * len(loader))
                self.sg_evaluator.reset()

        wandb.unwatch(self.relation_classifier)
        wandb.finish()
        self.is_relation_trained = True
        
    @torch.no_grad()
    def eval_relation_predictor(self, split: str):
        assert self.is_relation_trained, "Relation classifier is not trained yet."
        self._eval_relation_predictor_internal(split)

        self.sg_evaluator.save_results(os.path.join(self.get_save_path(), f"scene_graph_evaluation_{split}.json"))
        self.sg_evaluator.print_results()
        r = self.sg_evaluator.get_results_dict()
        self.sg_evaluator.reset()
        return r

    def _infer_K(self):
        return {
            CholecDataset: 4,
            CataractsDataset: 6,
        }[self.dataset_class]

    @torch.no_grad()
    def _eval_relation_predictor_internal(self, split: str):
        self.network.to(self.device)
        self.network.eval()
        self.relation_classifier.to(self.device)
        self.relation_classifier.eval()

        d = self.dataset_class.SCENE_GRAPH_DATASET(split=split, num_frames=self.relation_config["num_frames"], time_span=self.relation_config["time_span"])
        loader = torch.utils.data.DataLoader(d, batch_size=4, shuffle=False, num_workers=4, collate_fn=self._collate_fn)


        num_verbs = len(d.VERBS)-1
        self.sg_evaluator = SceneGraphEvaluator(num_verbs=num_verbs, K=self._infer_K(), 
                                                device=self.device, idx_to_verb=d.VERBS.inverse)

        loading_bar = tqdm.tqdm(loader, desc=f"Evaluating on {split} set")
        for i, data in enumerate(loading_bar):
            self._train_step(data["frames"], data["relations_dict"], online_eval=True)

    
class UNetTrainer(RelationClassifierTrainer):
    def __init__(self, dataset_class, architecture: str= "unet", train_w_softmax=True, classes: List[int]=None):
        super().__init__(dataset_class)
        if classes is None:
            num_classes = dataset_class.NUM_CLASSES if not train_w_softmax else dataset_class.NUM_CLASSES + 1
        else:
            num_classes = len(classes) if not train_w_softmax else len(classes) + 1

        self.architecture = architecture
        if architecture == "unet":
            network = SurgicalUNet(in_channels=3, out_classes=num_classes, padding="same").to(self.device)
        elif architecture == "segformer":
            network = SurgicalSegFormer(num_labels=num_classes).to(self.device)
        elif architecture == "swinunet":
            network = SurgicalSwinUNet(img_size=256, window_size=8, num_classes=num_classes).to(self.device)
        elif architecture == "mask2former":
            network = SurgicalMask2Former(num_labels=num_classes).to(self.device)
        elif architecture == "tiny_unet":
            network = TinySurgicalUNet(in_channels=3, out_classes=num_classes).to(self.device)
        else:
            raise ValueError(f"Unknown architecture {architecture}")
    
        self.network = network
        self.train_w_softmax = train_w_softmax
        self.classes = classes

    def get_config(self):
        return super().get_config() | {"train_w_softmax": self.train_w_softmax, "classes": self.classes, "architecture": self.architecture}

    def parse_config(self, config):
        super().parse_config(config)
        self.train_w_softmax = config["train_w_softmax"]
        self.classes = config["classes"]
        self.architecture = config.get("architecture", "unet")

    def train(self, dryrun):
        self.init_wandb(dryrun)
        training.train(
            max_num_epochs=3 if dryrun else self.max_num_epochs,
            network=self.network,
            classes=self.classes,
            dataset_class=self.dataset_class,
            batches_per_epoch=self.batches_per_epoch,
            initial_lr=self.initial_lr,
            dryrun=dryrun,
            train_w_softmax=self.train_w_softmax,
            best_dice_callback=self.best_dice_callback,
        )

    def eval(self, split: str, file_name="dices"):
        dataset = self.dataset_class(
            split, classes=self.classes, original_only=True, 
            return_for_softmax=self.train_w_softmax)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
        dices = eval_framewise(self.network, dataloader, softmax=self.train_w_softmax)
        if self.train_w_softmax:
            dices = dices[:, 1:]  # exclude background class


        print((dices.nanmean(dim=0) * 100).int())
        path = self.get_save_path()
        torch.save(dices, os.path.join(path, f"{file_name}_{split}.pth"))



        df = pd.DataFrame(dices.cpu().numpy(), 
                          columns=[
                              self.dataset_class.SCENE_GRAPH_DATASET.SEG_LABELS.inverse[i] + f" ({i})" for i in self.classes
                              ])
        df = df[[self.dataset_class.SCENE_GRAPH_DATASET.SEG_LABELS.inverse[i] + f" ({i})" for i in sorted(self.classes)]]
        df.to_csv(os.path.join(path, f"{file_name}_{split}.csv"), index=False)
        print(f"Saved dice scores to {path}")

    @staticmethod
    def load_checkpoint_internal(path: str, configs: dict, dataset_class, load_best):
        train_w_softmax = configs["train_w_softmax"]
        classes = configs["classes"]
        architecture = configs.get("architecture", "unet")
        model_file = "model_best.pth" if load_best else "model.pth"
        trainer = UNetTrainer(dataset_class, architecture=architecture, train_w_softmax=train_w_softmax, classes=classes)
        trainer.network.load_state_dict(torch.load(os.path.join(path, model_file),
                                      map_location='cpu', weights_only=True))
        trainer.network.to(trainer.device)
        return trainer
    

    def build_object_features(self, frame_logits: torch.Tensor, frame_states: torch.Tensor) -> List[dict]:
        T = frame_logits.shape[1]
        B = frame_logits.shape[0]

        object_features = [{} for _ in range(B)]


        for b in range(B):
            prob_map = F.softmax(frame_logits[b], dim=1) # (T, C, H, W)
            index_map = prob_map.argmax(dim=1)  # (T, H, W)
            for cls in self.dataset_class.SCENE_GRAPH_DATASET.SEG_LABELS.inverse.keys():
                hard_map = index_map == cls # (T, H, W)
                if self.object_present(hard_map[-1]):
                    object_features[b][cls] = self.extract_features(frame_states[b], hard_map) # (T C)
        return object_features

    def get_index_map(self, frame_logits):
        assert frame_logits.dim() == 3  # (C, H, W)
        prob_map = F.softmax(frame_logits, dim=0)
        index_map = prob_map.argmax(dim=0)  # (H, W)
        return index_map

    def get_hard_map(self, frame_logits, cls):
        index_map = self.get_index_map(frame_logits)
        hard_map = index_map == cls
        return hard_map


class IncrementalNCATrainer(RelationClassifierTrainer):
    def __init__(self, dataset_class):
        super().__init__(dataset_class)
        self.parents_ordered = []
        self.classes = []
        self.incremental_nca_params ={
            "fire_rate": 1.0,
            "num_channels": 8,
            "hidden_size": 32,
            "num_epochs_per_increment": 40,
            "num_batches_per_epoch_per_increment": 2_500,
        }
        

    def get_config(self):
        config = super().get_config()
        local_config = {
            "parents": self.parents_ordered,
            "classes": self.classes,
            "incremental_nca_params": self.incremental_nca_params,
        }
        return config | local_config
    
    def parse_config(self, config):
        super().parse_config(config)
        self.parents_ordered = config["parents"]
        self.classes = config["classes"]
        self.incremental_nca_params = config.get("incremental_nca_params", self.incremental_nca_params)

    def increment_nca(self, num_new_classes: int):
        num_channels = self.incremental_nca_params["num_channels"]
        hidden_size = self.incremental_nca_params["hidden_size"]
        if not hasattr(self, "network") or self.network is None:
            # first NCA
            self.network = get_cholec_model(num_new_classes, fire_rate=self.incremental_nca_params["fire_rate"]).to(self.device)
        elif isinstance(self.network, LargeOctreeNCA2D):
            self.network.add_new_nca(num_channels=num_channels, num_new_classes=num_new_classes, hidden_size=hidden_size)
        else:
            self.network = LargeOctreeNCA2D(self.network, num_channels=num_channels, num_new_classes = num_new_classes, hidden_size=hidden_size)

    def train(self, new_classes: list[int], dryrun: bool):
        self.increment_nca(len(new_classes))
        self.init_wandb(dryrun)
        self.parents_ordered.append(self.run_name)
        self.classes.append(new_classes)

        num_epochs = self.max_num_epochs
        num_batches = self.batches_per_epoch
        if isinstance(self.network, LargeOctreeNCA2D):
            num_epochs = self.incremental_nca_params["num_epochs_per_increment"]
            num_batches = self.incremental_nca_params["num_batches_per_epoch_per_increment"]

        training.train(
            max_num_epochs=3 if dryrun else num_epochs,
            network=self.network,
            classes=new_classes,
            dataset_class=self.dataset_class,
            batches_per_epoch=num_batches,
            initial_lr=self.initial_lr,
            dryrun=dryrun,
            train_w_softmax=False,
            best_dice_callback=self.best_dice_callback,
        )
        wandb.unwatch(self.network)
        wandb.finish()

    def eval(self, split: str, file_name="dices"):
        if isinstance(self.network, LargeOctreeNCA2D):
            return_all_logits_backup = self.network.return_all_logits
            self.network.return_all_logits = True
        dataset = self.dataset_class(
            split, classes=sum(self.classes, []), original_only=True, 
            return_for_softmax=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
        dices = eval_framewise(self.network, dataloader, softmax=False)
        print((dices.nanmean(dim=0) * 100).int())
        path = self.get_save_path()
        torch.save(dices, os.path.join(path, f"{file_name}_{split}.pth"))
        df = pd.DataFrame(dices.cpu().numpy(), 
                          columns=[
                              self.dataset_class.SCENE_GRAPH_DATASET.SEG_LABELS.inverse[i] + f" ({i})" for i in sum(self.classes, [])
                              ])
        df = df[[self.dataset_class.SCENE_GRAPH_DATASET.SEG_LABELS.inverse[i] + f" ({i})" for i in sorted(sum(self.classes, []))]]
        df.to_csv(os.path.join(path, f"{file_name}_{split}.csv"), index=False)
        print(f"Saved dice scores to {path}")
        if isinstance(self.network, LargeOctreeNCA2D):
            self.network.return_all_logits = return_all_logits_backup
    
    @staticmethod
    def load_checkpoint_internal(path: str, configs: dict, dataset_class, load_best):
        trainer = IncrementalNCATrainer(dataset_class)
        classes = configs["classes"]
        num_channels = configs.get("incremental_nca_params", {}).get("num_channels", 8)
        hidden_size = configs.get("incremental_nca_params", {}).get("hidden_size", 32)
        fire_rate = configs.get("incremental_nca_params", {}).get("fire_rate", 1.0)
        model_file = "model_best.pth" if load_best else "model.pth"
        if len(classes) > 1:
            trainer.network = LargeOctreeNCA2D.load_from_file(path, model_file, classes=classes, num_channels=num_channels, hidden_size=hidden_size, fire_rate=fire_rate)
        else:
            trainer.network = get_cholec_model(len(classes[0]), fire_rate=fire_rate)
            trainer.network.load_state_dict(torch.load(os.path.join(path, model_file), 
                                        map_location='cpu', weights_only=True))
        trainer.network.to(trainer.device)
        return trainer
    
    def get_seg_channel_of_class(self, cls: int):
        if not hasattr(self, "classes_consecutive"):
            self.classes_consecutive = sum(self.classes, [])
        return self.classes_consecutive.index(cls) 
    
    
    def get_eval_df(self, split: str):
        raise DeprecationWarning("Use eval method instead to save dice scores as CSV.")
        path = self.get_save_path()
        assert os.path.exists(os.path.join(path, f"dices_{split}.pth")), f"No dice scores found at {path} for split {split}"
        dices = torch.load(os.path.join(path, f"dices_{split}.pth"), map_location='cpu', weights_only=True)
        return pd.DataFrame(dices, columns = sum(self.classes, []))
    
    
    def build_object_features(self, frame_logits: torch.Tensor, frame_states: torch.Tensor) -> List[dict]:
        T = frame_logits.shape[1]
        B = frame_logits.shape[0]

        object_features = [{} for _ in range(B)]

        for b in range(B):
            for cls in self.dataset_class.SCENE_GRAPH_DATASET.SEG_LABELS.inverse.keys():
                hard_map = self.get_hard_map(frame_logits[b], cls) # (T, H, W)
                if self.object_present(hard_map[-1]):
                    object_features[b][cls] = self.extract_features(frame_states[b], hard_map) # (T C)
        return object_features

    def get_hard_map(self, frame_logits: torch.Tensor, cls: int) -> torch.Tensor:
        assert frame_logits.dim() == 4 # (T, C, H, W)
        channel_idx = self.get_seg_channel_of_class(cls)
        prob_map = F.sigmoid(frame_logits[:, channel_idx])
        hard_map = prob_map > 0.5
        return hard_map # (T, H, W)
    
    def get_index_map(self, frame_logits):
        assert frame_logits.dim() == 3  # (C, H, W)
        frame_probs = torch.sigmoid(frame_logits)
        segmentation = [frame_probs.max(dim=0).values <= 0.5] # initialize with background
        for label, label_index in self.dataset_class.SCENE_GRAPH_DATASET.SEG_LABELS.items():
            segmentation.append(frame_probs[self.get_seg_channel_of_class(label_index)])

        segmentation = torch.stack(segmentation, dim=0) # (C, H, W)
        return segmentation.argmax(dim=0)  # (H, W)
    
    def create_relation_optimizer(self):
        params = self.relation_classifier.parameters()
        if self.relation_config["feature_extraction"]["expand_nca"]:
            params = itertools.chain(params, filter(lambda p: p.requires_grad==True, self.network.parameters()))
        optimizer = torch.optim.Adam(params, lr=self.initial_lr)
        return optimizer

    def _init_relation_classifier(self):
        if self.relation_config["feature_extraction"]["expand_nca"]:
            self.network.add_new_nca(4, 0, 16)
        super()._init_relation_classifier()

    def compute_features(self, frames: torch.Tensor):
        T = frames.shape[1]
        B = frames.shape[0]
        frames = einops.rearrange(frames, "B T C H W -> (B T) C H W").to(self.device)
        
        use_grad = self.relation_config["feature_extraction"]["expand_nca"]
        with torch.set_grad_enabled(use_grad):
            network_output = self.network.compute_features(frames, return_states_all_levels=self.relation_config["feature_extraction"].get("from_all_levels", False))
            frame_logits = network_output['logits']
            frame_states = network_output['features']

        frame_logits = einops.rearrange(frame_logits, "(B T) C H W -> B T C H W", B=B)
        frame_states = einops.rearrange(frame_states, "(B T) C H W -> B T C H W", B=B)
        return frame_logits, frame_states

    @staticmethod
    def load_checkpoint(path, load_best=False) -> 'IncrementalNCATrainer':
        trainer = BaseTrainer.load_checkpoint(path, load_best=load_best)
        assert isinstance(trainer, IncrementalNCATrainer), f"Loaded trainer is not of type IncrementalNCATrainer but {trainer.__class__}"
        return trainer
