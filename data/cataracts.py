
from typing import Callable, List, Optional, Tuple
import os
import re
from glob import glob

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from tqdm import tqdm

from data.cataracts_scene_graph2 import CataractSceneGraphDataset
from utils.paths import path_data_cataracts, path_data_cataracts_meta, path_data_cataracts_original, path_data_cataracts_split
import numpy as np
import torch.nn.functional as F
import pickle as pkl
import data.cataracts_exps as cataracts_exps


label_lut, class_lut = cataracts_exps.build_lut(cataracts_exps.EXP2)
label_lut = torch.tensor(label_lut).long()

# list of classes: https://cataracts.grand-challenge.org/CaDIS/ (https://arxiv.org/pdf/1906.11586)
class CataractsDataset(Dataset):
    NUM_CLASSES = int(label_lut[label_lut != 255].max().item() + 1)
    IGNORE_INDEX = 255
    SCENE_GRAPH_DATASET = CataractSceneGraphDataset
    def __init__(self, split: str, original_only: bool=False, classes: Optional[List[int]]=None, return_for_softmax: bool=False, filter_cases = True):
        self.return_for_softmax = return_for_softmax
        self.original_only = original_only

        if return_for_softmax and classes is not None:
            assert not 0 in classes, "If return_for_softmax is True, a novel background class (0) will be created and added automatically, do not include it in classes"

        self.init_from_split(split)
        self.create_or_load_occurence_dict()

        if classes is not None:
            if filter_cases:
                new_video_frame_paths = []
                for video in self.video_frame_paths:
                    present_classes = self.occurence_dict[video]
                    if any([c in present_classes for c in classes]):
                        new_video_frame_paths.append(video)
                self.video_frame_paths = new_video_frame_paths

            self.class_idx = torch.tensor(classes).long()
        else:
            self.class_idx = None

    def get_occurence_list(self) -> List[List[int]]:
        labels = self.SCENE_GRAPH_DATASET.SEG_LABELS

        occurences = [[] for _ in range(max(labels.values()))]
        for i,video_frame in enumerate(self.video_frame_paths):
            for _class in self.occurence_dict[video_frame]:
                if _class == self.IGNORE_INDEX+1:
                    continue
                occurences[_class-1].append(i)
        return occurences

    def get_cases_for_split(self, split: str) -> List[str]:
        splits_file = os.path.join(path_data_cataracts_split, f"cataracts_{split}.txt")
        with open(splits_file, "r") as f:
            split_videos = f.read().splitlines()
        # TODO optionally filter by availability
        if self.original_only:
            split_videos = [v for v in split_videos if os.path.exists(os.path.join(path_data_cataracts_original, f"Video{v}"))]
        else:
            split_videos = [v for v in split_videos if os.path.exists(os.path.join(path_data_cataracts, "video_frames", f"train{v}"))]
        return split_videos

    def init_from_split(self, split: str | list[str]):
        self.split = split
        if split == "all":
            self.videos = self.get_cases_for_split("train") + \
            self.get_cases_for_split("val") + \
            self.get_cases_for_split("test")
        elif isinstance(split, list):
            self.videos = sum([self.get_cases_for_split(s) for s in split], [])
        else:
            self.videos = self.get_cases_for_split(split)
        
        self.video_frame_paths = []
        if self.original_only:
            for video in self.videos:
                for frame in os.listdir(os.path.join(path_data_cataracts_original, f"Video{video}", "Images")):
                    self.video_frame_paths.append((f"Video{video}", frame))
        else:
            for video in self.videos:
                for frame in os.listdir(os.path.join(path_data_cataracts, "video_frames", f"train{video}")):
                    frame_path = os.path.join(f"train{video}", frame[:-4])  # remove .jpg extension
                    self.video_frame_paths.append(frame_path)


    def create_or_load_occurence_dict(self):
        occurence_dict_path = os.path.join(path_data_cataracts_meta, 
                                           "occurence_dict.pkl" if not self.original_only else
                                           "occurence_dict_original.pkl"
                                           )
        if os.path.exists(occurence_dict_path):
            with open(occurence_dict_path, "rb") as f:
                self.occurence_dict = pkl.load(f)
        else:
            os.makedirs(path_data_cataracts_meta, exist_ok=True)
            occurence_dict = self.create_occurence_dict()
            with open(occurence_dict_path, "wb") as f:
                pkl.dump(occurence_dict, f)
            self.occurence_dict = occurence_dict

            

    def create_occurence_dict(self) -> dict:
        original_split = self.split
        self.init_from_split("all")  # use all data to create occurence dict
        occurence_dict = {}
        for path in tqdm(self.video_frame_paths, desc="Creating occurence dict"):
            if self.original_only:
                label = self.load_item_from_tuple(path, seg_only=True).numpy()
            else:
                label = self.load_item_from_path(path, seg_only=True).numpy()
            label += 1 # need to account for background class at 0
            unique = np.unique(label, return_counts=False)
            for u in unique:
                occurence_dict[path] = unique.tolist()
        self.init_from_split(original_split)  # restore original split
        return occurence_dict

    def __len__(self):
        return len(self.video_frame_paths)
    
    def load_item_from_idx(self, idx: int, seg_only: bool=False):
        if self.original_only:
            return self.load_item_from_tuple(self.video_frame_paths[idx], seg_only=seg_only)
        else:
            return self.load_item_from_path(self.video_frame_paths[idx], seg_only=seg_only)

    def process_label(self, label: torch.Tensor) -> torch.Tensor:
        if self.return_for_softmax:
            if self.class_idx is None:
                # build background mask
                label += 1  #shift all labels by one to make space for background at 0
                label[label == self.IGNORE_INDEX +1] = 0  #background
                raise NotImplementedError("dont use")
            else:
                new_label = torch.full_like(label, fill_value=0) #init with background
                for i, c in enumerate(self.class_idx):
                    new_label[label == c-1] = i + 1  #+1 to account for background class at 0
                label = new_label
        else:
            valid_mask = label != self.IGNORE_INDEX
            label_safe = label.clone() + 1 #shift all labels by one to make space for background at 0
            label_safe[~valid_mask] = 0
            label = F.one_hot(label_safe, num_classes=self.NUM_CLASSES+1).permute(2,0,1).float()
            
            if self.class_idx is None:
                raise NotImplementedError("dont use")
            else:
                label = label[self.class_idx]

        return label
    

    def process_img(self, img: Image.Image) -> torch.Tensor:
        img = img.resize((256,256))
        img = T.ToTensor()(img)
        return img

    def map_label(self, label: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "lut"):
            self.lut = label_lut.to(device=label.device, dtype=label.dtype)
        label = self.lut[label]

        return label

    def load_item_from_tuple(self, path_tuple: Tuple[str,str], seg_only: bool=False):
        path, frame = path_tuple
        label = Image.open(os.path.join(path_data_cataracts_original, path, "Labels", frame))
        label = label.resize((256,256), resample=Image.NEAREST)
        label = torch.from_numpy(np.array(label)).long()
        label = self.map_label(label)
        #assert torch.max(label) <= self.NUM_CLASSES+1, f"Label has values outside expected range [0,{self.NUM_CLASSES+1}], being {torch.max(label)}"
        if seg_only:
            return label
        
        label = self.process_label(label)


        img = Image.open(os.path.join(path_data_cataracts_original, path, "Images", frame))
        img = self.process_img(img)

        return img, label

    def load_item_from_path(self, path: str, seg_only: bool=False):
        label = Image.open(os.path.join(path_data_cataracts, "segm_ann", path + ".png"))
        label = label.resize((256,256), resample=Image.NEAREST)
        label = torch.from_numpy(np.array(label)).long()
        #label = self.map_label(label)
        #assert torch.max(label) <= self.NUM_CLASSES+1, f"Label has values outside expected range [0,{self.NUM_CLASSES+1}], being {torch.max(label)}"
        if seg_only:
            return label
        
        label = self.process_label(label)


        img = Image.open(os.path.join(path_data_cataracts, "video_frames", path + ".jpg"))
        img = self.process_img(img)
        return img, label

    def __getitem__(self, idx):
        return self.load_item_from_idx(idx, seg_only=False)