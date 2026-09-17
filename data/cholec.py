
from typing import Callable, List, Optional, Tuple
import os
import re
from glob import glob

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from tqdm import tqdm

from utils.paths import path_data_cholec, path_data_cholec_meta, path_data_cholec_original, path_data_cholec_split, path_data_cholec_original_masks
import numpy as np
import torch.nn.functional as F
import pickle as pkl
from data.cholec_scene_graph import CholecSceneGraphDataset

class CholecDataset(Dataset):
    NUM_CLASSES = 12
    IGNORE_INDEX = 255

    SCENE_GRAPH_DATASET = CholecSceneGraphDataset
    def __init__(self, split: str, classes: Optional[List[int]]=None, original_only: bool=False, return_for_softmax: bool=False, filter_cases: bool = True):
        self.original_only = original_only
        self.return_for_softmax = return_for_softmax

        if return_for_softmax and classes is not None:
            assert not 0 in classes, "If return_for_softmax is True, a novel background class (0) will be created and added automatically, do not include it in classes"

        self.init_from_split(split)
        self.create_or_load_occurence_dict()

        if classes is not None:
            if filter_cases:
                new_video_frame_paths = []
                for video in tqdm(self.video_frame_paths, desc="Filtering classes"):
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
                if _class in [self.IGNORE_INDEX+1, 12, 11 ]:
                    continue
                #print(_class)
                occurences[_class-1].append(i)
        return occurences

    def get_cases_for_split(self, split: str) -> List[str]:
        splits_file = os.path.join(path_data_cholec_split, f"cholec80_{split}.txt")
        with open(splits_file, "r") as f:
            split_videos = f.read().splitlines()
        split_videos = [f"video{v}" for v in split_videos]
        split_videos = [v for v in split_videos if os.path.exists(os.path.join(path_data_cholec, "video_frames", v))]
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
                for start_frame in os.listdir(os.path.join(path_data_cholec_original, video)):
                    for actual_frame in os.listdir(os.path.join(path_data_cholec_original, video, start_frame)):
                        if not actual_frame.endswith("_endo.png"):
                            continue
                        frame_number = actual_frame.split("_")[1]
                        self.video_frame_paths.append((os.path.join(video, start_frame), frame_number))
        else:
            for video in self.videos:
                for frame in os.listdir(os.path.join(path_data_cholec, "video_frames", video)):
                    frame_path = os.path.join(video, frame[:-4])  # remove .jpg extension
                    self.video_frame_paths.append(frame_path)


    def create_or_load_occurence_dict(self):
        occurence_dict_path = os.path.join(path_data_cholec_meta, 
                                           "occurence_dict.pkl" if not self.original_only else
                                           "occurence_dict_original.pkl"
                                           )
        if os.path.exists(occurence_dict_path):
            with open(occurence_dict_path, "rb") as f:
                self.occurence_dict = pkl.load(f)
        else:
            os.makedirs(path_data_cholec_meta, exist_ok=True)
            occurence_dict = self.create_occurence_dict()
            with open(occurence_dict_path, "wb") as f:
                pkl.dump(occurence_dict, f)
            self.occurence_dict = occurence_dict

            

    def create_occurence_dict(self) -> dict:
        original_split = self.split
        if self.original_only:
            self.init_from_split(["val", "test"])
        else:
            self.init_from_split("all")  # use all data to create occurence dict
        occurence_dict = {}
        for path in tqdm(self.video_frame_paths, desc="Creating occurence dict"):
            if self.original_only:
                label = self.load_item_from_tuple(path, seg_only=True).numpy()
            else:
                label = self.load_item_from_path(path, seg_only=True).numpy()
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
            if self.class_idx is not None:
                new_label = torch.full_like(label, fill_value=0) #init with background
                for i, c in enumerate(self.class_idx):
                    new_label[label == c] = i +1  #+1 to account for background class at 0
                label = new_label
        else:
            label = F.one_hot(label, num_classes=self.NUM_CLASSES+1).permute(2,0,1).float()
            if self.class_idx is not None:
                label = label[self.class_idx]
        return label

    def load_item_from_tuple(self, path_tuple: Tuple[str,str], seg_only: bool=False):
        path, frame_number = path_tuple
        label = Image.open(os.path.join(path_data_cholec_original_masks, path, f"frame_{frame_number}_mask.png"))
        label = label.resize((256,256), resample=Image.NEAREST)
        label = torch.from_numpy(np.array(label)).long()
        if seg_only:
            return label
        
        
        label = self.process_label(label)

        img = Image.open(os.path.join(path_data_cholec_original, path, f"frame_{frame_number}_endo.png"))
        img = img.resize((256,256))
        img = T.ToTensor()(img)

        return img, label

    def load_item_from_path(self, path: str, seg_only: bool=False):
        label = Image.open(os.path.join(path_data_cholec, "segm_ann", path + ".png"))
        assert np.max(label) <= self.NUM_CLASSES+1, f"Label has values outside expected range [0,{self.NUM_CLASSES+1}], being {np.max(label)}"
        label = label.resize((256,256), resample=Image.NEAREST)
        label = torch.from_numpy(np.array(label)).long()
        if seg_only:
            return label
        
        label = self.process_label(label)


        img = Image.open(os.path.join(path_data_cholec, "video_frames", path + ".jpg"))
        img = img.resize((256,256))
        img = T.ToTensor()(img)
        return img, label

    def __getitem__(self, idx):
        return self.load_item_from_idx(idx, seg_only=False)