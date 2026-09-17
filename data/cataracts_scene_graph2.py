import json
from bidict import bidict
from numpy import indices
import torch
from utils.paths import path_data_cataracts_split, path_data_cat_sg, path_data_cataracts_full_videos, path_data_cataracts_full_videos_preprocessed
import os
from PIL import Image
import torchvision.transforms as T
import imageio
from data.util import temporal_indices
import numpy as np

class CataractSceneGraphDataset(torch.utils.data.Dataset):
    SEG_LABELS = bidict({
        "Pupil": 1,
        "Surgical Tape": 2,
        "Hand": 3,
        "Eye Retractors": 4,
        "Iris": 5,
        "Skin": 6,
        "Cornea": 7,
        "Cannula": 8,
        "Cap. Cystotome": 9,
        "Tissue Forceps": 10,
        "Primary Knife": 11,
        "Ph. Handpiece": 12,
        "Lens Injector": 13,
        "I/A Handpiece": 14,
        "Secondary Knife": 15,
        "Micromanipulator": 16,
        "Cap. Forceps": 17,
    })
    SUBJECT_CLASSES = [3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    OBJECT_CLASSES = [1, 2, 4, 5, 6, 7, 8, 13, 14, 16]
    VERBS = bidict({
        'Holding': 0,
        'Activation': 1,
        'Pushing': 2,
        'Pulling': 3,
        'Cutting': 4,
        'Inserting': 5,
        'Retracting': 6,
        'null_verb': 7
        })

    SCENE_TO_SEG = {
        "Pupil": "Pupil",
        "Surgical Tape": "Surgical Tape",
        "Hand": "Hand",
        "Eye Retractors": "Eye Retractors",
        "Iris": "Iris",
        "Skin": "Skin",
        "Cornea": "Cornea",

        "Hydro. Cannula": "Cannula",
        "Visc. Cannula": "Cannula",
        "Rycroft Cannula": "Cannula",
        "Charleux Cannula": "Cannula",

        "Cap. Cystotome": "Cap. Cystotome",

        "Bonn Forceps": "Tissue Forceps",
        "Troutman Forceps": "Tissue Forceps",

        "Primary Knife": "Primary Knife",
        "Ph. Handpiece": "Ph. Handpiece",
        "Lens Injector": "Lens Injector",
        "I/A Handpiece": "I/A Handpiece",
        "Secondary Knife": "Secondary Knife",
        "Micromanipulator": "Micromanipulator",
        "Cap. Forceps": "Cap. Forceps",
    }
    FPS = 30
    POSSIBLE_PAIRS = {(15, 1): torch.tensor([1, 1, 0, 0, 1, 0, 0], dtype=torch.int32),
        (10, 1): torch.tensor([1, 1, 1, 1, 1, 0, 0], dtype=torch.int32),
        (15, 7): torch.tensor([0, 1, 0, 0, 1, 1, 1], dtype=torch.int32),
        (10, 7): torch.tensor([1, 0, 1, 1, 0, 1, 1], dtype=torch.int32),
        (11, 7): torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.int32),
        (11, 1): torch.tensor([1, 1, 0, 0, 1, 0, 0], dtype=torch.int32),
        (8, 7): torch.tensor([1, 1, 0, 1, 0, 1, 1], dtype=torch.int32),
        (8, 1): torch.tensor([1, 1, 1, 1, 0, 0, 0], dtype=torch.int32),
        (9, 7): torch.tensor([0, 0, 0, 0, 0, 1, 1], dtype=torch.int32),
        (9, 1): torch.tensor([1, 1, 1, 1, 1, 0, 0], dtype=torch.int32),
        (17, 7): torch.tensor([1, 1, 0, 1, 0, 1, 1], dtype=torch.int32),
        (17, 1): torch.tensor([1, 1, 1, 1, 1, 0, 0], dtype=torch.int32),
        (12, 7): torch.tensor([1, 1, 1, 1, 0, 1, 1], dtype=torch.int32),
        (12, 5): torch.tensor([0, 1, 1, 0, 0, 0, 0], dtype=torch.int32),
        (12, 1): torch.tensor([1, 1, 1, 1, 0, 0, 0], dtype=torch.int32),
        (16, 7): torch.tensor([1, 1, 0, 1, 0, 1, 1], dtype=torch.int32),
        (16, 1): torch.tensor([1, 1, 1, 1, 1, 0, 0], dtype=torch.int32),
        (16, 5): torch.tensor([1, 0, 1, 1, 0, 0, 0], dtype=torch.int32),
        (3, 16): torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
        (14, 1): torch.tensor([1, 1, 1, 1, 1, 0, 0], dtype=torch.int32),
        (14, 7): torch.tensor([1, 1, 1, 1, 1, 1, 1], dtype=torch.int32),
        (10, 5): torch.tensor([1, 0, 0, 1, 0, 0, 0], dtype=torch.int32),
        (3, 13): torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
        (3, 6): torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
        (8, 5): torch.tensor([0, 1, 1, 0, 0, 0, 0], dtype=torch.int32),
        (13, 7): torch.tensor([1, 1, 0, 1, 0, 1, 1], dtype=torch.int32),
        (13, 1): torch.tensor([1, 1, 1, 0, 0, 0, 0], dtype=torch.int32),
        (14, 5): torch.tensor([1, 1, 1, 1, 0, 0, 0], dtype=torch.int32),
        (3, 14): torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
        (3, 8): torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
        (3, 2): torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
        (3, 4): torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
        (4, 7): torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.int32),
        (11, 5): torch.tensor([0, 0, 0, 0, 1, 0, 0], dtype=torch.int32),
        (17, 13): torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
        (10, 6): torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
        (11, 6): torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
        (3, 7): torch.tensor([1, 1, 0, 0, 0, 1, 0], dtype=torch.int32),
        (10, 13): torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
        (13, 6): torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
        (13, 5): torch.tensor([0, 1, 0, 0, 0, 0, 0], dtype=torch.int32),
        (17, 5): torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.int32),
        (9, 5): torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.int32),
        (14, 2): torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.int32)}
    
    def __init__(self, split: str, num_frames: int=5, time_span: float=1.0):
        super().__init__()
        self.num_frames = num_frames
        self.time_span = time_span
        self.delta_t = temporal_indices(0, self.FPS, self.num_frames, self.time_span)
        self.init_from_split(split)
        print(f"Loading those frame indices {self.delta_t} for each sample")
        assert torch.cuda.is_available()
        CataractSceneGraphDataset.POSSIBLE_PAIRS = {k: v.cuda().bool() for k, v in self.POSSIBLE_PAIRS.items()}

    def init_from_split(self, split):
        splits_file = os.path.join(path_data_cataracts_split, f"cataracts_{split}.txt")
        with open(splits_file, "r") as f:
            split_videos = f.read().splitlines()
        self.videos = []
        for v in split_videos:
            v = int(v)
            if v < 26:
                self.videos.append(f"train{v:02d}")
            else:
                self.videos.append(f"test{v - 25:02d}")



        self.video_frame_tuples = []

        for video in self.videos:
            scene_graph_dict = json.load(open(os.path.join(path_data_cat_sg, "all", f"{video}.json")))
            scene_graph_dict = {k: v for k, v in scene_graph_dict.items() if len(v["semantic_relations"]) > 0}
            for frame_idx, frame_data in scene_graph_dict.items():
                frame_idx = int(frame_idx)
                if frame_idx < np.abs(self.delta_t.min()):
                    continue
                
                relations = frame_data["semantic_relations"]
                relations = list(set([tuple(t) for t in relations]))  # remove duplicate triplets (I assume they are annotations errors)
                relations = [self.map_triplet(triplet) for triplet in relations]
                relations = [r for r in relations if r is not None]
                if len(relations) == 0:
                    continue
                frame_tuple = (video, frame_idx, relations)
                # TODO preprocess relations to indices of verbs and entities

                self.video_frame_tuples.append(frame_tuple)

    def map_triplet(self, triplet):
        subject, verb, object_ = triplet

        verb_idx = self.VERBS[verb]
        try:
            subject = self.SCENE_TO_SEG[subject]
            object_ = self.SCENE_TO_SEG[object_]
        except KeyError:
            return None
        subject_idx = self.SEG_LABELS[subject]
        object_idx = self.SEG_LABELS[object_]
        return (subject_idx, verb_idx, object_idx)

    
    def get_occurence_list(self):
        occurences = [[] for _ in self.VERBS]
        for i, (_,_, triplets) in enumerate(self.video_frame_tuples):
            for subject, verb, object_ in triplets:
                occurences[verb].append(i)
        return occurences

    def __len__(self):
        return len(self.video_frame_tuples)
    
    def load_frame_from_video(self, video: str, frame_idx: int) -> Image.Image:
        raise NotImplementedError("should not be used")
        video_path = os.path.join(path_data_cataracts_full_videos, "micro", f"{video}.mp4")
        vid = imageio.get_reader(video_path,  'ffmpeg')
        frame = vid.get_data(frame_idx)
        vid.close()
        frame = Image.fromarray(frame).resize((256,256))
        return frame
        

    def load_frame(self, video: str, frame_idx: int) -> torch.Tensor:
        preprocessed_video_path = os.path.join(path_data_cataracts_full_videos_preprocessed, video, f"{frame_idx:08d}.png")
        if os.path.exists(preprocessed_video_path):
            frame = Image.open(preprocessed_video_path).convert("RGB")
        else:
            frame = self.load_frame_from_video(video, frame_idx)
            os.makedirs(os.path.dirname(preprocessed_video_path), exist_ok=True)
            frame.save(preprocessed_video_path)
        frame = T.ToTensor()(frame)
        return frame

    def build_relations_dict(self, relations: list[tuple]) -> dict:
        relations_dict = {}
        for subject, verb, object_ in relations:
            target_multiclass = relations_dict.get((subject, object_), torch.zeros(7)) # 7 verbs (excluding the null_verb)
            target_multiclass[verb] = 1
            relations_dict[(subject, object_)] = target_multiclass
        return relations_dict

    def __getitem__(self, idx):
        video, frame_idx, relations = self.video_frame_tuples[idx]
        frames = torch.stack([self.load_frame(video, frame_idx) for frame_idx in self.delta_t + frame_idx], dim=0)  # (T, C, H, W)
        relations_dict = self.build_relations_dict(relations)
        return {
            "id": f"{video}_{frame_idx}",
            "frames": frames,
            "relations_dict": relations_dict
        }