import json
from bidict import bidict
import numpy as np
import torch
from data.util import temporal_indices
from utils.paths import path_data_cholect50, path_data_cholec_split, path_data_cholec_full_videos_preprocessed
import os
from PIL import Image
import torchvision.transforms as T


class CholecSceneGraphDataset(torch.utils.data.Dataset):
    SEG_LABELS = bidict({
        "abdominal_wall": 1,
        "liver": 2,
        "gastrointestinal_tract": 3,
        "fat": 4,
        "grasper": 5,
        "connective_tissue": 6,
        "blood": 7,
        "cystic_duct": 8,
        "hook": 9,
        "gallbladder": 10,
    })
    SUBJECT_CLASSES = [5, 9]  # grasper, hook
    OBJECT_CLASSES = [1,2,3,4,6,7,8,10]  # all tissue classes
    VERBS = bidict({
        'grasp': 0,
        'retract': 1,
        'dissect': 2,
        'coagulate': 3,
        'clip': 4,
        'cut': 5,
        'aspirate': 6,
        'irrigate': 7,
        'pack': 8,
        'null_verb': 9
        })
    POSSIBLE_PAIRS = [(5, 4),
        (5, 2),
        (5, 3),
        (5, 10),
        (9, 4),
        (9, 10),
        (9, 8),
        (9, 7),
        (5, 7),
        (9, 6),
        (5, 8),
        (9, 2),
        (5, 6),
        (9, 1)]
    FPS = 25

    def __init__(self, split: str, num_frames: int=5, time_span: float=1.0):
        super().__init__()
        self.num_frames = num_frames
        self.time_span = time_span
        self.delta_t = temporal_indices(0, self.FPS, self.num_frames, self.time_span)
        self.init_from_split(split)
        print(f"Loading those frame indices {self.delta_t} for each sample")
        
    def get_scene_graph_json_path(self, video_id: str):
        return os.path.join(path_data_cholect50, "labels", f"VID{video_id}.json")
    
    def init_from_split(self, split):
        splits_file = os.path.join(path_data_cholec_split, f"cholec80_{split}.txt")
        with open(splits_file, "r") as f:
            split_videos = f.read().splitlines()
        self.videos = sorted([v for v in split_videos if os.path.exists(self.get_scene_graph_json_path(v))])
        self.videos = sorted([v for v in self.videos if int(v) <= 80]) # videos with index > 80 are not publicly available at >1 FPS (https://www.nature.com/articles/s41597-025-05163-w)



        self.video_frame_tuples = []

        for video in self.videos:
            scene_graph_path = self.get_scene_graph_json_path(video)
            if not os.path.exists(scene_graph_path):
                continue
            scene_graphs = json.load(open(scene_graph_path))
            scene_graphs = scene_graphs["annotations"]
            frames = os.listdir(os.path.join(path_data_cholect50, "videos", f"VID{video}"))
            frames = [int(frame[:-4]) for frame in frames]  # remove .png extension
            for frame_idx in sorted(frames):
                original_frame_idx = frame_idx * self.FPS

                if original_frame_idx < np.abs(self.delta_t.min()):
                    continue

                frame_tuple = (f"VID{video}", original_frame_idx)
                scene_graph = scene_graphs[str(frame_idx)] # use old index to index the scene graph annotations
                
                all_triplets = []
                for relation in scene_graph:
                    (triplet_id,
                    subject_id, sub_conf, sub_x, sub_y, sub_w, sub_h,
                    verb_id, target_id, tar_conf, tar_x, tar_y, tar_w, tar_h,
                    phase) = relation
                    assert sub_conf >= 1.0 and tar_conf >= 1.0, "Low confidence in scene graph!"
                    triplet = (subject_id, verb_id, target_id)
                    triplet = self.map_triplet(triplet)
                    if triplet is not None:
                        all_triplets.append(triplet)
                if len(all_triplets) == 0:
                    continue
                frame_tuple += (all_triplets,)
                self.video_frame_tuples.append(frame_tuple)


    def map_triplet(self, triplet):
        instr_id, verb_id, target_id = triplet


        # ---------- Instrument mapping ----------
        instrument_map = {
            0: "grasper",
            1: "hook",     # bipolar -> hook
            2: "hook",
            # scissors, clipper, irrigator -> drop
        }

        if instr_id not in instrument_map:
            return None

        instr_seg = self.SEG_LABELS[instrument_map[instr_id]]

        # ---------- Target mapping ----------
        target_map = {
            0: "gallbladder",
            1: "connective_tissue",   # cystic_plate
            2: "cystic_duct",
            3: "blood",               # cystic_artery
            4: "connective_tissue",   # cystic_pedicle
            5: "blood",               # blood_vessel
            7: "abdominal_wall",      # abdominal_wall_cavity
            8: "liver",
            9: "connective_tissue",   # adhesion
            10: "fat",                # omentum
            11: "connective_tissue",  # peritoneum
            12: "gastrointestinal_tract",
        }

        if target_id not in target_map:
            return None

        target_seg = self.SEG_LABELS[target_map[target_id]]

        # ---------- Null verb handling ----------
        if verb_id == 9:  # null_verb
            return None

        return (instr_seg, verb_id, target_seg)

    def get_occurence_list(self):
        occurences = [[] for _ in self.VERBS]
        for i, (_,_, triplets) in enumerate(self.video_frame_tuples):
            for subject, verb, object_ in triplets:
                occurences[verb].append(i)
        return occurences

    def get_str_triplet(self, triplet):
        instr_id, verb_id, target_id = triplet
        instr_name = self.SEG_LABELS.inverse[instr_id]
        target_name = self.SEG_LABELS.inverse[target_id]
        verb_name = self.VERBS.inverse[verb_id]
        return (instr_name, verb_name, target_name)

    def __len__(self):
        return len(self.video_frame_tuples)
    
    def build_relations_dict(self, relations: list[tuple]) -> dict:
        relations_dict = {}
        for subject, verb, object_ in relations:
            target_multiclass = relations_dict.get((subject, object_), torch.zeros(9)) # 9 verbs (excluding the null_verb)
            target_multiclass[verb] = 1
            relations_dict[(subject, object_)] = target_multiclass
        return relations_dict
    
    def __getitem__(self, idx):
        video_name, frame_idx, triplets = self.video_frame_tuples[idx]

        frames = []
        for t in self.delta_t + frame_idx:
            img = Image.open(
                os.path.join(path_data_cholec_full_videos_preprocessed, video_name, f"{t:08d}.png")
            ).convert("RGB")
            img = T.ToTensor()(img)
            frames.append(img)
            
        frames = torch.stack(frames, dim=0)  # (T, 3, H, W)
        relations_dict = self.build_relations_dict(triplets)

        return {
            "id": f"{video_name}_{frame_idx}",
            "frames": frames,
            "relations_dict": relations_dict
        }   