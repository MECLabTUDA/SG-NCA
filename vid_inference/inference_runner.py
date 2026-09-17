

import os

import cv2
import einops
import torch

from data.cataracts_scene_graph2 import CataractSceneGraphDataset
from data.util import temporal_indices
from evaluation.scene_graph_builder import SceneGraphBuilder
from models.LargeOctreeNCA2D import LargeOctreeNCA2D
from training_utils.trainer import RelationClassifierTrainer
import imageio
import numpy as np
from PIL import Image
from collections import OrderedDict, deque
import torchvision.transforms as T
from vid_inference.scenegraph_visualizer import SceneGraphVisualizer
import tqdm
import json

from vid_inference.temporal_smoother import TemporalSmoother


class InferenceRunner:
    def __init__(self, trainer: RelationClassifierTrainer, video_reader, output_dir):
        self.trainer = trainer
        self.video_reader = video_reader
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.prediction_dict = OrderedDict()

        self.area_threshold = self.trainer.relation_config["feature_extraction"]["area_threshold"] # for geometric relations

        self.render_graph = True
        self.full_sentence = True
        self.top_k_sampling = False # whether to use top-k sampling for relation extraction or thresholding
        self.threshold = 0.5
        self.k = self.trainer._infer_K()

        num_frames = self.trainer.relation_config["num_frames"]
        time_span = self.trainer.relation_config["time_span"]
        self.dataset_class = self.trainer.dataset_class.SCENE_GRAPH_DATASET
        self.delta_t = temporal_indices(0, self.dataset_class.FPS, num_frames, time_span)
        print(self.delta_t)
        
        cache_size = np.abs(self.delta_t.min())+1
        self.segmentation_queue = deque(maxlen=int(cache_size))

        self.visualizer = SceneGraphVisualizer(list(self.dataset_class.SEG_LABELS.keys()))
        self.smoother = TemporalSmoother(window_size=25, geo_vote_threshold=0.5)

    @torch.no_grad()
    @torch.inference_mode()
    def segment_frame(self, frame: Image.Image):
        frame = frame.resize((256,256))
        frame = np.array(frame)
        frame = T.ToTensor()(frame).float().unsqueeze(0) # 1 C H W
        frame = frame.to(self.trainer.device)
        if self.trainer.network.__class__ == LargeOctreeNCA2D:
            network_output = self.trainer.network.compute_features(frame, return_states_all_levels=self.trainer.relation_config["feature_extraction"].get("from_all_levels", False))
        else:
            network_output = self.trainer.network.compute_features(frame)
        self.segmentation_queue.append(network_output)

    def get_next_frame(self) -> Image.Image:
        frame = self.video_reader.get_next_data()
        frame = np.array(frame)
        frame = Image.fromarray(frame)
        return frame 

    def init(self):
        num_init_frames = np.abs(self.delta_t.min())
        for _ in range(num_init_frames):
            frame = self.get_next_frame()
            self.segment_frame(frame)

    def build_context(self):
        frame_logits = []
        frame_states = []
        for i in self.delta_t:
            idx = i-1
            frame_logits.append(self.segmentation_queue[idx]["logits"])
            frame_states.append(self.segmentation_queue[idx]["features"])
        frame_logits = torch.stack(frame_logits, dim=1) # 1 T C H W
        frame_states = torch.stack(frame_states, dim=1) # 1 T C H W
        
        return frame_logits, frame_states
    
    def get_semantic_relations(self, pred):
        if "relation_logits" not in pred:
            return []

        relation_logits = pred['relation_logits']
        relation_probabilities = torch.sigmoid(relation_logits)
        pair_proposals = pred["pair_proposals_original"][0]

        # ── temporal smoothing ──────────────────────────────────────────
        relation_probabilities, pair_proposals = self.smoother.update_semantic(
            relation_probabilities, pair_proposals
        )
        # ────────────────────────────────────────────────────────────────

        num_verbs = relation_probabilities.shape[1]

        if self.top_k_sampling:
            flat_probs = einops.rearrange(relation_probabilities, "N V -> (N V)")
            K_eff = min(self.k, flat_probs.numel())
            topk_scores, topk_idx = torch.topk(flat_probs, K_eff)
            pair_idx = topk_idx // num_verbs
            verb_idx = topk_idx % num_verbs

            pred_pairs = pair_proposals[pair_idx]
            pred_relations = set(
                (tuple(pred_pairs[i].tolist()), verb_idx[i].item(), topk_scores[i].item())
                for i in range(K_eff)
            )
            semantic_relations = [(sub, verb, obj, score) for (sub, obj), verb, score in pred_relations]
            semantic_relations = [(self.dataset_class.SEG_LABELS.inverse[int(sub)],
                                self.dataset_class.VERBS.inverse[int(verb)],
                                self.dataset_class.SEG_LABELS.inverse[int(obj)], score)
                                for (sub, verb, obj, score) in semantic_relations]
        else:
            positive_relations = relation_probabilities > self.threshold
            positive_pairs_mask = positive_relations.any(dim=1)

            positive_pair_proposals = pair_proposals[positive_pairs_mask]
            positive_relations = positive_relations[positive_pairs_mask]
            relation_probabilities = relation_probabilities[positive_pairs_mask]

            semantic_relations = []
            for rel_idx, relation in enumerate(positive_relations):
                for i, is_true in enumerate(relation):
                    if is_true:
                        subject = self.dataset_class.SEG_LABELS.inverse[int(positive_pair_proposals[rel_idx, 0])]
                        object_ = self.dataset_class.SEG_LABELS.inverse[int(positive_pair_proposals[rel_idx, 1])]
                        verb = self.dataset_class.VERBS.inverse[i]
                        score = float(relation_probabilities[rel_idx, i])
                        semantic_relations.append((subject, verb, object_, score))

        return sorted(semantic_relations, key=lambda x: x[3], reverse=True)

    def get_geometric_relations(self, frame_logits: torch.Tensor) -> list[tuple]:
        # print(frame_logits.shape) # (C H W)
        adjacent_classes = []
        for label_from, lbl_index_from in self.dataset_class.SEG_LABELS.items():
            for label_to, lbl_index_to in self.dataset_class.SEG_LABELS.items():
                if lbl_index_from <= lbl_index_to:
                    continue
                
                hard_map_from = self.trainer.get_hard_map(frame_logits[None], lbl_index_from)[0]
                hard_map_to = self.trainer.get_hard_map(frame_logits[None], lbl_index_to)[0]
                if not self.trainer.object_present(hard_map_from) or \
                not self.trainer.object_present(hard_map_to):
                    continue
                
                if SceneGraphBuilder.adjacent(hard_map_from, hard_map_to, radius=1):
                    adjacent_classes.append((label_from, label_to))



        return adjacent_classes

    def render_graph_panel(
        self,
        semantic_relations,
        geometric_relations,
        width=400,
        height=480
    ):
        #print(geometric_relations)
        return self.visualizer.render_graph_panel(semantic_relations, geometric_relations)

    def write_frame(self, frame: Image.Image, relations: list[tuple], geometric_relations: list[tuple], segmentation: np.ndarray):

        # Initialize VideoWriter if needed
        if not hasattr(self, "output_video_writer"):
            self.output_video_writer = cv2.VideoWriter(
                os.path.join(self.output_dir, "output.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),  # or "XVID"
                self.dataset_class.FPS,
                (854 + 400, 480) if self.render_graph else (854, 480)
            )
            if not self.output_video_writer.isOpened():
                raise RuntimeError("Failed to open VideoWriter!")

        frame = frame.resize((854, 480))
        frame = np.array(frame)

        # --- Ensure uint8 ---
        if np.issubdtype(frame.dtype, np.floating):
            frame_uint8 = np.clip(frame * 255, 0, 255).astype(np.uint8)
        else:
            frame_uint8 = frame.astype(np.uint8)

        # --- Convert PIL RGB → OpenCV BGR ---
        if frame_uint8.shape[2] == 3:
            frame_uint8 = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
        elif frame_uint8.shape[2] == 4:
            frame_uint8 = cv2.cvtColor(frame_uint8, cv2.COLOR_RGBA2BGR)
        elif frame_uint8.ndim == 2:
            frame_uint8 = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)

        frame_uint8 = self.overlay_segmentation_preview(
            frame_uint8,
            segmentation,
            scale=0.25,
            position="bottom_right"
        )


        if self.render_graph:
            graph_panel = self.render_graph_panel(relations, geometric_relations)
            combined = np.hstack([frame_uint8, graph_panel])

            frame_uint8 = combined
        

        
        # --- Optional: draw live caption ---
        for idx, (sub, verb, obj, score) in enumerate(relations):
            if self.full_sentence:
                verb = self.maybe_rename_verb(verb)
                text = f"The {sub} {verb} the {obj}."
                y = 30 + idx*15
                cv2.putText(frame_uint8, text, (854 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
            else:
                text = f"{sub}-{verb}->{obj} ({score:.2f})"
                y = 30 + idx*30
                cv2.putText(frame_uint8, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)


        self.output_video_writer.write(frame_uint8)

    def maybe_rename_verb(self, verb):
        if not self.dataset_class == CataractSceneGraphDataset:
            return f"{verb}s" 
        
        map = {
            'Holding': "holds",
            'Activation': "activates",
            'Pushing': "pushes",
            'Pulling': "pulls",
            'Cutting': "cuts",
            'Inserting': "is inserting into",
            'Retracting': "retracts",
            'null_verb': "null"
            }
        return map.get(verb, verb)

    def overlay_segmentation_preview(
        self,
        frame,
        mask,
        scale=0.25,
        margin=10,
        position="bottom_right"
    ):
        """
        Adds a small segmentation preview to the frame.
        """

        mask_rgb = self.mask_to_rgb(mask)

        # Resize preview
        h, w = mask_rgb.shape[:2]
        preview = cv2.resize(
            mask_rgb,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_NEAREST
        )

        ph, pw = preview.shape[:2]
        fh, fw = frame.shape[:2]

        # Compute placement
        if position == "bottom_right":
            y1 = fh - ph - margin
            x1 = fw - pw - margin
        elif position == "top_right":
            y1 = margin
            x1 = fw - pw - margin
        elif position == "bottom_left":
            y1 = fh - ph - margin
            x1 = margin
        else:  # top_left
            y1 = margin
            x1 = margin

        # Optional: draw border
        cv2.rectangle(
            frame,
            (x1 - 2, y1 - 2),
            (x1 + pw + 2, y1 + ph + 2),
            (255, 255, 255),
            1
        )

        frame[y1:y1 + ph, x1:x1 + pw] = preview

        return frame

    def mask_to_rgb(self, logits, threshold=0.5):
        """
        Convert sigmoid CHW segmentation logits to RGB mask.

        Args:
            logits: np.ndarray (C, H, W) or torch tensor
            threshold: foreground threshold

        Returns:
            H x W x 3 uint8 image
        """

        # --- Convert tensor → numpy ---
        if hasattr(logits, "cpu"):
            logits = logits.detach().cpu().numpy()


        C, H, W = logits.shape

        # --- Compute per-pixel max probability ---
        max_probs = logits.max(axis=0)
        class_ids = logits.argmax(axis=0)

        # --- Background mask ---
        background = max_probs < threshold

        # --- Create stable color palette ---
        if not hasattr(self, "class_colors"):
            rng = np.random.default_rng(0)
            self.class_colors = rng.integers(50, 255, size=(C, 3), dtype=np.uint8)

        # --- Build RGB output ---
        rgb = np.zeros((H, W, 3), dtype=np.uint8)

        # Assign foreground colors
        rgb[:] = self.class_colors[class_ids]

        # Override background
        rgb[background] = (30, 30, 30)

        return rgb


    def release(self):
        if hasattr(self, "output_video_writer"):
            self.output_video_writer.release()
        with open(os.path.join(self.output_dir, "predictions.json"), "w") as f:
            json.dump(self.prediction_dict, f)

    def __del__(self):
        self.release()

    @torch.no_grad()
    @torch.inference_mode()
    def run_inference(self, num_frames=None):
        self.init()

        if num_frames is None:
            num_frames = len(self.video_reader)

        for i in tqdm.tqdm(range(num_frames)):

            try:
                frame = self.get_next_frame()
            except IndexError:
                print("End of video reached.")
                break

            try:
                self.segment_frame(frame)


                frame_logits, frame_states = self.build_context()


                pred = self.trainer.predict_relations(frame_logits=frame_logits, frame_states=frame_states)
                semantic_relations = self.get_semantic_relations(pred)
                geometric_relations = self.get_geometric_relations(frame_logits[0, -1])
                geometric_relations = self.smoother.update_geometric(geometric_relations)  # ← add this
            except:
                print(f"Error processing frame {i}. Stopping.")
                break
            
            self.prediction_dict[i] = {
                "semantic_relations": semantic_relations,
                "geometric_relations": geometric_relations
            }
            self.write_frame(frame, semantic_relations, geometric_relations, frame_logits[0, -1])

        self.release()

        