from data.cholec_scene_graph import CholecSceneGraphDataset
import torch
import torch.nn.functional as F
from pyvis.network import Network
import data.cholec_vis as cholec_vis

class SceneGraphBuilder:
    def __init__(self):
        
        pass

    @staticmethod
    def dilate(mask, radius=3):
        kernel = torch.ones((1, 1, 2*radius+1, 2*radius+1), device=mask.device)
        mask = mask.unsqueeze(0).unsqueeze(0).float()
        dilated = F.conv2d(mask, kernel, padding=radius) > 0
        return dilated.squeeze()
    @staticmethod
    def adjacent(hard_mask_i, hard_mask_j, radius=3):
        mi = SceneGraphBuilder.dilate(hard_mask_i, radius)
        mj = hard_mask_j
        return (mi & mj).any()
    
    @staticmethod   
    def rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)   
    
    def build_scene_graph(self, trainer, frame_logits, relation_logits, pair_proposals):
        pair_proposal_strings = []
        for b, tool, tissue in pair_proposals:
            assert b == 0, "No batching supported in this example"
            tool = CholecSceneGraphDataset.SEG_LABELS.inverse[tool]
            tissue = CholecSceneGraphDataset.SEG_LABELS.inverse[tissue]
            pair_proposal_strings.append((tool, tissue))

        
        positive_relations = []
        for i, relation_idx in enumerate(relation_logits.argmax(dim=-1)):
            if relation_idx != CholecSceneGraphDataset.VERBS['null_verb']:
                tool, tissue = pair_proposal_strings[i]
                verb = list(CholecSceneGraphDataset.VERBS.keys())[relation_idx]
                positive_relations.append((tool, verb, tissue))

        adjacent_classes = []
        for label_from, lbl_index_from in CholecSceneGraphDataset.SEG_LABELS.items():
            for label_to, lbl_index_to in CholecSceneGraphDataset.SEG_LABELS.items():
                if lbl_index_from <= lbl_index_to:
                    continue
                hard_map_from = trainer.get_hard_map(frame_logits[0, 2], lbl_index_from)
                hard_map_to = trainer.get_hard_map(frame_logits[0, 2], lbl_index_to)
                if not trainer.object_present(hard_map_from) or \
                not trainer.object_present(hard_map_to):
                    continue
                
                if SceneGraphBuilder.adjacent(hard_map_from, hard_map_to, radius=1):
                    adjacent_classes.append((label_from, label_to))

        return {
            "positive_relations": positive_relations,
            "adjacent_classes": adjacent_classes
        }

    def visualize_scene_graph(self, trainer, frame_logits, relation_logits, pair_proposals,
                                output_file=None) -> Network:
        scene_graph = self.build_scene_graph(trainer, frame_logits, relation_logits, pair_proposals)
        return SceneGraphBuilder.visualize(
            scene_graph["positive_relations"],
            scene_graph["adjacent_classes"],
            output_file=output_file
        )

    @staticmethod
    def visualize(positive_relations, adjacent_classes,
                                height="600px",
                                width="100%",
                                output_file=None) -> Network:
        
        net = Network(
            height=height,
            width=width,
            directed=False,     # undirected graph
            notebook=True
        )
        net.set_options("""
        {
        "physics": {
            "enabled": false
        },
        "edges": {
            "smooth": false
        }
        }
        """)

        all_classes = set()
        for a, verb, b in positive_relations:
            all_classes.add(a)
            all_classes.add(b)
        for a, b in adjacent_classes:
            all_classes.add(a)
            all_classes.add(b)
        all_classes = list(all_classes)
        
        for n in all_classes:
            color = SceneGraphBuilder.rgb_to_hex(cholec_vis.get_cholecseg8k_colormap()[CholecSceneGraphDataset.SEG_LABELS[n]])

            net.add_node(
                n,
                label=str(n),
                color=color,
                font={"align": "center"}
            )

        # Add edges
        for a, verb, b in positive_relations:
            net.add_edge(a, b, label=verb, width=5)
        for a, b in adjacent_classes:
            net.add_edge(a, b)

        if output_file is not None:
            net.show(output_file)
        return net