
import einops
import torch
import torch.nn as nn


class RelationClassifier(nn.Module):
    def __init__(self, feature_dim, num_frames, num_verbs, method, frame_encoding, clip_encoding):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        if frame_encoding:
            self.frame_encoder = nn.Linear(feature_dim, frame_encoding, bias=False)
            feature_dim = frame_encoding
        else:
            self.frame_encoder = nn.Identity()

        if clip_encoding:
            self.clip_encoder = nn.Linear(feature_dim * num_frames, clip_encoding, bias=False)
            clip_dim = clip_encoding
        else:
            self.clip_encoder = nn.Identity()
            clip_dim = feature_dim * num_frames


        self.relation_predictor = nn.Sequential(
            nn.Linear(2* clip_dim, 2* clip_dim),
            nn.ReLU(),
            nn.Linear(2* clip_dim, clip_dim),
            nn.ReLU(),
            nn.Linear(clip_dim, num_verbs)
        )
    
    def forward(self, object_features_list, pair_indices_list):
        """
        Args:
            object_features_list: List[Tensor(Num_Objs, Feat_Dim)]
            pair_indices_list: List[List[(subj_idx, obj_idx)]]
        Returns:
            List[Tensor(num_pairs, num_verbs)] - one per image
        """
        batch_size = len(object_features_list)
        all_pair_logits = []

        for b in range(batch_size):
            obj_feats = object_features_list[b]          # (Num_Objs, T, C)
            obj_feats = self.frame_encoder(obj_feats)     # (Num_Objs, T, Frame_Enc_Dim)
            obj_feats = einops.rearrange(obj_feats, "N T C -> N (T C)")  # (Num_Objs, T*C)
            obj_feats = self.clip_encoder(obj_feats)     # (Num_Objs, Clip_Feats)
            pairs = pair_indices_list[b]

            if len(pairs) == 0:
                all_pair_logits.append(
                    torch.empty(0, self.relation_predictor[-1].out_features,
                                device=obj_feats.device)
                )
                continue

            subj_idxs = torch.tensor([p[0] for p in pairs], device=obj_feats.device)
            obj_idxs  = torch.tensor([p[1] for p in pairs], device=obj_feats.device)

            subj_feats = obj_feats[subj_idxs]  # (Pairs, Clip_Feats)
            obj_feats_ = obj_feats[obj_idxs]   # (Pairs, Clip_Feats)


            pair_feats = torch.stack([subj_feats, obj_feats_])  # (2P, C)
            pair_feats = einops.rearrange(pair_feats, "two P C -> P (two C)")  # (P, 2*C)


            logits = self.relation_predictor(pair_feats)  # (P, num_verbs)
            all_pair_logits.append(logits)

        all_pair_logits = torch.cat(all_pair_logits, dim=0)
        return all_pair_logits