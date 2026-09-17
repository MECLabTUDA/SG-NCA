import einops
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MotifRelationClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes, num_frames, hidden_dim=512, num_layers=2):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames

        # 1. Project input features (T*C or pooled) to hidden dim
        self.obj_projector = nn.Linear(num_frames*feature_dim, hidden_dim)

        # 2. Object Context (Bi-LSTM)
        # "Reads" the list of objects in the image to understand the scene
        self.obj_ctx_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        # 3. Edge Predictor (The "Motif" part)
        # In the paper, this combines: [Subj_Ctx, Obj_Ctx, Subj_Raw, Obj_Raw]
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, object_features_list, pair_indices_list):
        """
        Args:
            object_features_list: List[Tensor(Num_Objs, Feat_Dim)] - one tensor per image
            pair_indices_list: List[List[(subj_idx, obj_idx)]] - pairs to classify per image
        """
        object_features_list = [einops.rearrange(obj_feats, "N T C -> N (T C)") for obj_feats in object_features_list]
        batch_size = len(object_features_list)
        
        # --- Step 1: Prepare Batch for LSTM ---
        # We need to pad the sequences of objects to process them in one batch
        num_objs_per_img = [len(f) for f in object_features_list]
        max_objs = max(num_objs_per_img)
        feat_dim = object_features_list[0].shape[1]
        
        # (B, Max_Objs, Feat_Dim)
        padded_objs = torch.zeros(batch_size, max_objs, feat_dim, device=object_features_list[0].device)
        
        for i, feats in enumerate(object_features_list):
            padded_objs[i, :num_objs_per_img[i]] = feats

        # --- Step 2: Global Context (LSTM) ---
        # Project to hidden dimension
        projected_objs = self.obj_projector(padded_objs)

        # Pack sequence for LSTM (handles variable number of objects efficiently)
        packed_input = pack_padded_sequence(
            projected_objs, 
            torch.tensor(num_objs_per_img).cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # Run Bi-LSTM
        packed_output, _ = self.obj_ctx_lstm(packed_input)
        
        # Unpack back to (B, Max_Objs, Hidden_Dim)
        contextualized_objs, _ = pad_packed_sequence(packed_output, batch_first=True)

        # --- Step 3: Form Pairs & Classify ---
        all_pair_logits = []
        
        for b in range(batch_size):
            pairs = pair_indices_list[b]
            if len(pairs) == 0:
                continue
                
            # Extract indices
            subj_idxs = [p[0] for p in pairs]
            obj_idxs = [p[1] for p in pairs]
            
            # Gather features for the pairs
            # We use both the NEW context features and the OLD raw features (skip connection)
            subj_ctx = contextualized_objs[b, subj_idxs]
            obj_ctx = contextualized_objs[b, obj_idxs]
            subj_raw = projected_objs[b, subj_idxs]
            obj_raw = projected_objs[b, obj_idxs]
            
            # Combine: (Subj_Ctx, Obj_Ctx, Subj_Raw, Obj_Raw)
            edge_input = torch.cat([subj_ctx, obj_ctx, subj_raw, obj_raw], dim=1)
            
            # Predict
            logits = self.edge_classifier(edge_input)
            all_pair_logits.append(logits)


        if len(all_pair_logits) > 0:
            return torch.cat(all_pair_logits, dim=0)
        else:
            return torch.empty(0, self.num_classes, device=padded_objs.device)