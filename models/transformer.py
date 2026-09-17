import einops
import torch
import torch.nn as nn

class TransformerRelationClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes, num_frames, hidden_dim=256, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. Project input features to Transformer dimension
        self.obj_projector = nn.Linear(num_frames * feature_dim, hidden_dim)
        

        # 3. Transformer Encoder
        # batch_first=True makes input (Batch, Seq_Len, Dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Classification Head
        # Concatenates: [Subj_Ctx, Obj_Ctx] -> Prediction
        # (We can also add raw features like in Motif, but Transformers often don't need the skip connection as much)
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, object_features_list, pair_indices_list):
        """
        Args:
            object_features_list: List[Tensor(Num_Objs, Feat_Dim)]
            pair_indices_list: List[List[(subj_idx, obj_idx)]]
        """
        object_features_list = [einops.rearrange(obj_feats, "N T C -> N (T C)") for obj_feats in object_features_list]
        batch_size = len(object_features_list)
        
        # --- Step 1: Pad and Batch ---
        num_objs_per_img = [len(f) for f in object_features_list]
        max_objs = max(num_objs_per_img)
        
        if max_objs == 0:
             return torch.empty(0, self.num_classes, device=object_features_list[0].device)

        feat_dim = object_features_list[0].shape[1]
        
        # Prepare inputs
        padded_objs = torch.zeros(batch_size, max_objs, feat_dim, device=object_features_list[0].device)
        
        # Create Padding Mask (True = Ignore this position)
        # Shape: (Batch, Seq_Len)
        key_padding_mask = torch.ones(batch_size, max_objs, dtype=torch.bool, device=object_features_list[0].device)

        for i, feats in enumerate(object_features_list):
            n = num_objs_per_img[i]
            if n > 0:
                padded_objs[i, :n] = feats
                key_padding_mask[i, :n] = False # Do NOT ignore valid objects

        # --- Step 2: Transformer Pass ---
        # Project
        x = self.obj_projector(padded_objs)
        
        # (Optional) Add Class embeddings if you had class IDs passed in. 
        # For now, we rely on the features themselves being distinct enough.

        # Encode
        # output: (Batch, Max_Objs, Hidden_Dim)
        contextualized_objs = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # --- Step 3: Classify Pairs ---
        all_pair_logits = []
        
        for b in range(batch_size):
            pairs = pair_indices_list[b]
            if len(pairs) == 0:
                continue
                
            subj_idxs = [p[0] for p in pairs]
            obj_idxs = [p[1] for p in pairs]
            
            # Extract features for Subject and Object from the Transformer output
            subj_ctx = contextualized_objs[b, subj_idxs] # (Num_Pairs, Hidden_Dim)
            obj_ctx = contextualized_objs[b, obj_idxs]   # (Num_Pairs, Hidden_Dim)
            
            # Combine
            edge_input = torch.cat([subj_ctx, obj_ctx], dim=1)
            
            logits = self.edge_classifier(edge_input)
            all_pair_logits.append(logits)

        if len(all_pair_logits) > 0:
            return torch.cat(all_pair_logits, dim=0)
        else:
            return torch.empty(0, self.num_classes, device=padded_objs.device)