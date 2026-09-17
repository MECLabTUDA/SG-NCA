import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from nsca.dataset import ShapeSceneDataset
from nsca.nsca_model import NeuroSymbolicCA


device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuroSymbolicCA().to(device)
num_epochs = 30


dataset = ShapeSceneDataset(device=device, num_samples=500)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Loss functions
criterion_seg = nn.CrossEntropyLoss()
criterion_rel = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    total_loss = 0
    total_seg_loss = 0
    total_rel_loss = 0
    total_sym_penalty = 0
    for img, mask, nodes, edges in tqdm.tqdm(loader):
        optimizer.zero_grad()
        
        # 1. Forward Pass (CA Evolution)
        seg_logits, final_state = model(img)

        # 2. Segmentation Loss
        loss_seg = criterion_seg(seg_logits, mask)
        
        # 3. Relation Loss (Scene Graph)
        # Target for our 3 specific relations: [1, 1, 0] 
        # (Circle-Tri=1, Circle-Rect=1, Tri-Rect=0)
        rel_preds = model.predict_relations(final_state, mask)
        rel_targets = torch.tensor([1, 1, 0], device=img.device).repeat(img.shape[0], 1)
        
        # Flatten to compute CrossEntropy
        loss_rel = criterion_rel(rel_preds.view(-1, 2), rel_targets.view(-1))
        
        # 4. Symbolic Penalty (Optional)
        # Penalize any Tri-Rect relationship strongly
        tri_rect_relationship = rel_preds[:, 2, 1]
        symbolic_penalty = criterion_rel(tri_rect_relationship, torch.zeros_like(tri_rect_relationship))
        #symbolic_penalty = torch.mean(rel_preds[:, 2, 1]) # Probability of edge between 1 and 2
        
        loss = loss_seg + loss_rel + (0.5 * symbolic_penalty)
        assert loss > 0, f"{loss_seg}, {loss_rel}, {symbolic_penalty}"
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_seg_loss += loss_seg.item()
        total_rel_loss += loss_rel.item()
        total_sym_penalty += symbolic_penalty.item()
        
    print(f"Epoch {epoch} | Avg Loss: {total_loss/len(loader):.4f} | Seg Loss: {total_seg_loss/len(loader):.4f} | Rel Loss: {total_rel_loss/len(loader):.4f} | Sym Loss: {total_sym_penalty/len(loader):.4f}")

torch.save(model.state_dict(), "/local/scratch/clmn1/videoNCA/NSCA/model.pth")