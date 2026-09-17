from collections import deque
import torch
import numpy as np


class TemporalSmoother:
    """
    Smooths semantic relation probabilities and geometric relations over a
    sliding window of recent frames.

    Args:
        window_size:        Number of frames to average over.
        geo_vote_threshold: Fraction of recent frames a geometric relation must
                            appear in to be reported (0‑1).  0.5 = majority vote.
    """

    def __init__(self, window_size: int = 5, geo_vote_threshold: float = 0.5):
        self.window_size = window_size
        self.geo_vote_threshold = geo_vote_threshold

        # Each entry: tensor of shape (N_pairs, V) – raw sigmoid probabilities
        self._prob_buffer: deque[torch.Tensor | None] = deque(maxlen=window_size)
        # Each entry: set of (label_from, label_to) string pairs
        self._geo_buffer: deque[set] = deque(maxlen=window_size)

        # We also need to carry the pair proposals alongside probabilities so
        # we know which (sub, obj) each row belongs to.  Store them too.
        self._pair_buffer: deque[torch.Tensor | None] = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Semantic smoothing
    # ------------------------------------------------------------------

    def update_semantic(
        self,
        relation_probabilities: torch.Tensor,   # (N_pairs, V)
        pair_proposals: torch.Tensor,           # (N_pairs, 2)  subject/object class ids
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Push the current frame's data and return temporally smoothed
        probabilities and the corresponding pair proposals.

        Because the set of proposed pairs can change between frames we
        index by (subject_id, object_id) key and average only where a
        pair appears in multiple frames.

        Returns:
            smoothed_probs  – (M, V) averaged probabilities
            smoothed_pairs  – (M, 2) corresponding pair proposals
        """
        self._prob_buffer.append(relation_probabilities.detach().cpu())
        self._pair_buffer.append(pair_proposals.detach().cpu())

        # Aggregate across the window keyed by pair identity
        pair_accum: dict[tuple, list[torch.Tensor]] = {}
        for probs, pairs in zip(self._prob_buffer, self._pair_buffer):
            if probs is None or pairs is None:
                continue
            for row_idx in range(pairs.shape[0]):
                key = (int(pairs[row_idx, 0]), int(pairs[row_idx, 1]))
                pair_accum.setdefault(key, []).append(probs[row_idx])

        if not pair_accum:
            return relation_probabilities, pair_proposals  # fallback

        keys = list(pair_accum.keys())
        smoothed_probs = torch.stack(
            [torch.stack(pair_accum[k]).mean(dim=0) for k in keys]
        )  # (M, V)
        smoothed_pairs = torch.tensor(keys, dtype=pair_proposals.dtype)  # (M, 2)

        return smoothed_probs, smoothed_pairs

    # ------------------------------------------------------------------
    # Geometric smoothing
    # ------------------------------------------------------------------

    def update_geometric(self, geometric_relations: list[tuple]) -> list[tuple]:
        """
        Push the current frame's geometric relations and return only those
        that appear in at least `geo_vote_threshold` fraction of the window.
        """
        self._geo_buffer.append(set(geometric_relations))

        if not self._geo_buffer:
            return geometric_relations

        # Count how many frames each relation appears in
        vote_counts: dict[tuple, int] = {}
        for frame_rels in self._geo_buffer:
            for rel in frame_rels:
                vote_counts[rel] = vote_counts.get(rel, 0) + 1

        min_votes = max(1, int(np.ceil(len(self._geo_buffer) * self.geo_vote_threshold)))
        return [rel for rel, count in vote_counts.items() if count >= min_votes]
