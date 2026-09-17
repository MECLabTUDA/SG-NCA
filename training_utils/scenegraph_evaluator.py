



from collections import defaultdict
import torch
import json


class SceneGraphEvaluator():
    def __init__(self, num_verbs: int, K: int, device: torch.device, idx_to_verb: dict[int, str]=None):
        self.device = device
        self.num_verbs = num_verbs # excluding "no_relation"
        self.k = K
        self.idx_to_verb = idx_to_verb if idx_to_verb is not None else {i: str(i) for i in range(num_verbs)}

        self.reset()


    def reset(self):
        self.ap_data = defaultdict(list)     # verb -> list of (score, is_tp)
        self.gt_count = torch.zeros(self.num_verbs, device=self.device, dtype=torch.int)
        self.recall_sum = 0.0
        self.num_images = 0

        self.mr_tp = torch.zeros(self.num_verbs, device=self.device, dtype=torch.int)
        self.mr_gt = torch.zeros(self.num_verbs, device=self.device, dtype=torch.int)



    @torch.no_grad()
    def record_batch(self, prediction: dict[str, torch.Tensor], relations_dict: list[dict]):
        relation_logits = prediction["relation_logits"]
        num_pairs = prediction["num_pairs"]
        chunks = torch.split(relation_logits, num_pairs, dim=0)

        for b, rel_logits in enumerate(chunks):
            # -----------------------------
            # Predictions
            # -----------------------------
            pred_probs = torch.sigmoid(rel_logits)
            num_pairs_b, num_verbs = pred_probs.shape

            flat_probs = pred_probs.reshape(-1)
            K_eff = min(self.k, flat_probs.numel())

            topk_scores, topk_idx = torch.topk(flat_probs, k=K_eff)

            pair_idx = topk_idx // num_verbs
            verb_idx = topk_idx % num_verbs
            pred_pairs = prediction["pair_proposals_original"][b][pair_idx] # use these to extract ground truths
            #pred_pairs_reordered = prediction["pair_proposals"][b][pair_idx] # use these to extract predictions 


            pred_relations = set(
                (tuple(pred_pairs[i].tolist()), verb_idx[i].item())
                for i in range(K_eff)
            )
            #print(pred_relations)

            # -----------------------------
            # Ground truth (build ONCE)
            # -----------------------------
            gt_relations = set()
            gt_pairs_by_verb = {v: set() for v in range(num_verbs)}

            for pair, verb_labels in relations_dict[b].items():
                for v in range(num_verbs):
                    if verb_labels[v] == 1:
                        gt_relations.add((pair, v))
                        gt_pairs_by_verb[v].add(pair)
                        self.mr_gt[v] += 1
                        self.gt_count[v] += 1

            # -----------------------------
            # Recall@K (image-level)
            # -----------------------------
            if len(gt_relations) > 0:
                matched = gt_relations & pred_relations
                self.recall_sum += len(matched) / len(gt_relations)
                self.num_images += 1

            # -----------------------------
            # mRecall@K (predicate-wise)
            # -----------------------------
            for v in range(num_verbs):
                self.mr_tp[v] += len(
                    gt_pairs_by_verb[v] & {
                        pair for (pair, pv) in pred_relations if pv == v
                    }
                )

            # -----------------------------
            # mAP@K (one-to-one matching)
            # -----------------------------
            used_gt = {v: set() for v in range(num_verbs)}

            for i in range(K_eff):
                pair = tuple(pred_pairs[i].tolist())
                v = verb_idx[i].item()
                score = topk_scores[i].item()

                is_tp = pair in gt_pairs_by_verb[v] and pair not in used_gt[v]

                if is_tp:
                    used_gt[v].add(pair)

                self.ap_data[v].append((score, is_tp))

        
        #print(self.ap_data)
        #print(self.gt_count)
        #print(self.mr_tp)
        #print(self.mr_gt)
        #print(self.recall_sum)
        #print(self.num_images)
        #exit()


    def print_results(self):
        results_dict = self.get_results_dict()
        
        print(f"Recall@K: {results_dict['Recall@K']:.4f}, mRecall@K: {results_dict['mRecall@K']:.4f}, mAP@K: {results_dict['mAP@K']:.4f}")

        longest_verb_name = max(len(v) for v in results_dict["gt_count_per_verb"].keys()) + 4

        print(f"{'Verb':>{longest_verb_name}} | {'GT':>5} | {'Recall':>7} | {'AP@K':>6}")
        print("-" * (28 + longest_verb_name))
        for v in results_dict["gt_count_per_verb"].keys():
            print(
                f"{v:>{longest_verb_name}} | "
                f"{int(results_dict['gt_count_per_verb'][v]):>5} | "
                f"{results_dict['recall_per_verb'][v]:>7.3f} | "
                f"{results_dict['ap@K_per_verb'][v]:>6.3f}"
            )

    def get_results_dict(self) -> dict:
        Recall_k = self.recall_sum / max(self.num_images, 1)
        mRecall_k = torch.mean(
            self.mr_tp[self.mr_gt > 0] / self.mr_gt[self.mr_gt > 0]
        ).item()
        ap_values = []
        for v in range(self.num_verbs):
            ap_v = self.average_precision(self.ap_data[v], self.gt_count[v])
            if ap_v is not None:
                ap_values.append(ap_v)

        mAP_k = sum(ap_values) / len(ap_values)
        
        results_dict = {'K': self.k}
        results_dict["Recall@K"] = Recall_k
        results_dict["mRecall@K"] = mRecall_k
        results_dict["mAP@K"] = mAP_k.item()

        results_dict["gt_count_per_verb"] = {}
        results_dict["recall_per_verb"] = {}
        results_dict["ap@K_per_verb"] = {}


        ap_per_verb = {}
        for v in range(self.num_verbs):
            ap_v = self.average_precision(self.ap_data[v], self.gt_count[v])
            if ap_v is None:
                ap_v = float("nan")  # no GT for this verb
            else:
                ap_v = ap_v.item()
            ap_per_verb[v] = ap_v

        sorted_verbs = sorted(
            range(self.num_verbs),
            key=lambda v: self.gt_count[v],
            reverse=True
        )

        for v in sorted_verbs:
            if self.gt_count[v] > 0:
                recall_v = (self.mr_tp[v] / self.gt_count[v]).item()
            else:
                recall_v = float("nan")

            v_name = self.idx_to_verb[v]
            results_dict["gt_count_per_verb"][v_name] = int(self.gt_count[v])
            results_dict["recall_per_verb"][v_name] = recall_v
            results_dict["ap@K_per_verb"][v_name] = ap_per_verb[v]

        return results_dict

    def save_results(self, filepath: str):
        results_dict = self.get_results_dict()
        with open(filepath, "w") as f:
            json.dump(results_dict, f, indent=4)


    @staticmethod
    def average_precision(preds, num_gt):
        if num_gt == 0:
            return None

        preds = sorted(preds, key=lambda x: -x[0])

        tp = 0
        fp = 0
        ap = 0.0

        for _, is_tp in preds:
            if is_tp:
                tp += 1
                ap += tp / (tp + fp)
            else:
                fp += 1

        return ap / num_gt