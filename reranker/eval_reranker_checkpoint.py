#!/usr/bin/env python
# Evaluate a BGE-style reranker checkpoint on our processed triplets.
# Metrics: PairwiseAcc, R@1, MRR@10, nDCG@10, avg score margin.
# Requires: pip install -U "FlagEmbedding" ujson tqdm
'''
# activate your env first
python scripts/eval_reranker_checkpoint.py \
  --model_path outputs/reranker-medical-gemma/checkpoint-20000 \
  --data_path data/processed/medical_reranker_test.jsonl.gz
'''

import os, gzip, json, argparse
from pathlib import Path
from typing import List, Tuple
try:
    import ujson as jsonlib
except Exception:
    import json as jsonlib
from tqdm import tqdm
import math

def read_triplets(fp: Path, max_samples: int = None):
    opener = gzip.open if fp.suffix == ".gz" else open
    n = 0
    with opener(fp, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            row = jsonlib.loads(line)
            q = row.get("query", "").strip()
            pos = [p for p in row.get("pos", []) if p]
            neg = [n for n in row.get("neg", []) if n]
            if q and pos and neg:
                yield q, pos, neg
                n += 1
                if max_samples and n >= max_samples:
                    return

def dcg(scores: List[float], labels: List[int], k: int = 10):
    # sort by score desc, take top-k
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    res = 0.0
    for rank, idx in enumerate(order, start=1):
        rel = labels[idx]
        res += (2**rel - 1) / math.log2(rank + 1)
    return res

def ndcg_at_k(scores, labels, k=10):
    best = dcg(labels, labels, k)  # ideal = sort by label
    if best == 0:
        return 0.0
    return dcg(scores, labels, k) / best

def evaluate(model_path: str, data_path: str, batch: int = 16, limit: int = None, fp16=True, normalize=True):
    from FlagEmbedding import FlagReranker
    reranker = FlagReranker(model_path, use_fp16=fp16)

    pairwise_ok = 0
    total = 0
    r_at_1 = 0
    mrr10_sum = 0.0
    ndcg10_sum = 0.0
    margin_sum = 0.0

    for q, pos_list, neg_list in tqdm(read_triplets(Path(data_path), limit), desc="Eval"):
        # Candidates = all positives + all negatives
        cands = pos_list + neg_list
        labels = [1]*len(pos_list) + [0]*len(neg_list)

        # Compute scores (single call per sample)
        scores = reranker.compute_score(q, cands, normalize=normalize)

        # Pairwise accuracy: best positive beats best negative
        pos_best = max(scores[:len(pos_list)]) if pos_list else -1e9
        neg_best = max(scores[len(pos_list):]) if neg_list else -1e9
        pairwise_ok += int(pos_best > neg_best)

        # R@1
        top_idx = max(range(len(scores)), key=lambda i: scores[i])
        r_at_1 += int(labels[top_idx] == 1)

        # MRR@10
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        rr = 0.0
        for rank, idx in enumerate(order, start=1):
            if labels[idx] == 1:
                rr = 1.0 / rank
                break
        mrr10_sum += rr

        # nDCG@10
        ndcg10_sum += ndcg_at_k(scores, labels, k=10)

        # avg margin (best pos - best neg)
        margin_sum += (pos_best - neg_best)
        total += 1

    if total == 0:
        print("No samples found.")
        return

    print("\n=== Evaluation ===")
    print(f"samples           : {total}")
    print(f"Pairwise Accuracy : {pairwise_ok/total:.4f}")
    print(f"Recall@1          : {r_at_1/total:.4f}")
    print(f"MRR@10            : {mrr10_sum/total:.4f}")
    print(f"nDCG@10           : {ndcg10_sum/total:.4f}")
    print(f"Avg margin (P-N)  : {margin_sum/total:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="e.g., outputs/reranker-medical-gemma/checkpoint-20000 or 
.../merged")
    ap.add_argument("--data_path",  default="data/processed/medical_reranker_test.jsonl.gz")
    ap.add_argument("--batch_size", type=int, default=16)  # (kept for future batching)
    ap.add_argument("--limit", type=int, default=None, help="cap #samples for quick checks")
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--no_normalize", action="store_true")
    args = ap.parse_args()
    evaluate(
        model_path=args.model_path,
        data_path=args.data_path,
        batch=args.batch_size,
        limit=args.limit,
        fp16=not args.no_fp16,
        normalize=not args.no_normalize
    )

