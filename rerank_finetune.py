#!/usr/bin/env python
# Finetune BAAI/bge-reranker-v2-gemma on local SciDocs (BEIR-style, if qrels present),
# BioASQ-generated-queries (jsonl.gz), and MIRIAD 4.4M (parquet).
# Produces triplets JSONL (train/val/test), trains via FlagEmbedding (decoder-only reranker),
# logs richly (tensorboard/CSV), and runs offline evaluation with IR metrics + plots.
#
# Required wheels:
#   pip install -U "FlagEmbedding[finetune]" datasets pyarrow ujson numpy tqdm matplotlib
#

# - Evaluation uses the decoder-only LLM reranker inference class from FlagEmbedding docs.

import os, sys, gzip, glob, csv, random, argparse
from pathlib import Path

try:
    import ujson as jsonlib
except Exception:
    import json as jsonlib

import numpy as np
from datasets import load_dataset, disable_caching
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import subprocess
from tqdm import tqdm

# Optional plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# --------------------- utils ---------------------
def set_env(cache_dir):
    os.environ.setdefault("HF_HOME", str(Path(cache_dir).resolve()))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(cache_dir).resolve()))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def safe_join_title_text(title, text):
    title = (title or "").strip()
    text  = (text or "").strip()
    if title and text and not text.lower().startswith(title.lower()):
        return f"{title}\n\n{text}"
    return title or text

def write_jsonl(path: Path, rows_iter):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = gzip.open(path, "wt", encoding="utf-8") if path.suffix == ".gz" else open(path, "w", encoding="utf-8")
    with out as f:
        for r in rows_iter:
            f.write(jsonlib.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield jsonlib.loads(line)

def split_list(xs, val_ratio=0.05, test_ratio=0.05, seed=42):
    rnd = random.Random(seed)
    xs = list(xs)
    rnd.shuffle(xs)
    n = len(xs)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    val = xs[:n_val]
    test = xs[n_val:n_val+n_test]
    train = xs[n_val+n_test:]
    return train, val, test

# --------------------- loaders: local disk ---------------------
def load_scidocs_local(scidocs_dir: Path, negatives_per_query=4, max_queries=None, seed=42):
    """
    Reads BEIR-style SciDocs if (and only if) qrels exist:
      scidocs/corpus.jsonl.gz, scidocs/queries.jsonl.gz, scidocs/qrels/{*.tsv}
    Yields triplets: {"query", "pos": [...], "neg": [...], "source": "scidocs"}
    """
    rnd = random.Random(seed)
    corpus_fp = scidocs_dir / "corpus.jsonl.gz"
    queries_fp = scidocs_dir / "queries.jsonl.gz"
    qrels_dir = scidocs_dir / "qrels"

    if not corpus_fp.exists() or not queries_fp.exists():
        print(f"[SciDocs] Missing corpus/queries at {scidocs_dir} → skipping.", flush=True)
        return

    qrels_files = sorted(glob.glob(str(qrels_dir / "*.tsv")))
    if not qrels_files:
        print(f"[SciDocs] No qrels/*.tsv found at {qrels_dir} → skipping SciDocs (cannot form positives).", flush=True)
        return

    # load corpus
    doc_text = {}
    opener = gzip.open if corpus_fp.suffix == ".gz" else open
    with opener(corpus_fp, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = jsonlib.loads(line)
            did = row.get("_id")
            text = safe_join_title_text(row.get("title",""), row.get("text",""))
            if did and text:
                doc_text[did] = text
    all_doc_ids = list(doc_text.keys())

    # load queries
    queries = {}
    opener = gzip.open if queries_fp.suffix == ".gz" else open
    with opener(queries_fp, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = jsonlib.loads(line)
            qid = row.get("_id")
            qtx = (row.get("text") or "").strip()
            if qid and qtx:
                queries[qid] = qtx

    # read qrels (query-id \t corpus-id \t score)
    qpos = {}
    for qrels_fp in qrels_files:
        with open(qrels_fp, "r", encoding="utf-8") as tsv:
            reader = csv.reader(tsv, delimiter="\t")
            header = next(reader, None)
            for parts in reader:
                if not parts:
                    continue
                # be permissive with columns
                if len(parts) >= 3:
                    qid, did, score = parts[:3]
                elif len(parts) == 2:
                    qid, did = parts
                    score = "1"
                else:
                    continue
                try:
                    s = int(score)
                except:
                    s = 1
                if s > 0 and qid in queries and did in doc_text:
                    qpos.setdefault(qid, set()).add(did)

    emitted = 0
    for qid, pos_ids in qpos.items():
        q = queries.get(qid)
        if not q:
            continue
        positives = [doc_text[pid] for pid in list(pos_ids)[:3] if pid in doc_text]
        # sample negatives
        negs = set()
        tries, target = 0, negatives_per_query
        while len(negs) < target and tries < target * 40:
            did = rnd.choice(all_doc_ids)
            if did not in pos_ids:
                negs.add(did)
            tries += 1
        negatives = [doc_text[d] for d in negs]
        if positives and negatives:
            yield {"query": q, "pos": positives, "neg": negatives, "source": "scidocs"}
            emitted += 1
            if max_queries and emitted >= max_queries:
                break

def load_bioasq_generated_local(jsonl_gz: Path, negatives_per_query=4, max_samples=500000, buffer_size=100000, seed=42):
    """
    Reads BeIR/bioasq-generated-queries train.jsonl.gz (local).
    Fields include: _id, title, text, query. We use (query, title+text) as positive.
    Negatives sampled from a rolling buffer.
    """
    if not jsonl_gz.exists():
        print(f"[BioASQ-gen] {jsonl_gz} not found → skipping.", flush=True)
        return

    rnd = random.Random(seed)
    buf = []  # store docs for negatives
    emitted = 0
    opener = gzip.open if jsonl_gz.suffix == ".gz" else open

    with opener(jsonl_gz, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = jsonlib.loads(line)
            q = (row.get("query") or "").strip()
            doc = safe_join_title_text(row.get("title",""), row.get("text",""))
            if not q or not doc:
                continue

            # negatives from buffer
            negatives = []
            if buf:
                candidates = list(buf)
                rnd.shuffle(candidates)
                negatives = candidates[:negatives_per_query]

            if negatives:
                yield {"query": q, "pos": [doc], "neg": negatives, "source": "bioasq-generated-queries"}
                emitted += 1
                if max_samples and emitted >= max_samples:
                    break

            buf.append(doc)
            if len(buf) > buffer_size:
                buf.pop(0)

def load_miriad_local(parquet_glob: str, negatives_per_query=4, max_samples=350000, buffer_size=200000, seed=42):
    """
    Reads local parquet shards for MIRIAD: expected columns include question, passage_text.
    Streaming via HF datasets (local files only).
    """
    files = sorted(glob.glob(parquet_glob))
    if not files:
        print(f"[MIRIAD] No parquet files matched: {parquet_glob} → skipping.", flush=True)
        return

    rnd = random.Random(seed)
    ds = load_dataset("parquet", data_files={"train": files}, split="train", streaming=True)
    buf, emitted = [], 0
    for row in ds:
        q = (row.get("question") or "").strip()
        p = (row.get("passage_text") or "").strip()
        if not q or not p:
            continue

        negatives = []
        if buf:
            candidates = list(buf)
            rnd.shuffle(candidates)
            negatives = candidates[:negatives_per_query]

        if negatives:
            yield {"query": q, "pos": [p], "neg": negatives, "source": "miriad/miriad-4.4M"}
            emitted += 1
            if max_samples and emitted >= max_samples:
                break

        buf.append(p)
        if len(buf) > buffer_size:
            buf.pop(0)

# --------------------- stats & plots ---------------------
def summarize_triplets(triplets: List[Dict], out_dir: Path, tokenizer_name: str = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(triplets)
    by_src = Counter([t["source"] for t in triplets])
    neg_counts = [len(t["neg"]) for t in triplets]
    pos_counts = [len(t["pos"]) for t in triplets]
    report = {
        "total_examples": total,
        "by_source": dict(by_src),
        "neg_per_query": {"min": int(np.min(neg_counts) if neg_counts else 0),
                          "max": int(np.max(neg_counts) if neg_counts else 0),
                          "mean": float(np.mean(neg_counts) if neg_counts else 0)},
        "pos_per_query": {"min": int(np.min(pos_counts) if pos_counts else 0),
                          "max": int(np.max(pos_counts) if pos_counts else 0),
                          "mean": float(np.mean(pos_counts) if pos_counts else 0)},
    }
    (out_dir / "dataset_report.json").write_text(jsonlib.dumps(report, indent=2))
    # write a small sample for sanity
    sample_n = min(5, total)
    (out_dir / "samples.json").write_text(jsonlib.dumps(triplets[:sample_n], indent=2))

    # Optional token length histograms
    if tokenizer_name and HAS_MPL:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            q_lens, p_lens = [], []
            for t in tqdm(triplets[: min(20000, total)], desc="Token length sampling"):
                q = t["query"]
                qs = tok(q, add_special_tokens=False, truncation=True, max_length=1024)
                q_lens.append(len(qs["input_ids"]))
                for d in (t["pos"] + t["neg"])[:4]:
                    ds = tok(d, add_special_tokens=False, truncation=True, max_length=2048)
                    p_lens.append(len(ds["input_ids"]))
            plt.figure(); plt.hist(q_lens, bins=50); plt.title("Query token lengths"); plt.xlabel("tokens"); plt.ylabel("count")
            plt.savefig(out_dir / "hist_query_tokens.png", bbox_inches="tight"); plt.close()
            plt.figure(); plt.hist(p_lens, bins=50); plt.title("Passage token lengths"); plt.xlabel("tokens"); plt.ylabel("count")
            plt.savefig(out_dir / "hist_passage_tokens.png", bbox_inches="tight"); plt.close()
        except Exception as e:
            print(f"[viz] Skipping token length histograms: {e}", flush=True)

# --------------------- training (FlagEmbedding) ---------------------
def launch_flagembedding_train(
    train_path: Path,
    out_dir: Path,
    cache_dir: Path,
    model_name="BAAI/bge-reranker-v2-gemma",
    nproc=1,
    epochs=1,
    lr=2e-5,
    train_bs=1,
    grad_accum=16,
    query_max_len=256,
    passage_max_len=1024,
    lora_rank=64,
    lora_alpha=16,
    save_merged=True,
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    deepspeed_config=None,
    report_to="tensorboard",
    run_name=None,
    save_steps=2000,
    logging_steps=50,
    warmup_ratio=0.05,
    weight_decay=0.0,
):
    """
    Uses the documented decoder-only reranker finetuner:
      python -m FlagEmbedding.finetune.reranker.decoder_only.base ...
    Data format: JSONL with fields query: str, pos: List[str], neg: List[str].
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "FlagEmbedding.finetune.reranker.decoder_only.base",
        "--model_name_or_path", model_name,
        "--output_dir", str(out_dir),
        "--train_data", str(train_path),
        "--per_device_train_batch_size", str(train_bs),
        "--gradient_accumulation_steps", str(grad_accum),
        "--num_train_epochs", str(epochs),
        "--learning_rate", str(lr),
        "--logging_steps", str(logging_steps),
        "--save_strategy", "steps",
        "--save_steps", str(save_steps),
        "--save_total_limit", "2",
        "--warmup_ratio", str(warmup_ratio),
        "--lr_scheduler_type", "linear",
        "--dataloader_num_workers", "2",
        "--overwrite_output_dir",
        "--query_max_len", str(query_max_len),
        "--passage_max_len", str(passage_max_len),
        "--use_lora",
        "--lora_rank", str(lora_rank),
        "--lora_alpha", str(lora_alpha),
        "--cache_dir", str(cache_dir),
        "--report_to", report_to,
        "--logging_strategy", "steps",
    ]
    if run_name: cmd += ["--run_name", run_name]
    if save_merged: cmd += ["--save_merged_lora_model"]
    if fp16: cmd += ["--fp16"]
    if bf16: cmd += ["--bf16"]
    if gradient_checkpointing: cmd += ["--gradient_checkpointing"]
    if weight_decay and weight_decay > 0: cmd += ["--weight_decay", str(weight_decay)]
    if deepspeed_config: cmd += ["--deepspeed", deepspeed_config]

    if nproc and int(nproc) > 1:
        # keep DDP robust
        cmd = ["torchrun", "--nproc_per_node", str(nproc), "--ddp_find_unused_parameters", "False"] + cmd

    print("\n[Train CMD]\n", " ".join(cmd), "\n", flush=True)
    subprocess.run(cmd, check=True)

# --------------------- evaluation ---------------------
def locate_model_dir(output_dir: Path):
    """
    Try common locations for a merged model.
    - Prefer a 'merged' directory if present.
    - Else use the output_dir itself if it contains a model.
    - Else pick the largest checkpoint.
    """
    candidates = []
    for name in ["merged", "merged_model", "merged_16bit", ""]:
        p = output_dir / name if name else output_dir
        if (p / "config.json").exists():
            candidates.append(p)
    if candidates:
        # prefer the deepest first
        candidates = sorted(candidates, key=lambda x: len(str(x)))
        return candidates[-1]

    # look for checkpoints
    ckpts = [Path(p) for p in glob.glob(str(output_dir / "checkpoint-*"))]
    if ckpts:
        ckpts.sort(key=lambda p: int(p.name.split("-")[-1]))
        return ckpts[-1]
    return output_dir  # last resort

def metrics_from_labels(scores: np.ndarray, labels: np.ndarray, ks=(1,3,5,10)) -> Dict[str, float]:
    """Compute IR metrics for a single query ranking."""
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    gains = labels_sorted
    dcg = np.cumsum((2**gains - 1) / np.log2(np.arange(2, len(gains)+2)))
    ideal = np.sort(labels)[::-1]
    idcg = np.cumsum((2**ideal - 1) / np.log2(np.arange(2, len(ideal)+2)))
    ndcg_at = {f"nDCG@{k}": float(dcg[min(k, len(dcg))-1] / max(idcg[min(k, len(idcg))-1], 1e-9)) for k in ks}
    # MRR@10 and MAP@10
    ranks = np.where(labels_sorted == 1)[0]
    mrr10 = 0.0
    if len(ranks) > 0:
        first_rank = ranks[0] + 1
        mrr10 = 1.0 / first_rank if first_rank <= 10 else 0.0
    # AP@10
    ap = 0.0; hit = 0; precisions = []
    for i in range(min(10, len(labels_sorted))):
        if labels_sorted[i] == 1:
            hit += 1
            precisions.append(hit / (i+1))
    if precisions:
        ap = float(np.mean(precisions))
    hit1 = 1.0 if (len(labels_sorted) > 0 and labels_sorted[0] == 1) else 0.0
    return {"MRR@10": mrr10, "MAP@10": ap, "Hit@1": hit1, **ndcg_at}

def evaluate_jsonl(jsonl_path: Path, model_dir: Path, batch_size=64, query_max_len=256, max_len=1024, use_fp16=True, cache_dir: Path=None):
    """
    Loads the (merged) reranker and evaluates on a triplets JSONL by ranking
    pos+neg per query. Returns overall and per-source metrics.
    """
    from FlagEmbedding.inference.reranker.decoder_only.base import BaseLLMReranker  # documented API
    reranker = BaseLLMReranker(
        str(model_dir),
        use_fp16=use_fp16,
        cache_dir=str(cache_dir) if cache_dir else None,
        batch_size=batch_size,
        query_max_length=query_max_len,
        max_length=max_len,
        normalize=False,
    )

    all_pairs, all_labels, sources = [], [], []
    # we group by query; evaluate in blocks to keep memory bounded
    group = []
    with gzip.open(jsonl_path, "rt", encoding="utf-8") if jsonl_path.suffix == ".gz" else open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = jsonlib.loads(line)
            q = row["query"]
            cand = row["pos"] + row["neg"]
            labels = [1]*len(row["pos"]) + [0]*len(row["neg"])
            pairs = [(q, c) for c in cand]
            group.append((pairs, np.array(labels, dtype=np.int32), row.get("source","unknown")))
            if len(group) >= 1024:
                yield from _eval_flush_group(group, reranker)
                group = []
    if group:
        yield from _eval_flush_group(group, reranker)

def _eval_flush_group(group, reranker):
    """Helper to compute scores and yield per-query metrics tuples."""
    flat_pairs = [p for pairs, _, _ in group for p in pairs]
    scores = reranker.compute_score(flat_pairs, batch_size=reranker.batch_size, max_length=reranker.max_length, query_max_length=reranker.query_max_length)
    # scatter back
    idx = 0
    for pairs, labels, src in group:
        k = len(pairs)
        s = np.array(scores[idx: idx+k], dtype=np.float32)
        idx += k
        yield s, labels, src

def run_eval_to_files(val_path: Path, test_path: Path, output_dir: Path, model_dir: Path, cache_dir: Path, qlen: int, plen: int):
    for split, path in [("val", val_path), ("test", test_path)]:
        if not path.exists(): 
            print(f"[eval] {split} missing at {path}, skipping.")
            continue
        per_src = defaultdict(lambda: Counter())
        agg = Counter()
        n = 0
        for scores, labels, src in evaluate_jsonl(path, model_dir, batch_size=128, query_max_len=qlen, max_len=plen, use_fp16=True, cache_dir=cache_dir):
            m = metrics_from_labels(scores, labels, ks=(1,3,5,10))
            agg.update(m)
            per_src[src].update(m)
            n += 1
        # average
        def avg(counter, denom):
            return {k: float(v)/max(denom,1) for k,v in counter.items()}
        res = {"n_queries": n, "overall": avg(agg, n), "by_source": {k: avg(v, n) for k, v in per_src.items()}}
        fp = output_dir / f"eval_{split}.json"
        fp.write_text(jsonlib.dumps(res, indent=2))
        print(f"[eval] {split} metrics → {fp}", flush=True)

# --------------------- main ---------------------
def main():
    p = argparse.ArgumentParser()
    # paths
    p.add_argument("--scidocs_dir", default="data/scidocs", type=str)
    p.add_argument("--bioasq_jsonl", default="data/bioasq/train.jsonl.gz", type=str)
    p.add_argument("--miriad_glob",  default="data/miriad/data/train-*.parquet", type=str)
    p.add_argument("--processed_dir", default="data/processed", type=str)
    p.add_argument("--cache_dir", default="data/hf_cache", type=str)
    p.add_argument("--output_dir", default="outputs/reranker-medical-gemma", type=str)

    # sampling / negatives
    p.add_argument("--scidocs_neg", type=int, default=4)
    p.add_argument("--bioasq_neg",  type=int, default=4)
    p.add_argument("--miriad_neg",  type=int, default=4)
    p.add_argument("--scidocs_max_queries", type=int, default=None)
    p.add_argument("--bioasq_max_samples",  type=int, default=500000)  # keep RAM sane
    p.add_argument("--miriad_max_samples",  type=int, default=350000)
    p.add_argument("--bioasq_buffer", type=int, default=100000)
    p.add_argument("--miriad_buffer", type=int, default=200000)

    # split
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--test_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)

    # training cfg
    p.add_argument("--model_name", default="BAAI/bge-reranker-v2-gemma")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--train_bs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--query_max_len", type=int, default=256)
    p.add_argument("--passage_max_len", type=int, default=1024)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--save_merged", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--deepspeed_config", type=str, default=None)
    p.add_argument("--report_to", type=str, default="tensorboard", choices=["tensorboard","wandb","csv","none"])
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--warmup_ratio", type=float, default=0.05)

    # control
    p.add_argument("--no_train", action="store_true")
    p.add_argument("--no_eval", action="store_true")

    args = p.parse_args()

    # env
    set_env(args.cache_dir)
    disable_caching()
    random.seed(args.seed); np.random.seed(args.seed)

    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Build triplets ----------
    triplets = []

    # SciDocs
    print("[*] Loading SciDocs (local) …", flush=True)
    scidocs_y = list(load_scidocs_local(
        Path(args.scidocs_dir),
        negatives_per_query=args.scidocs_neg,
        max_queries=args.scidocs_max_queries,
    )) or []
    if scidocs_y:
        triplets.extend(scidocs_y)
        print(f"    SciDocs triplets: {len(scidocs_y)}", flush=True)
    else:
        print("    SciDocs skipped (qrels not found or zero pairs).", flush=True)

    # BioASQ-generated-queries
    print("[*] Loading BioASQ-generated-queries (local) …", flush=True)
    bioasq_iter = load_bioasq_generated_local(
        Path(args.bioasq_jsonl),
        negatives_per_query=args.bioasq_neg,
        max_samples=args.bioasq_max_samples,
        buffer_size=args.bioasq_buffer,
        seed=args.seed
    )
    bioasq_y = list(bioasq_iter) if bioasq_iter else []
    if bioasq_y:
        triplets.extend(bioasq_y)
        print(f"    BioASQ triplets: {len(bioasq_y)}", flush=True)
    else:
        print("    BioASQ-generated-queries skipped or empty.", flush=True)

    # MIRIAD
    print("[*] Loading MIRIAD parquet (local) …", flush=True)
    miriad_iter = load_miriad_local(
        args.miriad_glob,
        negatives_per_query=args.miriad_neg,
        max_samples=args.miriad_max_samples,
        buffer_size=args.miriad_buffer,
        seed=args.seed
    )
    miriad_y = list(miriad_iter) if miriad_iter else []
    if miriad_y:
        triplets.extend(miriad_y)
        print(f"    MIRIAD triplets: {len(miriad_y)}", flush=True)
    else:
        print("    MIRIAD skipped or empty.", flush=True)

    total = len(triplets)
    print(f"[*] Merged triplets: {total}", flush=True)
    if total == 0:
        print("No training pairs available. Ensure at least one dataset produced triplets.", flush=True)
        sys.exit(1)

    # ---------- Split & save ----------
    train, val, test = split_list(triplets, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
    out_train = processed_dir / "medical_reranker_train.jsonl.gz"
    out_val   = processed_dir / "medical_reranker_val.jsonl.gz"
    out_test  = processed_dir / "medical_reranker_test.jsonl.gz"
    write_jsonl(out_train, train)
    write_jsonl(out_val,   val)
    write_jsonl(out_test,  test)
    print(f"[*] Wrote:\n  {out_train}\n  {out_val}\n  {out_test}", flush=True)

    # ---------- Dataset report & plots ----------
    report_dir = Path(args.output_dir) / "reports"
    summarize_triplets(triplets, report_dir, tokenizer_name=args.model_name)

    # ---------- Train ----------
    if not args.no_train:
        run_name = args.run_name or f"ozstar-medical-gemma-{datetime.now().strftime('%Y%m%d_%H%M')}"
        launch_flagembedding_train(
            train_path=out_train,
            out_dir=Path(args.output_dir),
            cache_dir=Path(args.cache_dir),
            model_name=args.model_name,
            nproc=args.nproc,
            epochs=args.epochs,
            lr=args.lr,
            train_bs=args.train_bs,
            grad_accum=args.grad_accum,
            query_max_len=args.query_max_len,
            passage_max_len=args.passage_max_len,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            save_merged=args.save_merged,
            fp16=args.fp16,
            bf16=args.bf16,
            gradient_checkpointing=True,
            deepspeed_config=args.deepspeed_config,
            report_to=args.report_to,
            run_name=run_name,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
        )
    else:
        print("[*] --no_train set: triplets prepared, training skipped.", flush=True)

    # ---------- Eval (post-train) ----------
    if not args.no_eval:
        model_dir = locate_model_dir(Path(args.output_dir))
        print(f"[*] Evaluating with model_dir={model_dir}", flush=True)
        run_eval_to_files(out_val, out_test, Path(args.output_dir), model_dir, Path(args.cache_dir),
                          qlen=args.query_max_len, plen=args.passage_max_len)
    else:
        print("[*] --no_eval set: evaluation skipped.", flush=True)

if __name__ == "__main__":
    main()
