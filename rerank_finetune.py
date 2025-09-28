#!/usr/bin/env python
# Finetune BAAI/bge-reranker-v2-gemma using local SciDocs (BEIR-style), BioASQ-generated-queries (jsonl),
# and MIRIAD 4.4M (parquet). No internet calls. Produces triplets JSONL and runs FlagEmbedding finetuner.
#
# Requires: pip install -U "FlagEmbedding[finetune]" datasets pyarrow ujson

import os, sys, gzip, glob, csv, random, argparse, json
from pathlib import Path

try:
    import ujson as jsonlib
except Exception:
    import json as jsonlib

import numpy as np
from datasets import load_dataset, disable_caching

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
    Reads BEIR-style local SciDocs:
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

    # locate a qrels tsv if available
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
            for qid, did, score, *_ in reader:
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
        while len(negs) < target and tries < target * 20:
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
    Each line typically has: {"_id": ..., "title": ..., "text": ..., "query": ...}
    Treat (query, title+text) as a positive; negatives sampled from a rolling buffer.
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

            # build negatives from buffer
            if buf:
                candidates = list(buf)
                rnd.shuffle(candidates)
                negatives = candidates[:negatives_per_query]
            else:
                negatives = []

            if negatives:
                yield {"query": q, "pos": [doc], "neg": negatives, "source": "bioasq-generated-queries"}
                emitted += 1
                if max_samples and emitted >= max_samples:
                    break

            # push to buffer
            buf.append(doc)
            if len(buf) > buffer_size:
                buf.pop(0)

def load_miriad_local(parquet_glob: str, negatives_per_query=4, max_samples=350000, buffer_size=200000, seed=42):
    """
    Reads local parquet shards for MIRIAD: columns include question, passage_text.
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

        # negatives from buffer
        if buf:
            candidates = list(buf)
            rnd.shuffle(candidates)
            negatives = candidates[:negatives_per_query]
        else:
            negatives = []

        if negatives:
            yield {"query": q, "pos": [p], "neg": negatives, "source": "miriad/miriad-4.4M"}
            emitted += 1
            if max_samples and emitted >= max_samples:
                break

        buf.append(p)
        if len(buf) > buffer_size:
            buf.pop(0)

# --------------------- training (FlagEmbedding) ---------------------
import subprocess
def launch_flagembedding_train(
    train_path: Path,
    dev_path: Path,
    out_dir: Path,
    cache_dir: Path,
    model_name="BAAI/bge-reranker-v2-gemma",
    nproc=1,
    epochs=1,
    lr=2e-5,
    train_bs=1,
    eval_bs=1,
    grad_accum=16,
    query_max_len=256,
    passage_max_len=1024,
    lora_rank=64,
    lora_alpha=16,
    save_merged=True,
    fp16=True,
    gradient_checkpointing=True,
    deepspeed_config=None
):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "FlagEmbedding.llm_reranker.finetune_for_instruction.run",
        "--model_name_or_path", model_name,
        "--output_dir", str(out_dir),
        "--train_data", str(train_path),
        "--dev_data", str(dev_path),
        "--per_device_train_batch_size", str(train_bs),
        "--per_device_eval_batch_size", str(eval_bs),
        "--gradient_accumulation_steps", str(grad_accum),
        "--num_train_epochs", str(epochs),
        "--learning_rate", str(lr),
        "--logging_steps", "50",
        "--evaluation_strategy", "steps",
        "--eval_steps", "500",
        "--save_strategy", "epoch",
        "--save_total_limit", "2",
        "--warmup_ratio", "0.05",
        "--lr_scheduler_type", "linear",
        "--dataloader_num_workers", "2",
        "--overwrite_output_dir",
        "--query_max_len", str(query_max_len),
        "--passage_max_len", str(passage_max_len),
        "--use_lora",
        "--lora_rank", str(lora_rank),
        "--lora_alpha", str(lora_alpha),
        "--cache_dir", str(cache_dir),
    ]
    if save_merged: cmd += ["--save_merged_lora_model"]
    if fp16: cmd += ["--fp16"]
    if gradient_checkpointing: cmd += ["--gradient_checkpointing"]
    if deepspeed_config: cmd += ["--deepspeed", deepspeed_config]

    if nproc and int(nproc) > 1:
        cmd = ["torchrun", "--nproc_per_node", str(nproc)] + cmd

    print("\n[Train CMD]\n", " ".join(cmd), "\n", flush=True)
    subprocess.run(cmd, check=True)

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
    p.add_argument("--eval_bs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--query_max_len", type=int, default=256)
    p.add_argument("--passage_max_len", type=int, default=1024)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--save_merged", action="store_true")
    p.add_argument("--nproc", type=int, default=1)
    p.add_argument("--deepspeed_config", type=str, default=None)
    p.add_argument("--no_train", action="store_true")

    args = p.parse_args()

    # env
    set_env(args.cache_dir)
    disable_caching()
    rnd = random.Random(args.seed)
    np.random.seed(args.seed)

    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Build triplets ----------
    triplets = []

    # SciDocs (requires qrels)
    print("[*] Loading SciDocs (local) …", flush=True)
    scidocs_y = list(load_scidocs_local(
        Path(args.scidocs_dir),
        negatives_per_query=args.scidocs_neg,
        max_queries=args.scidocs_max_queries,
        seed=args.seed
    ))
    if scidocs_y:
        triplets.extend(scidocs_y)
        print(f"    SciDocs triplets: {len(scidocs_y)}", flush=True)
    else:
        print("    SciDocs skipped (no qrels found or zero pairs).", flush=True)

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

    # Quick sanity
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

    # ---------- Train ----------
    if not args.no_train:
        launch_flagembedding_train(
            train_path=out_train,
            dev_path=out_val,
            out_dir=Path(args.output_dir),
            cache_dir=Path(args.cache_dir),
            model_name=args.model_name,
            nproc=args.nproc,
            epochs=args.epochs,
            lr=args.lr,
            train_bs=args.train_bs,
            eval_bs=args.eval_bs,
            grad_accum=args.grad_accum,
            query_max_len=args.query_max_len,
            passage_max_len=args.passage_max_len,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            save_merged=args.save_merged,
            fp16=True,
            gradient_checkpointing=True,
            deepspeed_config=args.deepspeed_config
        )
    else:
        print("[*] --no_train set: triplets prepared, training skipped.", flush=True)

if __name__ == "__main__":
    main()
