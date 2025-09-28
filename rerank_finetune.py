# scripts/rerank_finetune.py
# Finetune BAAI/bge-reranker-v2-gemma on SciDocs, BioASQ-gen-queries, MIRIAD.
# Runs end-to-end: prepare JSONL triplets + launch FlagEmbedding trainer.
# Python 3.10+, pip: datasets, FlagEmbedding[finetune], ujson (optional), numpy

import os, sys, json, random, argparse, math, subprocess, itertools, gzip
from pathlib import Path

try:
    import ujson as jsonlib
except Exception:
    import json as jsonlib

import numpy as np
from datasets import load_dataset, disable_caching

# ---------- Utils ----------
def safe_join_title_text(title, text):
    title = (title or "").strip()
    text  = (text or "").strip()
    if title and text and not text.lower().startswith(title.lower()):
        return f"{title}\n\n{text}"
    return title or text

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with (gzip.open(path, "wt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "w", encoding="utf-8")) as f:
        for r in rows:
            f.write(jsonlib.dumps(r, ensure_ascii=False) + "\n")

def set_env(cache_dir):
    os.environ.setdefault("HF_HOME", str(Path(cache_dir).resolve()))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(cache_dir).resolve()))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------- BEIR helpers (SciDocs, BioASQ-generated-queries) ----------
def load_beir_triplets(dataset_name, cache_dir, negatives_per_query=4, max_queries=None, rng=None):
    """
    Returns generator of JSONL lines: {query, pos: [doc...], neg: [doc...], source}
    """
    ds = load_dataset(dataset_name, cache_dir=cache_dir)
    assert all(k in ds for k in ("corpus", "queries", "qrels")), f"{dataset_name} must have corpus/queries/qrels splits"

    # Build doc map
    doc_map = {}
    for row in ds["corpus"]:
        did = row["_id"]
        doc_map[did] = safe_join_title_text(row.get("title", ""), row.get("text", ""))

    # Group positives by query
    pos_ids = {}
    for row in ds["qrels"]:
        qid = row["_id"]
        did = row["doc_id"] if "doc_id" in row else row.get("docid") or row.get("_id_doc") or row.get("doc")
        score = row.get("score", 1)
        if score > 0 and did in doc_map:
            pos_ids.setdefault(qid, set()).add(did)

    all_doc_ids = list(doc_map.keys())
    rng = rng or random.Random(42)

    count = 0
    for q in ds["queries"]:
        if max_queries and count >= max_queries:
            break
        qid = q["_id"]
        qtext = (q.get("text") or "").strip()
        if not qtext:
            continue
        pids = list(pos_ids.get(qid, []))
        if not pids:
            continue

        positives = [doc_map[pid] for pid in pids if pid in doc_map][:3]  # cap a few pos per query
        # sample negatives randomly (fast & scalable); light-hardness by sampling from nearby ids
        negs = set()
        tries = 0
        while len(negs) < negatives_per_query and tries < negatives_per_query * 10:
            did = rng.choice(all_doc_ids)
            if did not in pids:
                negs.add(did)
            tries += 1
        negatives = [doc_map[d] for d in negs]
        if positives and negatives:
            yield {"query": qtext, "pos": positives, "neg": negatives, "source": dataset_name}
            count += 1

# ---------- MIRIAD helpers ----------
def load_miriad_triplets(cache_dir, negatives_per_query=4, max_samples=300000, buffer_size=8192, rng=None):
    """
    Stream MIRIAD (4.4M) and produce triplets with in-buffer random negatives.
    """
    ds = load_dataset("miriad/miriad-4.4M", split="train", cache_dir=cache_dir, streaming=True)
    rng = rng or random.Random(42)

    buffer = []
    n_emitted = 0
    for row in ds:
        q = (row.get("question") or "").strip()
        p = (row.get("passage_text") or "").strip()
        if not q or not p:
            continue

        # Negatives from buffer
        cand_docs = [x for x in buffer if x != p]
        rng.shuffle(cand_docs)
        negatives = cand_docs[:negatives_per_query]

        if len(negatives) < negatives_per_query and buffer:
            # top up with random samples (allow repeats)
            while len(negatives) < negatives_per_query:
                negatives.append(rng.choice(buffer))

        if negatives:
            yield {"query": q, "pos": [p], "neg": negatives, "source": "miriad/miriad-4.4M"}
            n_emitted += 1

        # push to buffer
        buffer.append(p)
        if len(buffer) > buffer_size:
            buffer.pop(0)

        if max_samples and n_emitted >= max_samples:
            break

# ---------- Merge & split ----------
def split_shards(items, val_ratio=0.05, test_ratio=0.05, seed=42):
    rng = random.Random(seed)
    xs = list(items)
    rng.shuffle(xs)
    n = len(xs)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    val = xs[:n_val]
    test = xs[n_val:n_val + n_test]
    train = xs[n_val + n_test:]
    return train, val, test

# ---------- Training launcher (FlagEmbedding) ----------
def launch_training(
    train_path, dev_path, out_dir, cache_dir, model_name="BAAI/bge-reranker-v2-gemma",
    nproc=1, epochs=1, lr=2e-5, train_bs=1, eval_bs=1, grad_accum=16,
    query_max_len=256, passage_max_len=1024, lora_rank=64, lora_alpha=16,
    save_merged=True, fp16=True, gc=True, deepspeed_config=None
):
    """
    Calls the official FlagEmbedding finetuner for decoder-only rerankers.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    base = [
        sys.executable, "-m", "FlagEmbedding.llm_reranker.finetune_for_instruction.run",
        "--model_name_or_path", model_name,
        "--output_dir", out_dir,
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
        # data args
        "--query_max_len", str(query_max_len),
        "--passage_max_len", str(passage_max_len),
        # model args (LoRA etc.)
        "--use_lora",
        "--lora_rank", str(lora_rank),
        "--lora_alpha", str(lora_alpha),
        "--save_merged_lora_model" if save_merged else "",
        "--cache_dir", str(cache_dir),
    ]
    if fp16: base += ["--fp16"]
    if gc:   base += ["--gradient_checkpointing"]

    if deepspeed_config:
        base += ["--deepspeed", deepspeed_config]

    # Filter empty args
    base = [x for x in base if x != ""]

    if nproc and int(nproc) > 1:
        cmd = ["torchrun", "--nproc_per_node", str(nproc)] + base
    else:
        cmd = base

    print("\n[Training command]\n", " ".join(cmd), "\n", flush=True)
    # Launch
    subprocess.run(cmd, check=True)

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="data/hf_cache", type=str)
    parser.add_argument("--processed_dir", default="data/processed", type=str)
    parser.add_argument("--output_dir", default="outputs/reranker-medical-gemma", type=str)

    # sampling / balance
    parser.add_argument("--beir_negatives_per_query", type=int, default=4)
    parser.add_argument("--miriad_negatives_per_query", type=int, default=4)
    parser.add_argument("--scidocs_max_queries", type=int, default=None)  # e.g., 10000 for quick runs
    parser.add_argument("--bioasq_max_queries", type=int, default=None)
    parser.add_argument("--miriad_max_samples", type=int, default=350000)  # ~0.35M pairs → good LoRA run

    # split
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    # trainer settings
    parser.add_argument("--model_name", default="BAAI/bge-reranker-v2-gemma")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_bs", type=int, default=1)
    parser.add_argument("--eval_bs", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--query_max_len", type=int, default=256)
    parser.add_argument("--passage_max_len", type=int, default=1024)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--save_merged", action="store_true")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--no_train", action="store_true")
    parser.add_argument("--deepspeed_config", type=str, default=None)

    args = parser.parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    set_env(args.cache_dir)
    disable_caching()  # prevent ~/.cache if HPC nodes are read-only

    processed = Path(args.processed_dir)
    processed.mkdir(parents=True, exist_ok=True)

    # 1) Build triplets
    print("Loading & converting BEIR/SciDocs …", flush=True)
    scidocs = list(load_beir_triplets(
        "BeIR/scidocs",
        cache_dir=args.cache_dir,
        negatives_per_query=args.beir_negatives_per_query,
        max_queries=args.scidocs_max_queries,
        rng=rng
    ))

    print("Loading & converting BEIR/BioASQ-generated-queries …", flush=True)
    bioasq = list(load_beir_triplets(
        "BeIR/bioasq-generated-queries",
        cache_dir=args.cache_dir,
        negatives_per_query=args.beir_negatives_per_query,
        max_queries=args.bioasq_max_queries,
        rng=rng
    ))

    print("Streaming MIRIAD 4.4M (sample) …", flush=True)
    miriad_gen = load_miriad_triplets(
        cache_dir=args.cache_dir,
        negatives_per_query=args.miriad_negatives_per_query,
        max_samples=args.miriad_max_samples,
        rng=rng
    )
    miriad = list(miriad_gen)

    # 2) Merge & shuffle (balance roughly across datasets)
    def take_balanced(a, b, c):
        # keep ratios roughly even across sources
        n = min(len(a), len(b), len(c))
        if n == 0:
            return a + b + c
        take_each = n
        xs = a[:take_each] + b[:take_each] + c[:take_each]
        # append remaining (cap to ~3 * take_each * 1.5)
        rest = a[take_each:] + b[take_each:] + c[take_each:]
        rng.shuffle(rest)
        cap = int(take_each * 4.5)
        xs += rest[:cap]
        rng.shuffle(xs)
        return xs

    merged = take_balanced(scidocs, bioasq, miriad)
    print(f"Merged samples: {len(merged)}  (SciDocs={len(scidocs)}  BioASQ={len(bioasq)}  MIRIAD={len(miriad)})")

    # 3) Split
    train, val, test = split_shards(merged, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
    print(f"Splits → train={len(train)}  val={len(val)}  test={len(test)}")

    # 4) Save
    train_path = processed / "medical_reranker_train.jsonl.gz"
    val_path   = processed / "medical_reranker_val.jsonl.gz"
    test_path  = processed / "medical_reranker_test.jsonl.gz"
    write_jsonl(train_path, train)
    write_jsonl(val_path,   val)
    write_jsonl(test_path,  test)
    print(f"Wrote:\n  {train_path}\n  {val_path}\n  {test_path}", flush=True)

    # 5) Train (FlagEmbedding finetuner)
    if not args.no_train:
        launch_training(
            train_path=train_path,
            dev_path=val_path,
            out_dir=args.output_dir,
            cache_dir=args.cache_dir,
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
            fp16=True,   # OzSTAR P100 lacks bf16 — keep fp16
            gc=True,
            deepspeed_config=args.deepspeed_config
        )
    else:
        print("\nSkipping training (--no_train). Use the command above to run.", flush=True)

if __name__ == "__main__":
    main()
