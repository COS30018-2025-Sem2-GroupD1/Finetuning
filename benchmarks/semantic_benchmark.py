#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
semantic_benchmark.py

Offline semantic evaluation of a fine-tuned model on healthcaremagic.jsonl
using a local embedding model (MedEmbed) for cosine similarity, plus optional
lexical token-overlap as a stricter gate.

Directory layout (example):
  data/healthcaremagic.jsonl
  embedding/MedEmbed-large-v0.1
  checkpoints/<your-finetuned-model>
  scripts/semantic_benchmark.py
  results/semantic_eval/  (auto-created)

Usage:
  python scripts/semantic_benchmark.py \
    --data-file data/healthcaremagic.jsonl \
    --embed-model-dir embedding/MedEmbed-large-v0.1 \
    --gen-model-dir checkpoints/medalpaca_pubmedqa_lora \
    --out-dir results/semantic_eval \
    --threshold 0.82 \
    --strict-overlap 0.10 \
    --max-samples 1000

Notes:
- The script auto-detects common field names for question/context/answer/id.
- It saves an embedding index for the reference answers to {data-file stem}_index.npz (+meta).
"""

import os, sys, json, argparse, time, math, re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch

# -------------------------------
# Utility: robust JSONL reader
# -------------------------------
def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

# -------------------------------
# Field autodetection (healthcaremagic)
# -------------------------------
QUESTION_KEYS = ["question", "query", "prompt", "user_question", "patient", "input"]
ANSWER_KEYS   = ["answer", "doctor_answer", "response", "output", "assistant", "gold"]
CONTEXT_KEYS  = ["context", "history", "background", "case", "notes"]
ID_KEYS       = ["id", "_id", "uid", "q_id", "qid"]

def pick_first(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None:
            val = d[k]
            if isinstance(val, (str, int)):
                return str(val)
    return None

def extract_fields(obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    q  = pick_first(obj, QUESTION_KEYS)
    a  = pick_first(obj, ANSWER_KEYS)
    cx = pick_first(obj, CONTEXT_KEYS)
    _id = pick_first(obj, ID_KEYS)
    return q, a, cx, _id

# -------------------------------
# Simple token overlap (Jaccard)
# -------------------------------
DEFAULT_STOP = {
    "the","a","an","and","or","to","of","in","on","for","with","as","by","is","are","was","were",
    "it","this","that","these","those","at","be","from","about","into","you","your","we","our"
}

def tokens(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z]+", text or "") if t]

def jaccard(a: str, b: str, stop=DEFAULT_STOP) -> float:
    ta = [t for t in tokens(a) if t not in stop]
    tb = [t for t in tokens(b) if t not in stop]
    sa, sb = set(ta), set(tb)
    if not sa and not sb: return 1.0
    return len(sa & sb) / max(1, len(sa | sb))

# -------------------------------
# Embedding model loader
# - Tries sentence_transformers first.
# - Falls back to HF AutoModel+mean pooling if needed.
# -------------------------------
def load_embedding_backend(model_dir: Path, device: torch.device):
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(str(model_dir), device=str(device))
        dim = m.get_sentence_embedding_dimension()

        def embed(texts: List[str], batch_size: int = 64) -> np.ndarray:
            return m.encode(texts,
                            batch_size=batch_size,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            show_progress_bar=False)
        backend = "sentence-transformers"
        return embed, dim, backend
    except Exception as e:
        # transformers fallback
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        mdl = AutoModel.from_pretrained(str(model_dir)).to(device)
        mdl.eval()
        dim = mdl.config.hidden_size

        def embed(texts: List[str], batch_size: int = 16) -> np.ndarray:
            vecs = []
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    enc = tok(batch, truncation=True, padding=True, return_tensors="pt", max_length=512).to(device)
                    out = mdl(**enc)
                    last = out.last_hidden_state         # [B, T, H]
                    mask = enc["attention_mask"].unsqueeze(-1).expand_as(last).float()
                    summed = (last * mask).sum(dim=1)    # [B, H]
                    denom = mask.sum(dim=1).clamp(min=1e-6)
                    meanp = summed / denom
                    meanp = torch.nn.functional.normalize(meanp, p=2, dim=1)
                    vecs.append(meanp.cpu().numpy())
            return np.vstack(vecs)
        backend = "transformers"
        return embed, dim, backend

# -------------------------------
# Generator (fine-tuned model)
# -------------------------------
def load_generator(gen_model_dir: Path, device: torch.device):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(str(gen_model_dir), use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # dtype choice: prefer bfloat16 on A100-class; fallback fp16; else float32
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(str(gen_model_dir),
                                                 device_map="auto",
                                                 torch_dtype=torch_dtype)
    model.config.pad_token_id = tok.pad_token_id
    return tok, model

def build_prompt(question: str, context: Optional[str]) -> str:
    instr = "Answer the patient's question accurately and concisely. Include a brief clinical rationale."
    inp = f"Question: {question.strip()}\n"
    if context and context.strip():
        inp += f"Context:\n{context.strip()}\n"
    return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n### Response:\n"

@torch.no_grad()
def generate_answer(tok, model, prompt: str, max_new_tokens=256) -> str:
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc,
                         max_new_tokens=max_new_tokens,
                         do_sample=False,
                         temperature=0.0,
                         repetition_penalty=1.0,
                         pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id)
    full = tok.decode(out[0], skip_special_tokens=True)
    pre  = tok.decode(enc.input_ids[0], skip_special_tokens=True)
    resp = full[len(pre):].strip()
    return resp

# -------------------------------
# Cosine similarity (unit vectors)
# -------------------------------
def cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    # inputs expected normalized; still safe-guard
    u = u / max(1e-9, np.linalg.norm(u))
    v = v / max(1e-9, np.linalg.norm(v))
    return float(np.dot(u, v))

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", required=True, help="JSONL with question/answer/context (healthcaremagic.jsonl)")
    ap.add_argument("--embed-model-dir", required=True, help="Local dir for MedEmbed (e.g., embedding/MedEmbed-large-v0.1)")
    ap.add_argument("--gen-model-dir", required=True, help="Local dir for fine-tuned generator (e.g., checkpoints/...)")
    ap.add_argument("--out-dir", required=True, help="Where to write logs/csv")
    ap.add_argument("--index-prefix", default=None, help="Optional prefix for saved index files (defaults to data-file stem)")
    ap.add_argument("--threshold", type=float, default=0.82, help="Cosine sim threshold to count as correct")
    ap.add_argument("--strict-overlap", type=float, default=0.10, help="Min token Jaccard overlap to also require (set 0 to disable)")
    ap.add_argument("--batch-embed", type=int, default=64, help="Embedding batch size")
    ap.add_argument("--max-samples", type=int, default=None, help="Limit examples for quick run")
    ap.add_argument("--skip-index", action="store_true", help="Skip writing the embedding index")
    args = ap.parse_args()

    data_path = Path(args.data_file)
    out_dir   = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    index_base = args.index_prefix or data_path.stem
    index_npz  = data_path.parent / f"{index_base}_index.npz"
    index_meta = data_path.parent / f"{index_base}_index_meta.json"
    index_meta_jl = data_path.parent / f"{index_base}_index_meta.jsonl"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    embed_fn, emb_dim, backend = load_embedding_backend(Path(args.embed_model_dir), device)
    tok, gen_model = load_generator(Path(args.gen_model_dir), device)

    # Read data and collect fields
    rows = []
    for i, obj in enumerate(read_jsonl(data_path)):
        q, a, cx, _id = extract_fields(obj)
        if not (q and a):
            continue
        rid = _id if _id is not None else str(i)
        rows.append({"id": rid, "question": q, "answer": a, "context": cx})
        if args.max_samples and len(rows) >= args.max_samples:
            break

    if not rows:
        print("No valid rows found; check field names in JSONL.")
        sys.exit(1)

    # ----------------- Build/snapshot embedding index for reference answers -----------------
    ref_texts = [r["answer"] for r in rows]
    t0 = time.time()
    ref_emb = embed_fn(ref_texts, batch_size=args.batch_embed)
    t1 = time.time()
    assert ref_emb.shape[0] == len(rows), "Embedding count mismatch"
    print(f"Embedded {len(rows)} reference answers in {t1 - t0:.1f}s (dim={ref_emb.shape[1]}, backend={backend}).")

    if not args.skip_index:
        np.savez_compressed(index_npz, embeddings=ref_emb.astype(np.float32), ids=np.array([r["id"] for r in rows]))
        meta = {
            "source_file": str(data_path),
            "n_items": len(rows),
            "embedding_dim": int(ref_emb.shape[1]),
            "backend": backend,
            "embed_model_dir": str(Path(args.embed_model_dir).resolve()),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with index_meta.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with index_meta_jl.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "question": r["question"]}, ensure_ascii=False) + "\n")
        print(f"Saved index: {index_npz}")
        print(f"Saved index meta: {index_meta}, {index_meta_jl}")

    # ----------------- Generate answers & score -----------------
    csv_path = out_dir / "semantic_eval.csv"
    log_path = out_dir / "semantic_eval.log.txt"
    summary_path = out_dir / "summary.txt"

    total = len(rows)
    n_pass = 0
    sims = []
    overlaps = []

    with csv_path.open("w", encoding="utf-8") as fcsv, log_path.open("w", encoding="utf-8") as flog:
        # CSV header
        fcsv.write("id,cosine_sim,token_overlap,pass,ref_len,gen_len\n")

        GEN_KW = dict(max_new_tokens=256)
        for i, r in enumerate(rows, 1):
            prompt = build_prompt(r["question"], r.get("context"))
            gen = generate_answer(tok, gen_model, prompt, max_new_tokens=GEN_KW["max_new_tokens"])
            # Embed generated text
            gen_vec = embed_fn([gen], batch_size=1)[0]
            ref_vec = ref_emb[i-1]

            sim = cosine_sim(gen_vec, ref_vec)
            sims.append(sim)

            ov = jaccard(gen, r["answer"]) if args.strict_overlap > 0 else 1.0
            overlaps.append(ov)

            ok = (sim >= args.threshold) and (ov >= args.strict_overlap)
            n_pass += int(ok)

            # Logs
            flog.write(f"[{i}/{total}] id={r['id']}  sim={sim:.4f}  overlap={ov:.3f}  "
                       f"pass={ok}  | Q: {r['question'][:120].replace('\\n',' ')}\n")
            # CSV
            fcsv.write(f"{r['id']},{sim:.6f},{ov:.4f},{int(ok)},{len(r['answer'])},{len(gen)}\n")

    acc = n_pass / total
    sims_np = np.array(sims, dtype=float)
    ovs_np  = np.array(overlaps, dtype=float)

    with summary_path.open("w", encoding="utf-8") as fs:
        fs.write("Semantic Benchmark Summary\n")
        fs.write("==========================\n")
        fs.write(f"Data file: {data_path}\n")
        fs.write(f"Generator: {args.gen_model_dir}\n")
        fs.write(f"Embedder : {args.embed_model_dir} (backend={backend}, dim={emb_dim})\n\n")
        fs.write(f"Examples : {total}\n")
        fs.write(f"Threshold: cosine>={args.threshold:.2f}"
                 + (f" & token_overlap>={args.strict_overlap:.2f}\n" if args.strict_overlap>0 else " (no token-overlap gate)\n"))
        fs.write(f"Accuracy : {acc:.4%}\n\n")
        fs.write(f"Cosine similarity: mean={sims_np.mean():.4f}, median={np.median(sims_np):.4f}, "
                 f"min={sims_np.min():.4f}, max={sims_np.max():.4f}\n")
        if args.strict_overlap > 0:
            fs.write(f"Token overlap    : mean={ovs_np.mean():.3f}, median={np.median(ovs_np):.3f}\n")

    print(f"\nSaved:\n  {csv_path}\n  {log_path}\n  {summary_path}")
    print(f"Accuracy: {acc:.2%}  (threshold={args.threshold}, overlap>={args.strict_overlap})")

if __name__ == "__main__":
    main()
