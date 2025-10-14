#!/usr/bin/env python3
import os, sys, json, math, time, argparse, random
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from bert_score import score as bertscore

torch.set_float32_matmul_precision('high')

INSTR = (
"You are a careful clinical assistant. Answer the patient question using "
"general, authoritative medical knowledge. Be concise (<=150 words), avoid 
speculation. "
"If unsure, say: I don't know."
)

def norm_text(s):
    return " ".join((s or "").split()).strip()

def tok_f1(ref, cand):
    r = norm_text(ref).lower().split()
    c = norm_text(cand).lower().split()
    if not r and not c: return 1.0
    if not r or not c:  return 0.0
    r_counts, c_counts = {}, {}
    for w in r: r_counts[w] = r_counts.get(w, 0) + 1
    for w in c: c_counts[w] = c_counts.get(w, 0) + 1
    overlap = sum(min(r_counts.get(w,0), c_counts.get(w,0)) for w in 
set(r_counts.keys()) | set(c_counts.keys()))
    prec = overlap / max(1, len(c))
    rec  = overlap / max(1, len(r))
    if prec+rec == 0: return 0.0
    return 2*prec*rec/(prec+rec)

def ngram_precision(ref, cand, n=1):
    r = norm_text(ref).lower().split()
    c = norm_text(cand).lower().split()
    if len(c) < n: return 0.0
    def ngrams(x, n): return [" ".join(x[i:i+n]) for i in 
range(len(x)-n+1)]
    rset = set(ngrams(r, n))
    cgrams = ngrams(c, n)
    if not cgrams: return 0.0
    hit = sum(1 for g in cgrams if g in rset)
    return hit / len(cgrams)

@torch.inference_mode()
def load_model(path, bf16_ok=True):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, 
use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or "<|pad|>"
    tok.padding_side = "left"
    tok.truncation_side = "left"
    dtype = torch.bfloat16 if (bf16_ok and torch.cuda.is_bf16_supported()) 
else (torch.float16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", 
torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
    model.generation_config.pad_token_id = tok.pad_token_id
    return tok, model

@torch.inference_mode()
def generate_batch(tok, model, prompts, max_new_tokens=256):
    # keep input within context
    ctx = getattr(model.config, "max_position_embeddings", 
getattr(model.config, "n_positions", 4096))
    max_inp = max(128, int(ctx) - int(max_new_tokens) - 8)
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, 
max_length=max_inp).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,              # greedy for stable eval
        temperature=None,
        use_cache=True,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        return_dict_in_generate=True
    )
    full = tok.batch_decode(out.sequences, skip_special_tokens=True)
    pre  = tok.batch_decode(enc.input_ids,   skip_special_tokens=True)
    return [f[len(p):].strip() for f, p in zip(full, pre)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-jsonl", required=True, 
help="data/medquad/processed/medquad_clean.jsonl")
    ap.add_argument("--model-dirs", nargs="+", required=True, help="e.g., 
model/medalpaca-7b model/medgemma-27b-text-it")
    ap.add_argument("--outdir", default="data/medquad/runs")
    ap.add_argument("--max-samples", type=int, default=5000, help="cap for 
speed; set higher to run full")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # Load dataset
    rows = []
    with open(args.data_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append({"id": obj["id"], "q": obj["question"], "a": 
obj["answer"]})
    random.shuffle(rows)
    if args.max_samples and args.max_samples < len(rows):
        rows = rows[:args.max_samples]

    # Prepare metric tools
    rscorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # Evaluate each model
    summary_rows = []
    for mpath in args.model_dirs:
        mname = os.path.basename(mpath.rstrip("/"))
        print(f"\n=== Evaluating {mname} on {len(rows)} items ===")

        tok, model = load_model(mpath)
        preds, refs, qids = [], [], []

        # Batched generation
        for i in tqdm(range(0, len(rows), args.batch_size)):
            batch = rows[i:i+args.batch_size]
            prompts = [
                f"{INSTR}\n\nQuestion: {norm_text(x['q'])}\n\nAnswer:"
                for x in batch
            ]
            outs = generate_batch(tok, model, prompts, 
max_new_tokens=args.max_new_tokens)
            for b, gen in zip(batch, outs):
                preds.append(norm_text(gen))
                refs.append(norm_text(b["a"]))
                qids.append(b["id"])

        # Metrics: Rouge-L, token F1, n-gram precision
        rougeL_f, tokF1, uniP, biP = [], [], [], []
        for ref, hyp in zip(refs, preds):
            r = rscorer.score(ref, hyp)["rougeL"]
            rougeL_f.append(r.fmeasure)
            tokF1.append(tok_f1(ref, hyp))
            uniP.append(ngram_precision(ref, hyp, 1))
            biP.append(ngram_precision(ref, hyp, 2))

        # BERTScore (batched)
        P, R, F = bertscore(preds, refs, lang="en", 
rescale_with_baseline=True)
        bsf = F.tolist()

        df = pd.DataFrame({
            "id": qids, "ref": refs, "pred": preds,
            "rougeL_f": rougeL_f, "bert_f": bsf,
            "tok_f1": tokF1, "uni_prec": uniP, "bi_prec": biP
        })
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_csv = os.path.join(args.outdir, 
f"{mname}_medquad_{stamp}.csv")
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[saved] {out_csv}")

        # Summary
        summ = {
            "model": mname,
            "n": len(df),
            "rougeL_f_mean": float(df["rougeL_f"].mean()),
            "bert_f_mean": float(df["bert_f"].mean()),
            "tok_f1_mean": float(df["tok_f1"].mean()),
            "uni_prec_mean (halluc-proxy)": float(df["uni_prec"].mean()),
            "bi_prec_mean (halluc-proxy)": float(df["bi_prec"].mean()),
            "detail_csv": out_csv
        }
        summary_rows.append(summ)

        # free VRAM between models
        del model; del tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write overall summary
    summary_df = pd.DataFrame(summary_rows)
    out_sum = os.path.join(args.outdir, 
f"SUMMARY_{time.strftime('%Y%m%d-%H%M%S')}.csv")
    summary_df.to_csv(out_sum, index=False)
    print("\n=== SUMMARY ===")
    print(summary_df)
    print(f"[saved] {out_sum}")

if __name__ == "__main__":
    main()

