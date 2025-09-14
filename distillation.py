import os
import sys
import json
import gzip
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import GenerationConfig
except ImportError:
    from transformers.generation.configuration_utils import GenerationConfig

from tqdm import tqdm

# -------------------------------
# Robust JSONL I/O
# -------------------------------
def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)
    print("Load data successfully")
def parse_key_list(arg: Optional[str], default_list: List[str]) -> List[str]:
    if not arg:
        return default_list
    return [x.strip() for x in arg.split(",") if x.strip()]

# -------------------------------
# Field autodetection (overridable)
# -------------------------------
DEFAULT_QUESTION_KEYS = ["question", "Question", "query", "prompt", "user_question", "patient", "input"]
DEFAULT_ANSWER_KEYS   = ["answer", "Answer", "doctor_answer", "response", "Response", "output", "gold", "assistant"]
DEFAULT_CONTEXT_KEYS  = ["context", "Context", "history", "background", "case", "notes"]
DEFAULT_ID_KEYS       = ["id", "ID", "_id", "uid", "q_id", "qid"]

def pick_first(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None:
            v = d[k]
            if isinstance(v, (str, int)):
                return str(v)
    return None

def extract_fields_flat(
    obj: Dict[str, Any],
    q_keys: List[str],
    a_keys: List[str],
    c_keys: List[str],
    id_keys: List[str]
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    q  = pick_first(obj, q_keys)
    a  = pick_first(obj, a_keys)   # optional gold
    cx = pick_first(obj, c_keys)
    _id = pick_first(obj, id_keys)
    return q, a, cx, _id

def extract_fields_any(
    obj: Dict[str, Any],
    q_keys: List[str],
    a_keys: List[str],
    c_keys: List[str],
    id_keys: List[str],
    default_instruction: str
) -> Tuple[str, str, Optional[str], Optional[str], Optional[str]]:
    """
    Return (instruction, input_text, context, gold, id)
    Prefers nested 'sft' if present, else falls back to flat keys.
    """
    rid = obj.get("id")
    sft = obj.get("sft")
    if isinstance(sft, dict) and ("input" in sft or "instruction" in sft or "output" in sft):
        instruction = sft.get("instruction") or default_instruction
        input_text  = sft.get("input") or ""
        context     = None
        gold        = sft.get("output")
        if rid is None:
            rid = pick_first(obj, DEFAULT_ID_KEYS)
        return instruction, input_text, context, gold, (rid if rid is not None else None)

    # Flat fallback
    q, a, cx, rid2 = extract_fields_flat(obj, q_keys, a_keys, c_keys, id_keys)
    instruction = default_instruction
    input_text  = q or ""
    context     = cx
    gold        = a
    rid_final   = rid if rid is not None else rid2
    return instruction, input_text, context, gold, (rid_final if rid_final is not None else None)

# -------------------------------
# Prompt construction (Alpaca-style)
# -------------------------------
DEFAULT_INSTRUCTION = (
    "Answer the patient's question accurately and concisely. Include a brief clinical rationale."
)

def build_input_block(question_or_input: str, context: Optional[str]) -> str:
    q = (question_or_input or "").strip()
    inp = f"Question: {q}" if q else "Question:"
    if context and context.strip():
        inp += f"\nContext:\n{context.strip()}"
    return inp

def build_prompt(instruction: str, input_block: str) -> str:
    return (
        "### Instruction:\n"
        f"{instruction.strip()}\n\n"
        "### Input:\n"
        f"{input_block.strip()}\n\n"
        "### Response:\n"
    )

# -------------------------------
# Teacher loader & generation
# -------------------------------
def _ensure_pad_eos(tok, model) -> None:
    """Guarantee valid pad/eos across tokenizer + model (+ generation_config)."""
    added = False
    if tok.eos_token_id is None:
        tok.add_special_tokens({'eos_token': '</s>'}); added = True
    if tok.pad_token_id is None:
        tok.add_special_tokens({'pad_token': '<|pad|>'}); added = True
    if added and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tok))
    model.config.pad_token_id = tok.pad_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tok.pad_token_id

def _make_generation_config(tok, model, temperature: float) -> GenerationConfig:
    """
    Build a clean GenerationConfig that avoids stray sampling flags warnings.
    Greedy by default; if temperature>0, enable sampling and set temperature.
    """
    # start from model.config to inherit eos/pad etc.
    cfg = GenerationConfig.from_model_config(model.config)
    # deterministic by default
    cfg.do_sample = False
    cfg.temperature = None
    # clear common sampling knobs that trigger warnings in greedy mode
    for attr in ("top_p", "top_k", "typical_p", "penalty_alpha"):
        if hasattr(cfg, attr):
            try:
                setattr(cfg, attr, None)
            except Exception:
                pass
    # ensure ids
    cfg.pad_token_id = tok.pad_token_id
    cfg.eos_token_id = tok.eos_token_id

    if temperature and temperature > 0.0:
        cfg.do_sample = True
        cfg.temperature = float(temperature)
        # leave top_p/top_k unset (None) unless you explicitly want them

    return cfg

def load_teacher(teacher_dir: Path, force_slow: bool=False, trust_remote_code: bool=False):
    """
    Robust loader with fast→slow fallback and dtype/torch_dtype selection
    that avoids the 'torch_dtype is deprecated!' warning on new Transformers.
    """
    tok = None
    tok_err = None

    if not force_slow:
        try:
            tok = AutoTokenizer.from_pretrained(
                str(teacher_dir), use_fast=True, trust_remote_code=trust_remote_code
            )
        except Exception as e:
            tok_err = e
            tok = None

    if tok is None:
        tok = AutoTokenizer.from_pretrained(
            str(teacher_dir), use_fast=False, trust_remote_code=trust_remote_code
        )

    tok.padding_side = "left"

    # choose dtype argument name based on Transformers version
    use_kwargs = {"device_map": "auto", "trust_remote_code": trust_remote_code}
    dtype_val = torch.bfloat16 if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(str(teacher_dir), **use_kwargs, torch_dtype=dtype_val, low_cpu_mem_usage=True)

    _ensure_pad_eos(tok, model)

    if tok_err is not None:
        print("[warn] Fast tokenizer failed; using slow tokenizer.")
        print(f"       Fast error: {repr(tok_err)}")
    if tok is not None and model is not None:
        print("Load model and tokenizer successfully")
    return tok, model

# Hard labelling
@torch.no_grad()
def generate_answer(tok, model, prompt: str, max_new_tokens: int, temperature: float) -> str:
    enc = tok(prompt, return_tensors="pt").to(model.device)
    gen_cfg = _make_generation_config(tok, model, temperature)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        generation_config=gen_cfg,
        )
    full = tok.decode(out[0], skip_special_tokens=True)
    pre  = tok.decode(enc.input_ids[0], skip_special_tokens=True)
    return full[len(pre):].strip()

# Soft labelling
@torch.no_grad()
def generate_with_scores(tok, model, prompt: str, max_new_tokens: int, temperature: float):
    """Return generated text + per-step logits for soft-label extraction."""
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=(temperature if temperature > 0.0 else None),
        repetition_penalty=1.0,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )
    seq = out.sequences[0]
    prompt_len = enc.input_ids.shape[1]
    gen_ids = seq[prompt_len:]
    full_text = tok.decode(seq, skip_special_tokens=True)
    prompt_text = tok.decode(enc.input_ids[0], skip_special_tokens=True)
    gen_text = full_text[len(prompt_text):].strip()
    return {"generated_text": gen_text, "generated_ids": gen_ids.detach().cpu(), "scores": out.scores}

def topk_logprobs_per_step(scores_list: List[torch.Tensor], gen_ids: torch.Tensor, k: int):
    """For each generated step t, take top-k tokens with their log-probs and mark the chosen id."""
    if k <= 0:
        return []
    out = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1)
    for t, logits in enumerate(scores_list):
        lp = logsoftmax(logits[0].float().cpu())
        topv, topi = torch.topk(lp, k)
        out.append({
            "t": t,
            "chosen_id": int(gen_ids[t].item()),
            "topk_ids": [int(i) for i in topi.tolist()],
            "topk_logprobs": [float(v) for v in topv.tolist()]
        })
    return out

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--command", choices=["text", "soft"], required=True,
                    help="text: text-only distillation; soft: add per-token top-k logprobs")
    ap.add_argument("--data-file", required=True, help="Path to data/healthcaremagic.jsonl")
    ap.add_argument("--teacher-dir", required=True, help="Path to model/medgemma-27b-text-it")
    ap.add_argument("--out-jsonl", required=True, help="Where to write distilled Alpaca-style JSONL")
    ap.add_argument("--log-file", default=None, help="Optional text log file")
    ap.add_argument("--max-samples", type=int, default=None, help="Limit examples for a quick run")
    ap.add_argument("--resume", action="store_true", help="Skip ids already present in out-jsonl")
    ap.add_argument("--include-gold", action="store_true", help="If available, include original gold in meta.gold")

    # Generation
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="Fallback instruction")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.0)

    # Tokenizer/model options
    ap.add_argument("--force-slow-tokenizer", action="store_true", help="Force use_fast=False (avoid fast tokenizer issues)")
    ap.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to HF loaders")

    # Soft-labels
    ap.add_argument("--topk-logprobs", type=int, default=10,
                    help="Top-k for soft labels (used only for --command soft)")
    ap.add_argument("--logprobs-file", default=None,
                    help="Optional JSONL(.gz) to store soft labels; if not set, embeds into out-jsonl")

    # JSONL key overrides (for flat formats)
    ap.add_argument("--question-keys", default=None, help="Comma-separated keys for question")
    ap.add_argument("--answer-keys",   default=None, help="Comma-separated keys for (gold) answer")
    ap.add_argument("--context-keys",  default=None, help="Comma-separated keys for context")
    ap.add_argument("--id-keys",       default=None, help="Comma-separated keys for id")
    args = ap.parse_args()

    data_path = Path(args.data_file)
    out_path  = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flog = open(args.log_file, "w", encoding="utf-8") if args.log_file else None

    # Parse key overrides for flat fallback
    q_keys  = parse_key_list(args.question_keys, DEFAULT_QUESTION_KEYS)
    a_keys  = parse_key_list(args.answer_keys,   DEFAULT_ANSWER_KEYS)
    c_keys  = parse_key_list(args.context_keys,  DEFAULT_CONTEXT_KEYS)
    id_keys = parse_key_list(args.id_keys,       DEFAULT_ID_KEYS)

    # Load teacher with robust tokenizer handling
    tok, model = load_teacher(Path(args.teacher_dir),
                              force_slow=args.force_slow_tokenizer,
                              trust_remote_code=args.trust_remote_code)

    # Resume: load already written ids to skip
    already = set()
    if args.resume and out_path.exists():
        if flog: flog.write(f"[resume] Reading existing ids from {out_path}\n")
        with out_path.open("r", encoding="utf-8") as fin:
            for line in fin:
                try:
                    obj = json.loads(line)
                    if "id" in obj:
                        already.add(str(obj["id"]))
                except Exception:
                    pass

    # Read data
    rows_raw = list(read_jsonl(data_path))
    if args.max_samples is not None:
        rows_raw = rows_raw[:args.max_samples]

    if not rows_raw:
        print("ERROR: No rows found in data file.")
        sys.exit(1)

    # Probe first few for sanity
    if flog:
        flog.write("[probe] first 5 rows field detection:\n")
        for i, obj in enumerate(rows_raw[:5]):
            instr, inp, cx, gold, rid = extract_fields_any(obj, q_keys, a_keys, c_keys, id_keys, args.instruction)
            flog.write(json.dumps({
                "pos": i,
                "id": rid if rid is not None else str(i),
                "instr_len": len(instr or ""),
                "input_len": len(inp or ""),
                "has_ctx": bool(cx),
                "has_gold": gold is not None,
                "keys_present": list(obj.keys())[:16]
            }, ensure_ascii=False) + "\n")

    # Soft-label output (optional separate file)
    mode = args.command
    soft_f = None
    if mode == "soft" and args.topk_logprobs > 0 and args.logprobs_file:
        soft_path = Path(args.logprobs_file)
        soft_path.parent.mkdir(parents=True, exist_ok=True)
        soft_f = gzip.open(soft_path, "wt", encoding="utf-8") if soft_path.suffix == ".gz" else open(soft_path, "w", encoding="utf-8")

    n_total = len(rows_raw)
    n_emit = 0
    n_skip = 0
    t0 = time.time()

    with out_path.open("a", encoding="utf-8") as fout:
        for i, obj in enumerate(rows_raw):
            instr, inp_text, ctx, gold, rid = extract_fields_any(obj, q_keys, a_keys, c_keys, id_keys, args.instruction)
            if rid is None:
                rid = str(i)

            if args.resume and rid in already:
                n_skip += 1
                continue

            if not inp_text or not inp_text.strip():
                n_skip += 1
                if flog: flog.write(f"[skip] idx={i} id={rid} (no input/question)\n")
                continue
            input_block = build_input_block(inp_text, ctx)
            prompt = build_prompt(instr or args.instruction, input_block)

            try:
                if mode == "text":
                    gen = generate_answer(tok, model, prompt, args.max_new_tokens, args.temperature)
                    rec = {
                        "instruction": instr or args.instruction,
                        "input": input_block,
                        "output": gen,
                        "source": "healthcaremagic_distilled_medgemma-27b-text-it",
                        "id": rid,
                        "task": "distillation",
                        "meta": {
                            "gen": {
                                "max_new_tokens": args.max_new_tokens,
                                "temperature": args.temperature,
                                "do_sample": (args.temperature > 0.0)
                            }
                        }
                    }
                    if args.include_gold and gold is not None:
                        rec["meta"]["gold"] = gold
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_emit += 1

                else:  # soft
                    out = generate_with_scores(tok, model, prompt, args.max_new_tokens, args.temperature)
                    gen_text = out["generated_text"]
                    gen_ids  = out["generated_ids"]
                    scores   = out["scores"]

                    rec = {
                        "instruction": instr or args.instruction,
                        "input": input_block,
                        "output": gen_text,
                        "source": "healthcaremagic_distilled_medgemma-27b-text-it",
                        "id": rid,
                        "task": "distillation",
                        "meta": {
                            "gen": {
                                "max_new_tokens": args.max_new_tokens,
                                "temperature": args.temperature,
                                "do_sample": (args.temperature > 0.0)
                            },
                            "gen_token_count": int(gen_ids.numel())
                        }
                    }
                    if args.include_gold and gold is not None:
                        rec["meta"]["gold"] = gold

                    if args.topk_logprobs > 0:
                        soft = topk_logprobs_per_step(scores, gen_ids, k=args.topk_logprobs)
                        if soft_f is not None:
                            soft_rec = {"id": rid, "topk": args.topk_logprobs, "steps": soft}
                            soft_f.write(json.dumps(soft_rec, ensure_ascii=False) + "\n")
                        else:
                            rec["soft_labels"] = {"topk": args.topk_logprobs, "steps": soft}

                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_emit += 1

                if flog and (n_emit % 10 == 0):
                    flog.write(f"[ok ] {n_emit}/{n_total} id={rid}\n")

            except Exception as e:
                n_skip += 1
                if flog:
                    flog.write(f"[err] idx={i} id={rid} error={repr(e)}\n")

    if soft_f is not None:
        soft_f.close()

    dt = time.time() - t0
    if flog:
        flog.write(f"\nDone: emitted={n_emit}, skipped={n_skip}, total={n_total}, time={dt/60.0:.2f} min\n")
        flog.close()

    if n_emit == 0:
        print("WARNING: No examples emitted. Check field keys and inputs.")
    else:
        print(f"Done. Wrote {n_emit} distilled examples to: {out_path}")
        if mode == "soft" and args.topk_logprobs > 0 and args.logprobs_file:
            print(f"Soft labels saved to: {args.logprobs_file}")

if __name__ == "__main__":
    main()
