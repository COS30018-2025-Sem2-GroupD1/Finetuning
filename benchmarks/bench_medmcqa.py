#!/usr/bin/env python
"""
evaluate_medmcqa.py

Evaluate a generative model (e.g. MedGemma) on the MedMCQA multiple‑choice QA dataset.
The data file should be a JSON Lines file with fields:
  question: str
  options: list[str] of length 4, corresponding to A, B, C, D
  label: str, one of "A", "B", "C", "D"
"""
import argparse, json, re
from pathlib import Path
from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def load_dataset(path: Path, max_samples: int = None) -> List[Tuple[str, List[str], str]]:
    """Load MedMCQA from JSONL; return (question, options list, label) tuples."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            obj = json.loads(line)
            items.append((obj["question"].strip(),
                          obj["options"],
                          obj["label"].strip().upper()))
    return items

def construct_prompt(question: str, options: List[str]) -> str:
    """Format the multiple‑choice question for the model."""
    template = (
        "You are a medical expert.\n"
        "Read the following multiple‑choice question and answer by returning the letter "
        "(A, B, C or D) corresponding to the correct option.\n\n"
        "Question: {question}\n\n"
        "Options:\n"
        "A. {A}\n"
        "B. {B}\n"
        "C. {C}\n"
        "D. {D}\n\n"
        "Answer:"
    )
    return template.format(question=question, A=options[0], B=options[1], C=options[2], D=options[3])

def extract_letter(text: str) -> str:
    """Extract the first occurrence of A/B/C/D from a string (case‑insensitive)."""
    m = re.search(r"\b([ABCD])\b", text, flags=re.IGNORECASE)
    return m.group(1).upper() if m else ""

@torch.no_grad()
def evaluate(model, tokenizer, dataset: List[Tuple[str, List[str], str]], max_new_tokens: int = 16) -> float:
    """Loop over dataset, prompt the model and compute accuracy."""
    correct = 0
    for question, options, label in tqdm(dataset, desc="Evaluating"):
        prompt = construct_prompt(question, options)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = tokenizer.decode(out[0], skip_special_tokens=True)
        answer = extract_letter(generated[len(prompt):])
        correct += int(answer == label)
    return correct / len(dataset)

def main():
    parser = argparse.ArgumentParser(description="Evaluate MedGemma on MedMCQA")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing MedGemma model files")
    parser.add_argument("--data-file", type=Path, required=True, help="Path to medmcqa.jsonl")
    parser.add_argument("--max-samples", type=int, default=None, help="Evaluate only the first N questions")
    args = parser.parse_args()

    samples = load_dataset(args.data_file, args.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    acc = evaluate(model, tokenizer, samples)
    print(f"Accuracy on {len(samples)} questions: {acc:.4f}")

if __name__ == "__main__":
    main()
