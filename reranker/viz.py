#!/usr/bin/env python
"""
Visualise Hugging Face Trainer training logs from trainer_state.json.
Usage:
  python3 viz.py --input /Users/khoale/Downloads/reranker/checkpoint-22000/trainer_state.json
"""

import json
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 11,
    "axes.grid": True,
    "savefig.dpi": 200
})


def load_trainer_state(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    logs = state.get("log_history", [])
    if not logs:
        raise ValueError(f"No 'log_history' found in {path}")
    df = pd.DataFrame(logs)
    df = df.sort_values("step")
    return df


def plot_metric(df: pd.DataFrame, metric: str, outdir: Path):
    if metric not in df.columns:
        return
    plt.figure()
    plt.plot(df["step"], df[metric], label=metric, linewidth=1.8)
    plt.xlabel("Training step")
    plt.ylabel(metric)
    plt.title(f"{metric} over steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{metric}_vs_steps.png")
    plt.close()

    # Also vs epoch if present
    if "epoch" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df[metric], label=metric, linewidth=1.8, color="tab:orange")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{metric} over epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{metric}_vs_epochs.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to trainer_state.json")
    parser.add_argument("--output_dir", default="outputs/plots", help="Directory to save plots")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_trainer_state(in_path)
    print(f"[+] Loaded {len(df)} log entries from {in_path}")

    # Save raw log CSV for convenience
    csv_path = out_dir / "trainer_logs.csv"
    df.to_csv(csv_path, index=False)
    print(f"[+] Saved metrics CSV → {csv_path}")

    # Plot all numeric metrics
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for col in numeric_cols:
        if col in {"step", "epoch"}:
            continue
        plot_metric(df, col, out_dir)

    # Combined loss + eval_loss
    if "loss" in df.columns or "eval_loss" in df.columns:
        plt.figure()
        if "loss" in df.columns:
            plt.plot(df["step"], df["loss"], label="train_loss", linewidth=1.6)
        if "eval_loss" in df.columns:
            plt.plot(df["step"], df["eval_loss"], label="eval_loss", linewidth=1.6)
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title("Training vs Evaluation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "loss_comparison.png")
        plt.close()

    print(f"[+] Plots saved in {out_dir}")

    if args.show:
        import webbrowser, os
        for f in sorted(out_dir.glob("*.png")):
            webbrowser.open(f"file://{os.path.abspath(f)}")


if __name__ == "__main__":
    main()

