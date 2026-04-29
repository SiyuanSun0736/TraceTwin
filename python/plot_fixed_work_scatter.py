import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "checkpoints" / "transformer" / "fixed_work_best" / "infer_20260330_154532.log"
OUTPUT_PDF = ROOT / "spie-proceedings-style" / "fixed-work-transformer-scatter.pdf"


PAIR_NAME_MAP = {
    "O1-g_vs_O3-g": "O1-g vs O3-g",
    "O2-bolt_vs_O2-bolt-opt": "O2-bolt vs O2-bolt-opt",
    "O3-bolt_vs_O3-bolt-opt": "O3-bolt vs O3-bolt-opt",
}


def parse_log(log_path: Path):
    pair_header = re.compile(r"版本对:\s+([^\s]+)")
    sample_line = re.compile(
        r"INFO:\s+\d+\s+(?P<program>.+?)\s+(?P<pred>-?\d+\.\d+)\s+(?P<true>-?\d+\.\d+)\s+(?P<err>[+-]?\d+\.\d+)\s+"
    )

    current_pair = None
    rows = []
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        pair_match = pair_header.search(raw_line)
        if pair_match:
            current_pair = pair_match.group(1)
            continue

        sample_match = sample_line.search(raw_line)
        if sample_match and current_pair is not None:
            rows.append(
                {
                    "pair": current_pair,
                    "program": sample_match.group("program").strip(),
                    "pred": float(sample_match.group("pred")),
                    "true": float(sample_match.group("true")),
                }
            )
    return rows


def pair_stats(rows):
    true_values = np.array([row["true"] for row in rows], dtype=float)
    pred_values = np.array([row["pred"] for row in rows], dtype=float)
    mae = np.mean(np.abs(pred_values - true_values))
    corr = np.corrcoef(true_values, pred_values)[0, 1] if len(rows) > 1 else np.nan
    return mae, corr


def main():
    rows = parse_log(LOG_PATH)
    if not rows:
        raise RuntimeError(f"No prediction rows parsed from {LOG_PATH}")

    grouped = {}
    for row in rows:
        grouped.setdefault(row["pair"], []).append(row)

    order = [pair for pair in PAIR_NAME_MAP if pair in grouped]
    colors = {
        "O1-g_vs_O3-g": "#1f77b4",
        "O2-bolt_vs_O2-bolt-opt": "#ff7f0e",
        "O3-bolt_vs_O3-bolt-opt": "#2ca02c",
    }

    all_true = np.array([row["true"] for row in rows], dtype=float)
    all_pred = np.array([row["pred"] for row in rows], dtype=float)
    lower = min(all_true.min(), all_pred.min()) - 0.03
    upper = max(all_true.max(), all_pred.max()) + 0.03

    fig, axes = plt.subplots(1, len(order), figsize=(14, 4.4), constrained_layout=True)
    if len(order) == 1:
        axes = [axes]

    for axis, pair in zip(axes, order):
        pair_rows = grouped[pair]
        x = np.array([row["true"] for row in pair_rows], dtype=float)
        y = np.array([row["pred"] for row in pair_rows], dtype=float)
        mae, corr = pair_stats(pair_rows)

        axis.scatter(
            x,
            y,
            s=24,
            alpha=0.75,
            color=colors[pair],
            edgecolors="white",
            linewidths=0.4,
        )
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="#444444", linewidth=1.0)
        axis.set_title(PAIR_NAME_MAP[pair], fontsize=11)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_xlabel("Ground-truth multiplier")
        axis.set_ylabel("Predicted multiplier")
        axis.grid(alpha=0.2)
        axis.text(
            0.04,
            0.96,
            f"n={len(pair_rows)}\nMAE={mae:.3f}\nr={corr:.3f}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
        )

    fig.suptitle("Transformer under fixed-workload labeling", fontsize=13)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()