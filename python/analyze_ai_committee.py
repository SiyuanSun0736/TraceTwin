#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = ROOT / "checkpoints"
DEFAULT_OUTPUT_DIR = ROOT / "spie-proceedings-style"


def latest_infer_log(relative_dir: str) -> Path:
    run_dir = CHECKPOINTS_DIR / relative_dir
    candidates = sorted(list(run_dir.glob("infer_*.log")) + list(run_dir.glob("infer_*.txt")))
    if not candidates:
        raise FileNotFoundError(f"No infer logs found in {run_dir}")
    return Path(relative_dir) / candidates[-1].name

LABEL_RUNS = {
    "fixed_time": {
        "CNN": latest_infer_log("cnn/fixed_time_best"),
        "LSTM": latest_infer_log("lstm/fixed_time_best"),
        "Transformer": latest_infer_log("transformer/fixed_time_best"),
    },
    "fixed_work": {
        "CNN": latest_infer_log("cnn/fixed_work_best"),
        "LSTM": latest_infer_log("lstm/fixed_work_best"),
        "Transformer": latest_infer_log("transformer/fixed_work_best"),
    },
    "inst": {
        "CNN": latest_infer_log("cnn/inst_best"),
        "LSTM": latest_infer_log("lstm/inst_best"),
        "Transformer": latest_infer_log("transformer/inst_best"),
    },
}

LABEL_DISPLAY = {
    "fixed_time": "Fixed-Time",
    "fixed_work": "Fixed-Work",
    "inst": "Inst-Retired",
}

MODEL_COLORS = {
    "CNN": "#16697a",
    "LSTM": "#c46d3c",
    "Transformer": "#2f4b7c",
    "Majority vote": "#6c8f3d",
    "Mean score": "#7b2cbf",
    "Unanimous committee": "#d1495b",
}
PAIR_COLORS = {
    "O1-g_vs_O3-g": "#cc7a3b",
    "O2-bolt_vs_O2-bolt-opt": "#2a9d8f",
    "O3-bolt_vs_O3-bolt-opt": "#5c7cfa",
}

PAIR_HEADER = re.compile(r"版本对:\s+([^\s]+)")
SAMPLE_LINE = re.compile(
    r"INFO:\s+\d+\s+(?P<program>.+?)\s+(?P<pred>-?\d+\.\d+)\s+(?P<true>-?\d+\.\d+)\s+(?P<err>[+-]?\d+\.\d+)\s+"
)
DIR_ACC_LINE = re.compile(r"方向准确率\s*=\s*(\d+)/(\d+)\s*\(([0-9.]+)%\)")


@dataclass(frozen=True)
class SamplePrediction:
    pair: str
    program: str
    pred: float
    true: float


@dataclass(frozen=True)
class AlignedSample:
    pair: str
    program: str
    true: float
    cnn_pred: float
    lstm_pred: float
    transformer_pred: float

    @property
    def predictions(self) -> dict[str, float]:
        return {
            "CNN": self.cnn_pred,
            "LSTM": self.lstm_pred,
            "Transformer": self.transformer_pred,
        }

    @property
    def mean_pred(self) -> float:
        return float(np.mean(list(self.predictions.values())))

    @property
    def std_pred(self) -> float:
        return float(np.std(list(self.predictions.values())))

    @property
    def mean_margin(self) -> float:
        return abs(self.mean_pred - 1.0)

    @property
    def true_gap(self) -> float:
        return abs(self.true - 1.0)

    @property
    def votes(self) -> dict[str, bool]:
        return {name: pred >= 1.0 for name, pred in self.predictions.items()}

    @property
    def majority_vote(self) -> bool:
        return sum(self.votes.values()) >= 2

    @property
    def unanimous(self) -> bool:
        return len(set(self.votes.values())) == 1

    @property
    def majority_correct(self) -> bool:
        return direction_ok(1.01 if self.majority_vote else 0.99, self.true)


@dataclass(frozen=True)
class AccuracyRecord:
    correct: int
    total: int

    @property
    def pct(self) -> float:
        return 100.0 * self.correct / self.total


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.facecolor": "#fcfbf7",
            "figure.facecolor": "#f5efe6",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "grid.color": "#9aa5b1",
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def parse_log(path: Path) -> dict[tuple[str, str], SamplePrediction]:
    rows: dict[tuple[str, str], SamplePrediction] = {}
    current_pair: str | None = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        pair_match = PAIR_HEADER.search(line)
        if pair_match:
            current_pair = pair_match.group(1)
            continue
        sample_match = SAMPLE_LINE.search(line)
        if sample_match and current_pair is not None:
            program = sample_match.group("program").strip()
            rows[(current_pair, program)] = SamplePrediction(
                pair=current_pair,
                program=program,
                pred=float(sample_match.group("pred")),
                true=float(sample_match.group("true")),
            )
    return rows


def parse_accuracy_summary(path: Path) -> AccuracyRecord:
    matches = DIR_ACC_LINE.findall(path.read_text(encoding="utf-8", errors="ignore"))
    if len(matches) < 3:
        raise ValueError(f"Cannot find 3 direction-accuracy summary lines in {path}")
    correct = sum(int(match[0]) for match in matches[:3])
    total = sum(int(match[1]) for match in matches[:3])
    return AccuracyRecord(correct=correct, total=total)


def direction_ok(pred: float, true: float) -> bool:
    return (pred >= 1.0 and true >= 1.0) or (pred < 1.0 and true < 1.0)


def load_aligned_samples(label: str) -> list[AlignedSample]:
    logs = {name: parse_log(CHECKPOINTS_DIR / rel_path) for name, rel_path in LABEL_RUNS[label].items()}
    shared_keys = sorted(set.intersection(*(set(rows.keys()) for rows in logs.values())))
    samples = []
    for pair, program in shared_keys:
        cnn = logs["CNN"][(pair, program)]
        lstm = logs["LSTM"][(pair, program)]
        transformer = logs["Transformer"][(pair, program)]
        samples.append(
            AlignedSample(
                pair=pair,
                program=program,
                true=transformer.true,
                cnn_pred=cnn.pred,
                lstm_pred=lstm.pred,
                transformer_pred=transformer.pred,
            )
        )
    return samples


def compute_policy_summary(samples: list[AlignedSample], label: str) -> list[dict[str, float | str | int]]:
    total = len(samples)
    summary = []
    for model in ["CNN", "LSTM", "Transformer"]:
        official = parse_accuracy_summary(CHECKPOINTS_DIR / LABEL_RUNS[label][model])
        summary.append(
            {
                "policy": model,
                "coverage": 100.0,
                "accuracy": official.pct,
                "correct": official.correct,
                "total": official.total,
                "source": "official-log-summary",
            }
        )

    majority_correct = sum(sample.majority_correct for sample in samples)
    summary.append(
        {
            "policy": "Majority vote",
            "coverage": 100.0,
            "accuracy": 100.0 * majority_correct / total,
            "correct": majority_correct,
            "total": total,
            "source": "aligned-sample-recompute",
        }
    )

    mean_score_correct = sum(direction_ok(sample.mean_pred, sample.true) for sample in samples)
    mean_mae = float(np.mean([abs(sample.mean_pred - sample.true) for sample in samples]))
    mean_rmse = float(np.sqrt(np.mean([(sample.mean_pred - sample.true) ** 2 for sample in samples])))
    summary.append(
        {
            "policy": "Mean score",
            "coverage": 100.0,
            "accuracy": 100.0 * mean_score_correct / total,
            "correct": mean_score_correct,
            "total": total,
            "mae": mean_mae,
            "rmse": mean_rmse,
            "source": "aligned-sample-recompute",
        }
    )

    unanimous_samples = [sample for sample in samples if sample.unanimous]
    unanimous_correct = sum(sample.majority_correct for sample in unanimous_samples)
    summary.append(
        {
            "policy": "Unanimous committee",
            "coverage": 100.0 * len(unanimous_samples) / total,
            "accuracy": 100.0 * unanimous_correct / len(unanimous_samples),
            "correct": unanimous_correct,
            "total": len(unanimous_samples),
            "source": "aligned-sample-recompute",
        }
    )
    return summary


def compute_agreement_groups(samples: list[AlignedSample]) -> list[dict[str, float | str | int]]:
    groups = {
        "Unanimous": [sample for sample in samples if sample.unanimous],
        "2-1 split": [sample for sample in samples if not sample.unanimous],
    }
    total = len(samples)
    results = []
    for name, group_samples in groups.items():
        correct = sum(sample.majority_correct for sample in group_samples)
        results.append(
            {
                "group": name,
                "coverage": 100.0 * len(group_samples) / total,
                "accuracy": 100.0 * correct / len(group_samples),
                "count": len(group_samples),
            }
        )
    return results


def compute_margin_gate_stats(
    samples: list[AlignedSample],
    thresholds: list[float] | None = None,
) -> list[dict[str, float | int]]:
    if thresholds is None:
        thresholds = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]

    total = len(samples)
    results = []
    for threshold in thresholds:
        kept = [sample for sample in samples if sample.mean_margin >= threshold]
        correct = sum(sample.majority_correct for sample in kept)
        results.append(
            {
                "threshold": threshold,
                "coverage": 100.0 * len(kept) / total,
                "accuracy": 100.0 * correct / len(kept) if kept else 0.0,
                "count": len(kept),
            }
        )
    return results


def top_disagreement_cases(samples: list[AlignedSample], limit: int = 20) -> list[dict[str, float | str | bool]]:
    sorted_samples = sorted(samples, key=lambda sample: (-sample.std_pred, sample.mean_margin, sample.program))
    cases = []
    for sample in sorted_samples[:limit]:
        cases.append(
            {
                "pair": sample.pair,
                "program": sample.program,
                "true": sample.true,
                "cnn_pred": sample.cnn_pred,
                "lstm_pred": sample.lstm_pred,
                "transformer_pred": sample.transformer_pred,
                "mean_pred": sample.mean_pred,
                "std_pred": sample.std_pred,
                "mean_margin": sample.mean_margin,
                "true_gap": sample.true_gap,
                "majority_correct": sample.majority_correct,
                "unanimous": sample.unanimous,
            }
        )
    return cases


def plot_policy_landscape(
    policy_summary: list[dict[str, float | str | int]],
    agreement_groups: list[dict[str, float | str | int]],
    output_dir: Path,
    label: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), constrained_layout=True)

    label_offsets = {
        "CNN": (1.2, 0.2),
        "LSTM": (1.2, 0.15),
        "Transformer": (2.2, -0.2),
        "Majority vote": (2.2, 0.55),
        "Mean score": (2.2, 1.15),
        "Unanimous committee": (0.8, 0.55),
    }
    for item in policy_summary:
        policy = str(item["policy"])
        coverage = float(item["coverage"])
        accuracy = float(item["accuracy"])
        axes[0].scatter(
            coverage,
            accuracy,
            s=120,
            color=MODEL_COLORS[policy],
            edgecolors="white",
            linewidths=0.9,
            zorder=3,
        )
        dx, dy = label_offsets[policy]
        label_x = coverage + dx
        label_y = accuracy + dy
        if coverage >= 99.5:
            axes[0].plot([coverage + 0.15, label_x - 0.25], [accuracy, label_y], color="#7b8794", linewidth=0.8, zorder=2)
        axes[0].text(
            label_x,
            label_y,
            f"{policy}\n{float(item['accuracy']):.1f}%",
            fontsize=8,
            color="#243b53",
            bbox={"facecolor": "#fcfbf7", "edgecolor": "none", "pad": 0.18, "alpha": 0.92},
            zorder=4,
        )
    axes[0].set_xlim(44.5, 112.5)
    axes[0].set_ylim(60, 92.5)
    axes[0].set_xlabel("Coverage retained (%)")
    axes[0].set_ylabel("Directional accuracy (%)")
    axes[0].set_title("Committee policies")

    x = np.arange(len(agreement_groups))
    bars = axes[1].bar(
        x,
        [float(item["accuracy"]) for item in agreement_groups],
        color=["#d1495b", "#577590"],
        width=0.58,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(item["group"]) for item in agreement_groups])
    axes[1].set_ylim(50, 95)
    axes[1].set_ylabel("Directional accuracy (%)")
    axes[1].set_title("Agreement groups")
    for bar, item in zip(bars, agreement_groups):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"cov={float(item['coverage']):.1f}%\n(n={int(item['count'])})",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle(f"AI committee experiment ({label})", fontsize=13)
    fig.savefig(output_dir / f"ai-committee-policy-{label}.pdf")
    fig.savefig(output_dir / f"ai-committee-policy-{label}.png")
    plt.close(fig)


def plot_margin_screening(
    samples: list[AlignedSample],
    margin_gate_stats: list[dict[str, float | int]],
    output_dir: Path,
    label: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)

    correct = [sample for sample in samples if sample.majority_correct]
    wrong = [sample for sample in samples if not sample.majority_correct]
    axes[0].plot(
        [float(item["coverage"]) for item in margin_gate_stats],
        [float(item["accuracy"]) for item in margin_gate_stats],
        color="#7b2cbf",
        linewidth=2.2,
        marker="o",
        markersize=5,
    )
    highlighted = [item for item in margin_gate_stats if float(item["threshold"]) in {0.0, 0.01, 0.02, 0.05}]
    for item in highlighted:
        axes[0].text(
            float(item["coverage"]) + 1.0,
            float(item["accuracy"]) + 0.18,
            f"m>={float(item['threshold']):.3f}",
            fontsize=7,
            color="#243b53",
        )
    axes[0].set_xlim(0, 103)
    axes[0].set_ylim(84, 100.8)
    axes[0].set_xlabel("Coverage retained (%)")
    axes[0].set_ylabel("Directional accuracy (%)")
    axes[0].set_title("Margin-gated selective screening")

    bins = np.linspace(0.0, max(sample.mean_margin for sample in samples) + 0.02, 18)
    axes[1].hist(
        [sample.mean_margin for sample in correct],
        bins=bins,
        density=True,
        alpha=0.72,
        color="#2a9d8f",
        label="Majority correct",
    )
    axes[1].hist(
        [sample.mean_margin for sample in wrong],
        bins=bins,
        density=True,
        alpha=0.75,
        color="#d1495b",
        label="Majority wrong",
    )
    axes[1].axvline(0.02, color="#334e68", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Committee mean margin |mean(y_hat)-1|")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Margin separation")
    axes[1].legend(loc="upper right")

    fig.suptitle(f"AI confidence experiment ({label})", fontsize=13)
    fig.savefig(output_dir / f"ai-committee-confidence-{label}.pdf")
    fig.savefig(output_dir / f"ai-committee-confidence-{label}.png")
    plt.close(fig)


def plot_cross_label_confidence(
    label_results: dict[str, dict[str, list[dict[str, float | str | int]]]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), constrained_layout=True)

    for label, results in label_results.items():
        display = LABEL_DISPLAY[label]
        margin_gate_stats = results["margin_gate_stats"]
        axes[0].plot(
            [float(item["coverage"]) for item in margin_gate_stats],
            [float(item["accuracy"]) for item in margin_gate_stats],
            linewidth=2.1,
            marker="o",
            markersize=4.5,
            label=display,
        )
    axes[0].set_xlim(0, 103)
    axes[0].set_ylim(60, 100.8)
    axes[0].set_xlabel("Coverage retained (%)")
    axes[0].set_ylabel("Directional accuracy (%)")
    axes[0].set_title("Selective confidence curves")
    axes[0].legend(loc="lower left")

    thresholds = [0.01, 0.02, 0.05]
    x = np.arange(len(thresholds))
    width = 0.24
    label_order = list(LABEL_RUNS)
    for index, label in enumerate(label_order):
        gate_map = {float(item["threshold"]): item for item in label_results[label]["margin_gate_stats"]}
        offsets = x + (index - 1) * width
        bars = axes[1].bar(
            offsets,
            [float(gate_map[threshold]["accuracy"]) for threshold in thresholds],
            width=width,
            label=LABEL_DISPLAY[label],
        )
        for bar, threshold in zip(bars, thresholds):
            coverage = float(gate_map[threshold]["coverage"])
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.35,
                f"{coverage:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"m>={threshold:.2f}" for threshold in thresholds])
    axes[1].set_ylim(60, 101.5)
    axes[1].set_ylabel("Directional accuracy (%)")
    axes[1].set_title("Accuracy at shared operating points")

    fig.suptitle("Cross-label AI confidence comparison", fontsize=13)
    fig.savefig(output_dir / "ai-committee-cross-label-confidence.pdf")
    fig.savefig(output_dir / "ai-committee-cross-label-confidence.png")
    plt.close(fig)


def write_summary_files(
    policy_summary: list[dict[str, float | str | int]],
    agreement_groups: list[dict[str, float | str | int]],
    margin_gate_stats: list[dict[str, float | int]],
    disagreement_cases: list[dict[str, float | str | bool]],
    output_dir: Path,
    label: str,
) -> None:
    summary_payload = {
        "label": label,
        "policies": policy_summary,
        "agreement_groups": agreement_groups,
        "margin_gate_stats": margin_gate_stats,
        "top_disagreement_cases": disagreement_cases,
    }
    (output_dir / f"ai_committee_summary_{label}.json").write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    fieldnames = list(disagreement_cases[0].keys())
    with (output_dir / f"ai_committee_top_cases_{label}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(disagreement_cases)


def print_summary(
    policy_summary: list[dict[str, float | str | int]],
    agreement_groups: list[dict[str, float | str | int]],
    margin_gate_stats: list[dict[str, float | int]],
) -> None:
    print("Policy summary")
    for item in policy_summary:
        line = f"- {item['policy']}: coverage={float(item['coverage']):.1f}%, accuracy={float(item['accuracy']):.1f}%"
        if "mae" in item:
            line += f", mae={float(item['mae']):.4f}, rmse={float(item['rmse']):.4f}"
        print(line)
    print("\nAgreement groups")
    for item in agreement_groups:
        print(f"- {item['group']}: coverage={float(item['coverage']):.1f}%, accuracy={float(item['accuracy']):.1f}%")

    print("\nMargin gates")
    for item in margin_gate_stats:
        if float(item["threshold"]) in {0.01, 0.02, 0.05}:
            print(
                f"- margin>={float(item['threshold']):.3f}: coverage={float(item['coverage']):.1f}%, "
                f"accuracy={float(item['accuracy']):.1f}%"
            )


def run_label_analysis(label: str, output_dir: Path) -> dict[str, list[dict[str, float | str | int]]]:
    samples = load_aligned_samples(label)
    policy_summary = compute_policy_summary(samples, label)
    agreement_groups = compute_agreement_groups(samples)
    margin_gate_stats = compute_margin_gate_stats(samples)
    disagreement_cases = top_disagreement_cases(samples)

    plot_policy_landscape(policy_summary, agreement_groups, output_dir, label)
    plot_margin_screening(samples, margin_gate_stats, output_dir, label)
    write_summary_files(policy_summary, agreement_groups, margin_gate_stats, disagreement_cases, output_dir, label)
    print(f"\n=== {LABEL_DISPLAY[label]} ===")
    print_summary(policy_summary, agreement_groups, margin_gate_stats)
    return {
        "policy_summary": policy_summary,
        "agreement_groups": agreement_groups,
        "margin_gate_stats": margin_gate_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI-themed committee experiments from existing infer logs.")
    parser.add_argument("--label", choices=sorted(LABEL_RUNS), default="fixed_work")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--all-labels", action="store_true")
    args = parser.parse_args()

    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_labels:
        label_results = {label: run_label_analysis(label, args.output_dir) for label in LABEL_RUNS}
        plot_cross_label_confidence(label_results, args.output_dir)
        (args.output_dir / "ai_committee_cross_label_summary.json").write_text(
            json.dumps(label_results, indent=2),
            encoding="utf-8",
        )
        return

    run_label_analysis(args.label, args.output_dir)


if __name__ == "__main__":
    main()