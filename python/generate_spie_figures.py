#!/usr/bin/env python3

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "spie-proceedings-style"


def latest_infer_log(relative_dir: str) -> Path:
    run_dir = CHECKPOINTS_DIR / relative_dir
    candidates = sorted(list(run_dir.glob("infer_*.log")) + list(run_dir.glob("infer_*.txt")))
    if not candidates:
        raise FileNotFoundError(f"No infer logs found in {run_dir}")
    return Path(relative_dir) / candidates[-1].name

PAIR_ORDER = [
    "O1-g_vs_O3-g",
    "O2-bolt_vs_O2-bolt-opt",
    "O3-bolt_vs_O3-bolt-opt",
]
PAIR_SHORT = {
    "O1-g_vs_O3-g": "O1-g vs O3-g",
    "O2-bolt_vs_O2-bolt-opt": "O2-bolt vs O2-bolt-opt",
    "O3-bolt_vs_O3-bolt-opt": "O3-bolt vs O3-bolt-opt",
}
PAIR_PANEL = {
    "O1-g_vs_O3-g": "Compiler-level",
    "O2-bolt_vs_O2-bolt-opt": "BOLT pair A",
    "O3-bolt_vs_O3-bolt-opt": "BOLT pair B",
}
MODEL_ORDER = ["CNN", "LSTM", "Transformer"]
LABEL_ORDER = ["fixed_time", "fixed_work", "inst"]
LABEL_TITLES = {
    "fixed_time": "Fixed-Time",
    "fixed_work": "Fixed-Work",
    "inst": "Inst-Retired",
}
MODEL_COLORS = {
    "CNN": "#16697a",
    "LSTM": "#c46d3c",
    "Transformer": "#2f4b7c",
}
LABEL_COLORS = {
    "fixed_time": "#c46d3c",
    "fixed_work": "#2a9d8f",
    "inst": "#6c8f3d",
}
HOTSPOT_COLOR = "#d1495b"
RISK_CURVE_COLOR = "#1f4e79"
SCREENING_THRESHOLDS = [0.00, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08]
PAIR_COLORS = {
    "O1-g_vs_O3-g": "#cc7a3b",
    "O2-bolt_vs_O2-bolt-opt": "#2a9d8f",
    "O3-bolt_vs_O3-bolt-opt": "#5c7cfa",
}
ACC_CMAP = LinearSegmentedColormap.from_list(
    "tracetwin_acc",
    ["#fff8ec", "#f6c56d", "#4ea699", "#204b74"],
)

BEST_RUNS = [
    ("CNN", "fixed_time", latest_infer_log("cnn/fixed_time_best")),
    ("CNN", "fixed_work", latest_infer_log("cnn/fixed_work_best")),
    ("CNN", "inst", latest_infer_log("cnn/inst_best")),
    ("LSTM", "fixed_time", latest_infer_log("lstm/fixed_time_best")),
    ("LSTM", "fixed_work", latest_infer_log("lstm/fixed_work_best")),
    ("LSTM", "inst", latest_infer_log("lstm/inst_best")),
    ("Transformer", "fixed_time", latest_infer_log("transformer/fixed_time_best")),
    ("Transformer", "fixed_work", latest_infer_log("transformer/fixed_work_best")),
    ("Transformer", "inst", latest_infer_log("transformer/inst_best")),
]
TRANSFORMER_VARIANTS = {
    "fixed_time": [
        ("v1", Path("transformer/fixed_timev1/infer_20260331_193632.log")),
        ("v2", Path("transformer/fixed_timev2/infer_20260331_201750.log")),
        ("v3", Path("transformer/fixed_timev3/infer_v3.txt")),
        ("best", latest_infer_log("transformer/fixed_time_best")),
    ],
    "fixed_work": [
        ("v2", Path("transformer/fixed_workv2/infer_v2.log")),
        ("v3", Path("transformer/fixed_workv3/infer_v3.log")),
        ("best", latest_infer_log("transformer/fixed_work_best")),
    ],
    "inst": [
        ("v1", Path("transformer/inst_retiredv1/infer_20260330_192629.txt")),
        ("v2", Path("transformer/inst_retiredv2/infer_20260331_172047.log")),
        ("v3 (=best)", latest_infer_log("transformer/inst_best")),
    ],
}
TRANSFORMER_LABEL_RUNS = {
    "fixed_time": latest_infer_log("transformer/fixed_time_best"),
    "fixed_work": latest_infer_log("transformer/fixed_work_best"),
    "inst": latest_infer_log("transformer/inst_best"),
}

PAIR_HEADER = re.compile(r"版本对:\s+([^\s]+)")
SAMPLE_LINE = re.compile(
    r"INFO:\s+\d+\s+(?P<program>.+?)\s+(?P<pred>-?\d+\.\d+)\s+(?P<true>-?\d+\.\d+)\s+(?P<err>[+-]?\d+\.\d+)\s+"
)
DIR_ACC_LINE = re.compile(r"方向准确率\s*=\s*(\d+)/(\d+)\s*\(([0-9.]+)%\)")


@dataclass(frozen=True)
class AccuracyRecord:
    correct: int
    total: int

    @property
    def pct(self) -> float:
        return 100.0 * self.correct / self.total


@dataclass(frozen=True)
class SampleRow:
    pair: str
    program: str
    pred: float
    true: float

    @property
    def abs_error(self) -> float:
        return abs(self.pred - self.true)

    @property
    def gain_gap(self) -> float:
        return abs(self.true - 1.0)


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


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig: Figure, stem: str, export_svg: bool = False) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    fig.savefig(OUTPUT_DIR / f"{stem}.png")
    if export_svg:
        fig.savefig(OUTPUT_DIR / f"{stem}.svg")
    plt.close(fig)


def parse_samples(path: Path) -> list[SampleRow]:
    rows: list[SampleRow] = []
    current_pair: str | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        pair_match = PAIR_HEADER.search(raw_line)
        if pair_match:
            current_pair = pair_match.group(1)
            continue

        sample_match = SAMPLE_LINE.search(raw_line)
        if sample_match and current_pair is not None:
            rows.append(
                SampleRow(
                    pair=current_pair,
                    program=sample_match.group("program").strip(),
                    pred=float(sample_match.group("pred")),
                    true=float(sample_match.group("true")),
                )
            )
    return rows


def parse_accuracy_summary(path: Path) -> dict[str, AccuracyRecord]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = DIR_ACC_LINE.findall(text)
    if len(matches) != 3:
        raise ValueError(f"expected 3 direction-accuracy summaries in {path}, got {len(matches)}")
    return {
        pair: AccuracyRecord(correct=int(correct), total=int(total))
        for pair, (correct, total, _pct) in zip(PAIR_ORDER, matches)
    }


def direction_ok(pred: float, true: float) -> bool:
    return (pred >= 1.0 and true >= 1.0) or (pred < 1.0 and true < 1.0)


def benchmark_family(program: str) -> str:
    ignored = {
        "llvm",
        "test",
        "tests",
        "suite",
        "benchmark",
        "benchmarks",
        "multisource",
        "singlesource",
        "external",
        "applications",
        "microbenchmarks",
    }
    for token in [item for item in re.split(r"[/.\\-]+", program) if item]:
        lowered = token.lower()
        if lowered in ignored:
            continue
        if any(char.isalpha() for char in token):
            return token
    return program


def build_family_hotspots(rows: list[SampleRow], min_samples: int = 8) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[SampleRow]] = {}
    for row in rows:
        grouped.setdefault(benchmark_family(row.program), []).append(row)

    stats = []
    for family, family_rows in grouped.items():
        if len(family_rows) < min_samples:
            continue
        wrong = sum(not direction_ok(row.pred, row.true) for row in family_rows)
        stats.append(
            {
                "family": family,
                "total": len(family_rows),
                "wrong": wrong,
                "accuracy": 100.0 * (len(family_rows) - wrong) / len(family_rows),
                "mae": float(np.mean([row.abs_error for row in family_rows])),
                "mean_gap": float(np.mean([row.gain_gap for row in family_rows])),
            }
        )

    stats.sort(key=lambda item: (-int(item["wrong"]), float(item["accuracy"]), str(item["family"])))
    displayed_wrong = sum(int(item["wrong"]) for item in stats) or 1
    for item in stats:
        item["error_share"] = 100.0 * int(item["wrong"]) / displayed_wrong
    return stats


def build_selective_screening(rows: list[SampleRow], thresholds: list[float]) -> list[dict[str, float | int]]:
    total = len(rows)
    stats = []
    for threshold in thresholds:
        retained = [row for row in rows if abs(row.pred - 1.0) >= threshold]
        if retained:
            accuracy = 100.0 * sum(direction_ok(row.pred, row.true) for row in retained) / len(retained)
            mae = float(np.mean([row.abs_error for row in retained]))
        else:
            accuracy = 0.0
            mae = 0.0
        stats.append(
            {
                "threshold": threshold,
                "retained": len(retained),
                "coverage": 100.0 * len(retained) / total,
                "accuracy": accuracy,
                "mae": mae,
            }
        )
    return stats


def selective_pair_operating_point(rows: list[SampleRow], threshold: float) -> list[dict[str, float | int | str]]:
    stats = []
    for pair in PAIR_ORDER:
        pair_rows = [row for row in rows if row.pair == pair]
        retained = [row for row in pair_rows if abs(row.pred - 1.0) >= threshold]
        accuracy = 100.0 * sum(direction_ok(row.pred, row.true) for row in retained) / len(retained) if retained else 0.0
        stats.append(
            {
                "pair": pair,
                "coverage": 100.0 * len(retained) / len(pair_rows),
                "accuracy": accuracy,
                "retained": len(retained),
            }
        )
    return stats


def aggregate_accuracy(summary: dict[str, AccuracyRecord]) -> AccuracyRecord:
    return AccuracyRecord(
        correct=sum(item.correct for item in summary.values()),
        total=sum(item.total for item in summary.values()),
    )


def sample_metrics(rows: list[SampleRow]) -> tuple[float, float, float]:
    pred = np.array([row.pred for row in rows], dtype=float)
    true = np.array([row.true for row in rows], dtype=float)
    mae = np.abs(pred - true).mean()
    rmse = np.sqrt(((pred - true) ** 2).mean())
    corr = float(np.corrcoef(true, pred)[0, 1]) if len(rows) > 1 else float("nan")
    return float(mae), float(rmse), corr


def add_box(ax, xy, width, height, text, facecolor, edgecolor="#234", fontsize=10, weight="bold"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1f2933",
        weight=weight,
    )
    return box


def add_arrow(ax, start, end, color="#49657a", lw=1.4, style="-|>"):
    arrow = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=11, linewidth=lw, color=color)
    ax.add_patch(arrow)


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.06, 0.93, "Paired telemetry", fontsize=12, weight="bold", color="#334e68")
    ax.text(0.33, 0.93, "Shared encoder", fontsize=12, weight="bold", color="#334e68")
    ax.text(0.60, 0.93, "Attention pooling", fontsize=12, weight="bold", color="#334e68")
    ax.text(0.80, 0.93, "Comparative head", fontsize=12, weight="bold", color="#334e68")

    add_box(ax, (0.04, 0.60), 0.18, 0.20, "Version v1\n60 x 6 PMU/LBR sequence", "#fde6d8", edgecolor="#c46d3c")
    add_box(ax, (0.04, 0.20), 0.18, 0.20, "Version v2\n60 x 6 PMU/LBR sequence", "#dff4ef", edgecolor="#2a9d8f")
    ax.text(0.13, 0.54, "Z-score normalized\naligned windows", ha="center", va="top", fontsize=9, color="#52606d")

    add_box(ax, (0.30, 0.18), 0.22, 0.64, "", "#edf2ff", edgecolor="#2f4b7c", fontsize=12)
    ax.text(0.41, 0.77, "Shared encoder  f(theta)", ha="center", va="center", fontsize=11, color="#243b53", weight="bold")
    add_box(ax, (0.335, 0.60), 0.15, 0.10, "1D-CNN", "#f4f0e8", edgecolor="#8f6b3b", fontsize=10)
    add_box(ax, (0.335, 0.46), 0.15, 0.10, "BiLSTM", "#f4f0e8", edgecolor="#8f6b3b", fontsize=10)
    add_box(ax, (0.335, 0.32), 0.15, 0.10, "Transformer", "#f4f0e8", edgecolor="#8f6b3b", fontsize=10)
    ax.text(0.41, 0.24, "shared weights +\nsequence inductive bias", ha="center", va="center", fontsize=9, color="#52606d")

    add_box(ax, (0.58, 0.61), 0.12, 0.17, "Pool -> V1", "#fff3db", edgecolor="#c46d3c", fontsize=11)
    add_box(ax, (0.58, 0.22), 0.12, 0.17, "Pool -> V2", "#e6f4f1", edgecolor="#2a9d8f", fontsize=11)
    ax.text(0.64, 0.49, "attention masking\nignores padding", ha="center", va="center", fontsize=9, color="#52606d")

    add_box(ax, (0.76, 0.32), 0.15, 0.32, "Fusion\n[V1 ; V2 ;\nDelta V]", "#eef2f5", edgecolor="#4a6572", fontsize=11)
    add_box(ax, (0.93, 0.40), 0.06, 0.16, "MLP\n-> Y", "#dceef8", edgecolor="#2f4b7c", fontsize=11)
    ax.text(0.96, 0.33, "continuous relative\nperformance multiplier", ha="center", va="top", fontsize=9, color="#52606d")

    add_arrow(ax, (0.22, 0.70), (0.30, 0.70))
    add_arrow(ax, (0.22, 0.30), (0.30, 0.30))
    add_arrow(ax, (0.52, 0.70), (0.58, 0.70))
    add_arrow(ax, (0.52, 0.30), (0.58, 0.30))
    add_arrow(ax, (0.70, 0.69), (0.76, 0.56))
    add_arrow(ax, (0.70, 0.30), (0.76, 0.40))
    add_arrow(ax, (0.91, 0.48), (0.93, 0.48))

    ax.text(
        0.50,
        0.06,
        "The recognition target is comparative: the model suppresses program-specific baseline variation and emphasizes the temporal difference that signals which binary is faster.",
        ha="center",
        va="center",
        fontsize=10,
        color="#334e68",
    )

    # Keep the hand-authored SVG source intact; Matplotlib flattens it into a hard-to-edit export.
    save_figure(fig, "model-architecture")


def plot_ai_decision_workflow() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.08, 0.90, "Runtime evidence", fontsize=12, weight="bold", color="#334e68")
    ax.text(0.34, 0.90, "Neural comparison", fontsize=12, weight="bold", color="#334e68")
    ax.text(0.58, 0.90, "AI decision layer", fontsize=12, weight="bold", color="#334e68")
    ax.text(0.82, 0.90, "Deployment actions", fontsize=12, weight="bold", color="#334e68")

    add_box(ax, (0.03, 0.58), 0.17, 0.18, "Version v1\nPMU + LBR trace", "#fde6d8", edgecolor="#c46d3c")
    add_box(ax, (0.03, 0.24), 0.17, 0.18, "Version v2\nPMU + LBR trace", "#dff4ef", edgecolor="#2a9d8f")
    ax.text(0.115, 0.51, "single representative run\nper candidate binary", ha="center", va="top", fontsize=8.8, color="#52606d")

    add_box(ax, (0.28, 0.28), 0.17, 0.42, "Shared temporal\nencoder\n+\ncomparative head", "#edf2ff", edgecolor="#2f4b7c", fontsize=11)
    ax.text(0.365, 0.20, "learned pairwise comparator\nreplaces hand-designed rules", ha="center", va="center", fontsize=8.8, color="#52606d")

    add_box(ax, (0.53, 0.56), 0.16, 0.14, "Score s = y_hat - 1", "#fff3db", edgecolor="#c46d3c", fontsize=11)
    add_box(ax, (0.53, 0.30), 0.16, 0.14, "Gate c = |y_hat - 1|", "#e6f4f1", edgecolor="#2a9d8f", fontsize=11)
    ax.text(0.61, 0.22, "selective prediction /\nabstention signal", ha="center", va="center", fontsize=8.8, color="#52606d")

    add_box(ax, (0.77, 0.63), 0.18, 0.13, "Auto rank /\npromote faster binary", "#dceef8", edgecolor="#2f4b7c", fontsize=10)
    add_box(ax, (0.77, 0.43), 0.18, 0.13, "Fallback benchmark\nqueue for low margin", "#fce8e6", edgecolor="#d1495b", fontsize=10)
    add_box(ax, (0.77, 0.23), 0.18, 0.13, "Hotspot dashboard\nfor hard workloads", "#eef2f5", edgecolor="#4a6572", fontsize=10)

    add_arrow(ax, (0.20, 0.67), (0.28, 0.60))
    add_arrow(ax, (0.20, 0.33), (0.28, 0.38))
    add_arrow(ax, (0.45, 0.58), (0.53, 0.63))
    add_arrow(ax, (0.45, 0.40), (0.53, 0.37))
    add_arrow(ax, (0.69, 0.63), (0.77, 0.69))
    add_arrow(ax, (0.69, 0.37), (0.77, 0.49))
    add_arrow(ax, (0.61, 0.30), (0.86, 0.30), style="->")

    ax.text(
        0.50,
        0.05,
        "TraceTwin is deployed as an AI comparator with a defer option: high-margin pairs are decided automatically, while ambiguous pairs are routed to explicit measurement.",
        ha="center",
        va="center",
        fontsize=10,
        color="#334e68",
    )

    # Keep the hand-authored SVG source intact; Matplotlib flattens it into a hard-to-edit export.
    save_figure(fig, "ai-decision-workflow")


def build_best_summaries() -> dict[tuple[str, str], dict[str, AccuracyRecord]]:
    summaries: dict[tuple[str, str], dict[str, AccuracyRecord]] = {}
    for model, label, rel_path in BEST_RUNS:
        summaries[(model, label)] = parse_accuracy_summary(CHECKPOINTS_DIR / rel_path)
    return summaries


def plot_overall_accuracy_landscape(best_summaries: dict[tuple[str, str], dict[str, AccuracyRecord]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), gridspec_kw={"width_ratios": [1.7, 1.0]}, constrained_layout=True)

    x = np.arange(len(LABEL_ORDER))
    for model in MODEL_ORDER:
        values = [aggregate_accuracy(best_summaries[(model, label)]).pct for label in LABEL_ORDER]
        axes[0].plot(
            x,
            values,
            color=MODEL_COLORS[model],
            linewidth=2.2,
            marker="o",
            markersize=7,
            label=model,
        )
        for xpos, value in zip(x, values):
            axes[0].text(xpos, value + 1.1, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([LABEL_TITLES[label] for label in LABEL_ORDER])
    axes[0].set_ylim(55, 90)
    axes[0].set_ylabel("Directional Accuracy (%)")
    axes[0].set_title("Backbone comparison across supervision semantics")
    axes[0].legend(loc="upper left", ncol=3)
    axes[0].axvspan(0.8, 1.2, color="#e8f6f3", alpha=0.6)
    axes[0].text(1.0, 56.3, "best supervision zone", ha="center", va="bottom", fontsize=8, color="#2a9d8f")

    label_totals = []
    for label in LABEL_ORDER:
        overall = aggregate_accuracy(
            {
                f"{model}-{label}": aggregate_accuracy(best_summaries[(model, label)])
                for model in MODEL_ORDER
            }
        )
        label_totals.append(overall)

    y = np.arange(len(LABEL_ORDER))
    axes[1].barh(
        y,
        [item.pct for item in label_totals],
        color=[LABEL_COLORS[label] for label in LABEL_ORDER],
        height=0.58,
    )
    for ypos, label, item in zip(y, LABEL_ORDER, label_totals):
        axes[1].text(item.pct + 0.8, ypos, f"{item.pct:.2f}%", va="center", fontsize=9)
        axes[1].text(52.5, ypos, f"{item.correct}/{item.total}", va="center", fontsize=8, color="#5b6870")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([LABEL_TITLES[label] for label in LABEL_ORDER])
    axes[1].invert_yaxis()
    axes[1].set_xlim(50, 80)
    axes[1].set_xlabel("Accuracy (%)")
    axes[1].set_title("Aggregate effect of label semantics")

    save_figure(fig, "overall-accuracy-best-grouped")


def plot_overall_heatmap(best_summaries: dict[tuple[str, str], dict[str, AccuracyRecord]]) -> None:
    row_defs = [
        ("CNN", "fixed_time"),
        ("CNN", "fixed_work"),
        ("CNN", "inst"),
        ("LSTM", "fixed_time"),
        ("LSTM", "fixed_work"),
        ("LSTM", "inst"),
        ("Transformer", "fixed_time"),
        ("Transformer", "fixed_work"),
        ("Transformer", "inst"),
    ]
    matrix = []
    row_labels = []
    for model, label in row_defs:
        summary = best_summaries[(model, label)]
        overall = aggregate_accuracy(summary)
        matrix.append([summary[pair].pct for pair in PAIR_ORDER] + [overall.pct])
        row_labels.append(f"{model} / {LABEL_TITLES[label]}")

    data = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(8.7, 5.9), constrained_layout=True)
    image = ax.imshow(data, cmap=ACC_CMAP, vmin=58, vmax=88, aspect="auto")
    ax.set_title("Recognition accuracy by optimization pair and model family")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([PAIR_SHORT[pair] for pair in PAIR_ORDER] + ["Overall"])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            ax.text(col, row, f"{data[row, col]:.1f}", ha="center", va="center", fontsize=8, color="#102a43")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Directional Accuracy (%)")

    save_figure(fig, "overall-accuracy-best-heatmap")


def plot_transformer_label_metrics() -> None:
    summaries = {label: parse_accuracy_summary(CHECKPOINTS_DIR / path) for label, path in TRANSFORMER_LABEL_RUNS.items()}
    rows = {label: parse_samples(CHECKPOINTS_DIR / path) for label, path in TRANSFORMER_LABEL_RUNS.items()}

    direction_values = [aggregate_accuracy(summaries[label]).pct for label in LABEL_ORDER]
    rmse_values = [sample_metrics(rows[label])[1] for label in LABEL_ORDER]
    corr_values = [sample_metrics(rows[label])[2] for label in LABEL_ORDER]

    panels = [
        (direction_values, "Directional accuracy (%)", (55, 90), "accuracy"),
        (rmse_values, "RMSE", (0.0, 0.12), "rmse"),
        (corr_values, "Pearson r", (0.5, 1.02), "corr"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.1, 4.1), constrained_layout=True)
    x = np.arange(len(LABEL_ORDER))

    for axis, (values, ylabel, ylim, kind) in zip(axes, panels):
        colors = [LABEL_COLORS[label] for label in LABEL_ORDER]
        bars = axis.bar(x, values, color=colors, width=0.62)
        axis.set_xticks(x)
        axis.set_xticklabels([LABEL_TITLES[label] for label in LABEL_ORDER], rotation=12)
        axis.set_ylim(*ylim)
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel)
        for bar, value in zip(bars, values):
            fmt = f"{value:.2f}" if kind == "accuracy" else f"{value:.4f}"
            axis.text(bar.get_x() + bar.get_width() / 2, value + (ylim[1] - ylim[0]) * 0.03, fmt, ha="center", va="bottom", fontsize=8)
        if kind == "accuracy":
            axis.axhline(83.44, color="#2a9d8f", linewidth=1.0, linestyle=":")

    fig.suptitle("Same Transformer backbone, different label semantics", fontsize=13)

    save_figure(fig, "transformer-label-metrics")


def plot_fixed_work_scatter() -> None:
    rows = parse_samples(CHECKPOINTS_DIR / TRANSFORMER_LABEL_RUNS["fixed_work"])
    grouped: dict[str, list[SampleRow]] = {pair: [] for pair in PAIR_ORDER}
    for row in rows:
        grouped[row.pair].append(row)

    all_pred = np.array([row.pred for row in rows], dtype=float)
    all_true = np.array([row.true for row in rows], dtype=float)
    lower = min(all_true.min(), all_pred.min()) - 0.03
    upper = max(all_true.max(), all_pred.max()) + 0.03

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), sharex=True, sharey=True, constrained_layout=True)

    for axis, pair in zip(axes, PAIR_ORDER):
        pair_rows = grouped[pair]
        x = np.array([row.true for row in pair_rows], dtype=float)
        y = np.array([row.pred for row in pair_rows], dtype=float)
        mae, rmse, corr = sample_metrics(pair_rows)

        band = np.linspace(lower, upper, 200)
        axis.fill_between(band, band - 0.03, band + 0.03, color="#ecf6f4", alpha=0.85)
        axis.scatter(
            x,
            y,
            s=28,
            alpha=0.82,
            color=PAIR_COLORS[pair],
            edgecolors="white",
            linewidths=0.5,
        )
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="#334e68", linewidth=1.1)
        axis.set_title(f"{PAIR_PANEL[pair]}\n{PAIR_SHORT[pair]}", fontsize=11)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_xlabel("Ground-truth multiplier")
        axis.set_ylabel("Predicted multiplier")
        axis.text(
            0.04,
            0.96,
            f"n={len(pair_rows)}\nMAE={mae:.4f}\nRMSE={rmse:.4f}\nr={corr:.3f}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd2d9", "alpha": 0.95},
        )

    fig.suptitle("Fixed-work Transformer: calibration of relative gain prediction", fontsize=13)

    save_figure(fig, "fixed-work-transformer-scatter")


def plot_margin_difficulty() -> None:
    rows = parse_samples(CHECKPOINTS_DIR / TRANSFORMER_LABEL_RUNS["fixed_work"])
    bucket_defs = [
        ("Near-tie\n|Y-1| <= 0.02", lambda item: item.gain_gap <= 0.02),
        ("Moderate\n0.02 < |Y-1| <= 0.05", lambda item: 0.02 < item.gain_gap <= 0.05),
        ("Clear\n|Y-1| > 0.05", lambda item: item.gain_gap > 0.05),
    ]
    row_defs = [("Overall", rows)] + [(PAIR_SHORT[pair], [row for row in rows if row.pair == pair]) for pair in PAIR_ORDER]

    value_matrix = np.zeros((len(row_defs), len(bucket_defs)), dtype=float)
    count_matrix = np.zeros_like(value_matrix, dtype=int)
    for row_index, (_label, group_rows) in enumerate(row_defs):
        for col_index, (_bucket_label, predicate) in enumerate(bucket_defs):
            bucket_rows = [row for row in group_rows if predicate(row)]
            count_matrix[row_index, col_index] = len(bucket_rows)
            if bucket_rows:
                value_matrix[row_index, col_index] = 100.0 * sum(direction_ok(row.pred, row.true) for row in bucket_rows) / len(bucket_rows)

    fig, ax = plt.subplots(figsize=(8.3, 4.8), constrained_layout=True)
    image = ax.imshow(value_matrix, cmap=ACC_CMAP, vmin=70, vmax=100, aspect="auto")
    ax.set_title("Recognition difficulty rises sharply near the decision boundary")
    ax.set_xticks(np.arange(len(bucket_defs)))
    ax.set_xticklabels([label for label, _predicate in bucket_defs])
    ax.set_yticks(np.arange(len(row_defs)))
    ax.set_yticklabels([label for label, _group_rows in row_defs])
    for row_index in range(value_matrix.shape[0]):
        for col_index in range(value_matrix.shape[1]):
            ax.text(
                col_index,
                row_index,
                f"{value_matrix[row_index, col_index]:.1f}%\n(n={count_matrix[row_index, col_index]})",
                ha="center",
                va="center",
                fontsize=8,
                color="#102a43",
            )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Directional Accuracy (%)")

    save_figure(fig, "fixed-work-difficulty-buckets")


def plot_family_hotspots() -> None:
    rows = parse_samples(CHECKPOINTS_DIR / TRANSFORMER_LABEL_RUNS["fixed_work"])
    stats = build_family_hotspots(rows)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.6),
        gridspec_kw={"width_ratios": [1.3, 0.9]},
        constrained_layout=True,
    )
    y = np.arange(len(stats))
    accuracies = [float(item["accuracy"]) for item in stats]
    errors = [float(item["error_share"]) for item in stats]

    axes[0].hlines(y, 35, accuracies, color="#d9e2ec", linewidth=5.0, zorder=1)
    scatter = axes[0].scatter(
        accuracies,
        y,
        s=[int(item["total"]) * 16 for item in stats],
        c=[float(item["mean_gap"]) for item in stats],
        cmap="cividis",
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    for ypos, item in zip(y, stats):
        accuracy = float(item["accuracy"])
        total = int(item["total"])
        mae = float(item["mae"])
        if accuracy >= 84:
            text_x = accuracy - 5.4
            ha = "right"
        else:
            text_x = accuracy + 1.1
            ha = "left"
        axes[0].text(
            text_x,
            ypos,
            f"n={total}, MAE={mae:.3f}",
            ha=ha,
            va="center",
            fontsize=8,
            color="#334e68",
            bbox={"facecolor": "#fcfbf7", "edgecolor": "none", "pad": 0.18, "alpha": 0.88},
        )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([str(item["family"]) for item in stats])
    axes[0].invert_yaxis()
    axes[0].set_xlim(35, 102)
    axes[0].set_xlabel("Directional accuracy (%)")
    axes[0].set_title("Benchmark-family screening quality")
    colorbar = fig.colorbar(scatter, ax=axes[0], shrink=0.86)
    colorbar.set_label("Mean |Y-1|")

    bars = axes[1].barh(y, errors, color=HOTSPOT_COLOR, height=0.58)
    for bar, item in zip(bars, stats):
        axes[1].text(
            bar.get_width() + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{int(item['wrong'])} errors",
            va="center",
            fontsize=8,
            color="#58151c",
        )
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(errors) + 10)
    axes[1].set_xlabel("Share of shown direction mistakes (%)")
    axes[1].set_title("Error hotspots")

    fig.suptitle("Fixed-work Transformer: errors concentrate in a small set of benchmark families", fontsize=13)
    save_figure(fig, "family-hotspot-analysis")


def plot_selective_screening_tradeoff() -> None:
    rows = parse_samples(CHECKPOINTS_DIR / TRANSFORMER_LABEL_RUNS["fixed_work"])
    screening = build_selective_screening(rows, SCREENING_THRESHOLDS)
    operating_point = selective_pair_operating_point(rows, threshold=0.02)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, 4.5),
        gridspec_kw={"width_ratios": [1.15, 0.95]},
        constrained_layout=True,
    )

    coverages = [float(item["coverage"]) for item in screening]
    accuracies = [float(item["accuracy"]) for item in screening]
    retained = [int(item["retained"]) for item in screening]
    baseline = accuracies[0]

    axes[0].plot(coverages, accuracies, color=RISK_CURVE_COLOR, linewidth=2.2, marker="o", markersize=6)
    axes[0].scatter(
        coverages,
        accuracies,
        s=[value * 0.55 for value in retained],
        color="#ff9f1c",
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    axes[0].axhline(baseline, color="#c46d3c", linestyle=":", linewidth=1.1)
    axes[0].text(97.5, baseline + 0.35, "full set baseline", ha="right", va="bottom", fontsize=8, color="#9b5d32")
    for item in screening:
        threshold = float(item["threshold"])
        label = f"{threshold:.3f}".rstrip("0").rstrip(".")
        axes[0].text(
            float(item["coverage"]) + 1.1,
            float(item["accuracy"]) + 0.25,
            f"tau={label}",
            fontsize=8,
            color="#243b53",
        )
    axes[0].set_xlim(0, 102)
    axes[0].set_ylim(82, 101)
    axes[0].set_xlabel("Coverage retained (%)")
    axes[0].set_ylabel("Directional accuracy (%)")
    axes[0].set_title("Margin-based selective screening")

    xpos = np.arange(len(PAIR_ORDER))
    width = 0.34
    accuracy_bars = axes[1].bar(
        xpos - width / 2,
        [float(item["accuracy"]) for item in operating_point],
        width=width,
        color="#2a9d8f",
        label="Accuracy",
    )
    coverage_bars = axes[1].bar(
        xpos + width / 2,
        [float(item["coverage"]) for item in operating_point],
        width=width,
        color="#c46d3c",
        label="Coverage",
    )
    for bar, item in zip(accuracy_bars, operating_point):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"{float(item['accuracy']):.1f}", ha="center", va="bottom", fontsize=8)
    for bar, item in zip(coverage_bars, operating_point):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"{float(item['coverage']):.1f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels([PAIR_SHORT[pair] for pair in PAIR_ORDER], rotation=10)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Percent (%)")
    axes[1].set_title("Deployment point at tau = 0.02")
    axes[1].legend(loc="upper right")

    fig.suptitle("Prediction margin can route low-confidence comparisons to fallback measurement", fontsize=13)
    save_figure(fig, "confidence-screening-tradeoff")


def plot_margin_outcome_separation() -> None:
    rows = parse_samples(CHECKPOINTS_DIR / TRANSFORMER_LABEL_RUNS["fixed_work"])
    correct_margin = np.array([abs(row.pred - 1.0) for row in rows if direction_ok(row.pred, row.true)], dtype=float)
    wrong_margin = np.array([abs(row.pred - 1.0) for row in rows if not direction_ok(row.pred, row.true)], dtype=float)
    bins = np.linspace(0.0, 0.12, 18)
    support = np.linspace(0.0, 0.12, 120)

    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.4), constrained_layout=True)

    axes[0].hist(correct_margin, bins=bins, density=True, alpha=0.72, color="#2a9d8f", label="Correct direction")
    axes[0].hist(wrong_margin, bins=bins, density=True, alpha=0.75, color="#d1495b", label="Wrong direction")
    axes[0].axvline(0.02, color="#334e68", linestyle="--", linewidth=1.0)
    axes[0].axvline(0.05, color="#7b341e", linestyle=":", linewidth=1.1)
    axes[0].set_xlabel("Predicted margin |y_hat - 1|")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Margin distribution by outcome")
    axes[0].legend(loc="upper right")

    wrong_cdf = [100.0 * np.mean(wrong_margin <= value) for value in support]
    correct_cdf = [100.0 * np.mean(correct_margin <= value) for value in support]
    axes[1].plot(support, wrong_cdf, color="#d1495b", linewidth=2.2, label="Wrong predictions")
    axes[1].plot(support, correct_cdf, color="#2a9d8f", linewidth=2.0, label="Correct predictions")
    for threshold, label in [(0.02, "tau=0.02"), (0.05, "tau=0.05")]:
        captured = 100.0 * np.mean(wrong_margin <= threshold)
        axes[1].axvline(threshold, color="#334e68", linestyle="--", linewidth=0.9)
        axes[1].text(threshold + 0.002, captured + 2.0, f"{label}\n{captured:.1f}% wrong captured", fontsize=8, color="#243b53")
    axes[1].set_xlim(0.0, 0.12)
    axes[1].set_ylim(0, 102)
    axes[1].set_xlabel("Predicted margin threshold")
    axes[1].set_ylabel("Cumulative share below threshold (%)")
    axes[1].set_title("Why abstention works")
    axes[1].legend(loc="lower right")

    fig.suptitle("Most screening mistakes cluster in the low-margin region", fontsize=13)
    save_figure(fig, "margin-outcome-separation")


def plot_transformer_variants() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), sharey=True, constrained_layout=True)

    for axis, label in zip(axes, LABEL_ORDER):
        entries = TRANSFORMER_VARIANTS[label]
        names = []
        values = []
        for name, rel_path in entries:
            summary = parse_accuracy_summary(CHECKPOINTS_DIR / rel_path)
            names.append(name)
            values.append(aggregate_accuracy(summary).pct)
        xpos = np.arange(len(names))
        axis.plot(
            xpos,
            values,
            color=LABEL_COLORS[label],
            linewidth=2.0,
            marker="o",
            markersize=6,
        )
        axis.scatter(xpos[-1], values[-1], s=95, marker="*", color="#1d3557", zorder=4)
        for x_value, y_value in zip(xpos, values):
            axis.text(x_value, y_value + 0.8, f"{y_value:.1f}", ha="center", va="bottom", fontsize=8)
        axis.set_xticks(xpos)
        axis.set_xticklabels(names)
        axis.set_title(LABEL_TITLES[label])
        axis.set_ylim(55, 90)
        axis.set_yticks(np.arange(55, 91, 5))
        axis.set_xlabel("Archived run")

    axes[0].set_ylabel("Overall Accuracy (%)")
    fig.suptitle("Archived Transformer runs: the retained checkpoint is substantially stronger", fontsize=13)

    save_figure(fig, "transformer-variant-accuracy")


def main() -> None:
    configure_style()
    ensure_output_dir()
    best_summaries = build_best_summaries()
    plot_architecture()
    plot_ai_decision_workflow()
    plot_overall_accuracy_landscape(best_summaries)
    plot_overall_heatmap(best_summaries)
    plot_transformer_label_metrics()
    plot_fixed_work_scatter()
    plot_margin_difficulty()
    plot_family_hotspots()
    plot_selective_screening_tradeoff()
    plot_margin_outcome_separation()
    plot_transformer_variants()
    print(f"saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()