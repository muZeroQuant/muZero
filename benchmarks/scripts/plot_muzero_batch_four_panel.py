from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 26,
        "axes.labelsize": 26,
        "xtick.labelsize": 20,
        "ytick.labelsize": 22,
        "legend.fontsize": 22,
        "figure.titlesize": 19,
    }
)

GRID_KW = {"alpha": 0.4, "linewidth": 0.6, "which": "both"}
BBOX_STYLE = {"facecolor": "white", "alpha": 0.85, "edgecolor": "none"}

METHOD_ORDER = ["bf16", "mu_zero_4bit", "mu_zero_2bit"]
METHOD_LABELS = {
    "bf16": "BF16",
    "mu_zero_4bit": r"$\mu$-Zero 4-bit",
    "mu_zero_2bit": r"$\mu$-Zero 2-bit",
}
METHOD_COLORS = {
    "bf16": "#9a9a9a",
    "mu_zero_4bit": "#2a9d8f",
    "mu_zero_2bit": "#6a4c93",
}
METHOD_MARKERS = {
    "bf16": "o",
    "mu_zero_4bit": "s",
    "mu_zero_2bit": "D",
}


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    parsed = parse_float(value)
    return None if parsed is None else int(parsed)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed = dict(row)
            for key in ["sweep_value", "context_len", "new_tokens", "batch_size", "generated_tokens"]:
                if key in parsed:
                    parsed[key] = parse_int(parsed[key])
            for key in [
                "tokens_per_sec",
                "speedup_vs_bf16",
                "peak_mem_mb",
                "peak_reserved_mb",
                "cache_total_mb",
                "avg_latency_s",
            ]:
                if key in parsed:
                    parsed[key] = parse_float(parsed[key])
            rows.append(parsed)
        return rows


def resolve_csv(input_dir: Path, explicit_csv: str | None) -> Path:
    if explicit_csv:
        path = Path(explicit_csv).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    preferred = input_dir / "result1.csv"
    if preferred.exists():
        return preferred
    candidates = sorted(
        path for path in input_dir.glob("result*.csv") if ".details" not in path.name
    )
    if not candidates:
        raise FileNotFoundError(f"No result*.csv files found under {input_dir}")
    return candidates[-1]


def filtered_rows(rows: list[dict[str, Any]], experiment: str, min_batch: int = 0) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("experiment") == experiment
        and row.get("method") in METHOD_ORDER
        and row.get("batch_size") is not None
        and int(row["batch_size"]) >= min_batch
    ]


def rows_by_method(rows: list[dict[str, Any]], sort_key: str) -> dict[str, list[dict[str, Any]]]:
    grouped = {method: [] for method in METHOD_ORDER}
    for row in rows:
        grouped[str(row["method"])].append(row)
    for method_rows in grouped.values():
        method_rows.sort(key=lambda row: int(row[sort_key]))
    return grouped


def plot_metric_panel(
    ax: plt.Axes,
    grouped: dict[str, list[dict[str, Any]]],
    *,
    metric: str,
    title: str,
    x_field: str,
    xlabel: str,
    ylabel: str,
    x_scale: float = 1.0,
    scale: float = 1.0,
    y_min: float | None = None,
    yscale: str = "linear",
) -> None:
    for method in METHOD_ORDER:
        method_rows = [row for row in grouped[method] if row.get("status") == "ok" and row.get(metric) is not None]
        if not method_rows:
            continue
        x = [int(row[x_field]) / x_scale for row in method_rows]
        y = [float(row[metric]) / scale for row in method_rows]
        ax.plot(
            x,
            y,
            marker=METHOD_MARKERS[method],
            markersize=6,
            linewidth=3.2,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            zorder=3,
    )
    # ax.set_title(title, fontweight="bold", pad=16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(**GRID_KW)
    ax.set_yscale(yscale)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)


def annotate_capacity(ax: plt.Axes, grouped: dict[str, list[dict[str, Any]]]) -> None:
    lines = []
    for method in METHOD_ORDER:
        ok_batches = [int(row["batch_size"]) for row in grouped[method] if row.get("status") == "ok"]
        oom_batches = [int(row["batch_size"]) for row in grouped[method] if row.get("status") == "oom"]
        if ok_batches:
            line = f"{METHOD_LABELS[method]} max ok: {max(ok_batches)}"
            later_oom_batches = [batch for batch in oom_batches if batch > max(ok_batches)]
            if later_oom_batches:
                line += f"  (first OOM: {min(later_oom_batches)})"
            lines.append(line)
        elif oom_batches:
            lines.append(f"{METHOD_LABELS[method]} first OOM: {min(oom_batches)}")
    if lines:
        ax.text(
            0.03,
            0.97,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            bbox=BBOX_STYLE,
        )


def build_figure(input_csv: Path, context_csv: Path, output_path: Path, min_batch: int) -> None:
    batch_data = filtered_rows(read_rows(input_csv), "batch", min_batch)
    if not batch_data:
        raise ValueError(f"No batch rows with batch_size >= {min_batch} found in {input_csv}")
    batch_grouped = rows_by_method(batch_data, "batch_size")

    context_data = filtered_rows(read_rows(context_csv), "context")
    if not context_data:
        raise ValueError(f"No context rows found in {context_csv}")
    context_grouped = rows_by_method(context_data, "context_len")

    fig, axes = plt.subplots(1, 4, figsize=(24, 4), constrained_layout=True)
    plot_metric_panel(
        axes[0],
        batch_grouped,
        metric="tokens_per_sec",
        title="(a) Throughput",
        x_field="batch_size",
        xlabel="Batch size",
        ylabel="Tokens / second",
        y_min=300,
    )
    plot_metric_panel(
        axes[1],
        batch_grouped,
        metric="peak_reserved_mb",
        title="(b) Reserved memory",
        x_field="batch_size",
        xlabel="Batch size",
        ylabel="Peak memory (GB)",
        scale=1024.0,
        y_min=10,
    )
    plot_metric_panel(
        axes[2],
        context_grouped,
        metric="tokens_per_sec",
        title="(c) Long-context throughput",
        x_field="context_len",
        xlabel="Context length (K tokens)",
        ylabel="Tokens / second",
        x_scale=1024.0,
        y_min=0,
    )
    plot_metric_panel(
        axes[3],
        context_grouped,
        metric="peak_reserved_mb",
        title="(d) Long-context memory",
        x_field="context_len",
        xlabel="Context length (K tokens)",
        ylabel="Peak memory (GB)",
        x_scale=1024.0,
        scale=1024.0,
        y_min=10,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linewidth=2.2,
            markersize=7,
            label=METHOD_LABELS[method],
        )
        for method in METHOD_ORDER
    ]
    axes[0].legend(handles=legend_handles, loc="lower right", frameon=True, labelspacing=0.25, handletextpad=0.4,
        borderpad=0.2,
        borderaxespad=0.2,)
    axes[1].legend(handles=legend_handles, loc="lower right", frameon=True, labelspacing=0.25, handletextpad=0.4,
        borderpad=0.2,
        borderaxespad=0.2,)
    
    axes[2].legend(handles=legend_handles, loc="upper right", frameon=True, labelspacing=0.25, handletextpad=0.4,
        borderpad=0.2,
        borderaxespad=0.2,)
    axes[3].legend(handles=legend_handles, loc="lower right", frameon=True, labelspacing=0.25, handletextpad=0.4,
        borderpad=0.2,
        borderaxespad=0.2,)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot MuZero batch-scaling results as four parallel panels.")
    parser.add_argument("--input-dir", default="/root/autodl-tmp/muZero/runs/muzero_best_batch_4096")
    parser.add_argument("--csv", default="", help="Optional explicit result CSV. Defaults to result1.csv if present.")
    parser.add_argument("--context-csv", default="/root/autodl-tmp/muZero/runs/muzero_ctx_sweep_batch64/result.csv")
    parser.add_argument("--output", default="", help="Output image path. Defaults to <input-dir>/batch_four_panel.png")
    parser.add_argument("--min-batch", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    input_csv = resolve_csv(input_dir, args.csv or None)
    context_csv = Path(args.context_csv).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else input_dir / "batch_four_panel.png"
    build_figure(input_csv, context_csv, output_path, args.min_batch)
    print(f"[plot] input: {input_csv}")
    print(f"[plot] context input: {context_csv}")
    print(f"[plot] wrote: {output_path}")


if __name__ == "__main__":
    main()
