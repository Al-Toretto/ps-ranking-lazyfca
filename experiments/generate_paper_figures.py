"""Generate manuscript figures in journal-ready formats.

The journal requests PS, JPEG, or TIFF figures. The plots are color line
figures, so the raster outputs are saved at 1200 dpi to satisfy the stricter
line-art requirement, and PostScript copies are also written for vector output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


DATASET_ORDER = [
    "breast_cancer",
    "ionosphere",
    "parkinsons",
    "rice",
    "sonar",
    "spambase",
    "waveform",
    "vehicle",
    "page_blocks",
    "glass",
    "image_segmentation",
]

DATASET_LABELS = {
    "breast_cancer": "Breast cancer",
    "ionosphere": "Ionosphere",
    "parkinsons": "Parkinsons",
    "rice": "Rice",
    "sonar": "Sonar",
    "spambase": "Spambase",
    "waveform": "Waveform",
    "vehicle": "Vehicle",
    "page_blocks": "Page blocks",
    "glass": "Glass",
    "image_segmentation": "Image seg.",
}

COMPACT_METRICS = [
    ("global_topk", "query_weighted_precision", "Query-weighted precision", "#d62728", "-", "s"),
    ("global_topk", "precision", "Precision", "#2ca02c", "-", "^"),
    ("global_topk", "description_volume", "Description volume", "#9467bd", "--", "D"),
    ("global_topk", "query_weighted_log_odds_ratio", "Query-weighted log-odds-like", "#8c564b", "--", "P"),
    ("global_topk", "log_odds_ratio", "Log-odds-like", "#ff7f0e", "--", "v"),
    ("random_topk", "random", "Random top-k", "#7f7f7f", ":", "x"),
]

FULL_METRICS = [
    ("global_topk", "query_weighted_precision", "QW precision", "#d62728", "-"),
    ("global_topk", "query_similarity", "Query similarity", "#2ca02c", "-"),
    ("global_topk", "precision", "Precision", "#ff7f0e", "-"),
    ("global_topk", "description_volume", "Descr. volume", "#9467bd", "-"),
    ("global_topk", "query_weighted_log_odds_ratio", "QW LOR", "#1f77b4", "-"),
    ("global_topk", "log_odds_ratio", "LOR", "#8c564b", "-"),
    ("random_topk", "random", "Random", "#7f7f7f", "--"),
]

RASTER_DPI = 1200


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "legend.fontsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "ps.fonttype": 42,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def load_plot_data(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "topk_plot_data.csv"
    data = pd.read_csv(path)
    return data[data["dataset"].isin(DATASET_ORDER)].copy()


def metric_curve(data: pd.DataFrame, dataset: str, method: str, metric: str) -> pd.DataFrame:
    mask = (
        (data["dataset"] == dataset)
        & (data["method"] == method)
        & (data["metric"] == metric)
    )
    return data.loc[mask].sort_values("k")


def save_all_formats(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compile-friendly and archival copy.
    fig.savefig(output_dir / f"{stem}.png", dpi=RASTER_DPI)

    # Journal-preferred raster formats.
    fig.savefig(
        output_dir / f"{stem}.jpg",
        dpi=RASTER_DPI,
        pil_kwargs={"quality": 95, "optimize": True},
    )
    tiff_path = output_dir / f"{stem}.tif"
    fig.savefig(tiff_path, dpi=RASTER_DPI)
    convert_tiff_to_rgb(tiff_path)

    # Journal-preferred vector/line-art format.
    fig.savefig(output_dir / f"{stem}.ps", format="ps")


def convert_tiff_to_rgb(path: Path) -> None:
    with Image.open(path) as image:
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            rgb = background
        else:
            rgb = image.convert("RGB")
        rgb.save(path, dpi=(RASTER_DPI, RASTER_DPI), compression="tiff_lzw")


def plot_compact(data: pd.DataFrame) -> plt.Figure:
    compact = data[data["k"].between(1, 10)].copy()
    fig, axes = plt.subplots(6, 2, figsize=(5.4, 8.7), sharex=False)
    flat_axes = axes.ravel()

    for ax, dataset in zip(flat_axes, DATASET_ORDER):
        for method, metric, label, color, linestyle, marker in COMPACT_METRICS:
            curve = metric_curve(compact, dataset, method, metric)
            if curve.empty:
                continue
            ax.plot(
                curve["k"],
                curve["primary_f1_mean"],
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=2.7,
                linewidth=1.1 if metric != "query_weighted_precision" else 1.5,
            )
        ax.set_title(DATASET_LABELS[dataset], pad=2.0)
        ax.set_xlim(1, 10)
        ax.set_xticks(range(1, 11))
        ax.grid(True, color="#d9d9d9", linewidth=0.35)
        ax.set_ylabel("Macro-F1")
        ax.set_xlabel("Retained candidates $k$", labelpad=1.5)

    for ax in flat_axes[len(DATASET_ORDER) :]:
        ax.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.012),
        columnspacing=1.8,
        handlelength=2.3,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.0, w_pad=1.0)
    return fig


def plot_full_grid(data: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(4, 3, figsize=(6.27, 6.27))
    flat_axes = axes.ravel()

    for ax, dataset in zip(flat_axes, DATASET_ORDER):
        max_k = 1
        for method, metric, label, color, linestyle in FULL_METRICS:
            curve = metric_curve(data, dataset, method, metric)
            if curve.empty:
                continue
            max_k = max(max_k, int(curve["k"].max()))
            ax.plot(
                curve["k"],
                curve["primary_f1_mean"],
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=1.15 if metric != "query_weighted_precision" else 1.55,
            )
        ax.set_title(DATASET_LABELS[dataset], pad=2.0)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.grid(True, color="#d9d9d9", linewidth=0.35)
        ax.set_xlabel("retained candidates $k$ (log scale)")
        ax.set_ylabel("macro-F1")
        ax.axvline(max_k, color="#bdbdbd", linewidth=0.45, linestyle=":")
        ax.text(
            0.98,
            0.04,
            "full pool",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5,
            color="#555555",
        )

    for ax in flat_axes[len(DATASET_ORDER) :]:
        ax.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
        columnspacing=1.0,
        handlelength=2.2,
    )
    fig.suptitle("Macro-F1 over the full retained-candidate grid", y=0.99, fontsize=10)
    fig.tight_layout(rect=(0, 0.045, 1, 0.955), h_pad=1.0, w_pad=0.9)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/paper"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paper/figures"),
    )
    args = parser.parse_args()

    configure_matplotlib()
    data = load_plot_data(args.results_dir)

    compact_fig = plot_compact(data)
    save_all_formats(compact_fig, args.output_dir, "compact_metric_curves_k1_10_all_datasets")
    plt.close(compact_fig)

    full_fig = plot_full_grid(data)
    save_all_formats(full_fig, args.output_dir, "full_k_metric_curves_all_datasets")
    plt.close(full_fig)


if __name__ == "__main__":
    main()
