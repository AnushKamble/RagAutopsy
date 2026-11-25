"""Static evaluation visualizers: confusion matrix heatmaps and bootstrap CI plots."""

import json
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless / CI / laptop environments
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. Install it with `pip install rag-autopsy[plots]` "
            "or `pip install matplotlib`."
        )


def plot_confusion_matrix(
    matrix: list[list[float]] | Any,
    labels: list[str],
    title: str = "Attribution Confusion Matrix",
    output_path: str | Path | None = None,
    normalize: bool = True,
    cmap: str = "Blues",
) -> Any:
    """Plot a single confusion matrix heatmap with cell annotations.

    Args:
        matrix: 2D array or list of lists containing confusion matrix values.
        labels: Class label names for axes.
        title: Plot title.
        output_path: File path to save PNG. If None, figure is returned without saving.
        normalize: If True, format values as percentages (0.0% - 100.0%).
        cmap: Matplotlib colormap name.

    Returns:
        Matplotlib Figure object.
    """
    _check_matplotlib()

    arr = np.array(matrix, dtype=float)
    n_rows, n_cols = arr.shape

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0 if normalize else None)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    if normalize:
        cbar.set_label("Proportion of True Class", rotation=270, labelpad=15, fontsize=10)

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel("Predicted Cause Label", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Injected Fault Label", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    thresh = (arr.max() + arr.min()) / 2.0 if arr.size > 0 else 0.5
    for i in range(n_rows):
        for j in range(n_cols):
            val = arr[i, j]
            if normalize:
                text_val = f"{val * 100:.1f}%" if val > 0 else "0.0%"
            else:
                text_val = f"{int(val)}"
            color = "white" if val > thresh else "black"
            ax.text(j, i, text_val, ha="center", va="center", color=color, fontsize=9, fontweight="semibold")

    fig.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_regime_comparison(
    matrix_a: list[list[float]] | Any,
    matrix_b: list[list[float]] | Any,
    labels: list[str],
    title: str = "D4 Policy Identifiability: Regime A vs Regime B",
    output_path: str | Path | None = None,
) -> Any:
    """Side-by-side comparison of Regime A (Candidate Pool) vs Regime B (Blind Top-k).

    Demonstrates the identifiability collapse when candidate pool scores are hidden.
    """
    _check_matplotlib()

    arr_a = np.array(matrix_a, dtype=float)
    arr_b = np.array(matrix_b, dtype=float)
    n_rows, n_cols = arr_a.shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, arr, subtitle in [
        (ax1, arr_a, "Regime A (Candidate Pool Scores Visible)\nF2 Identified via Score Cliff (F1=0.701)"),
        (ax2, arr_b, "Regime B (Blind Top-k Retrieval)\nF2 Collapses into F0/Clean (F1=0.468)"),
    ]:
        im = ax.imshow(arr, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel("Predicted Cause", fontsize=10, fontweight="bold")
        ax.set_ylabel("True Injected Fault", fontsize=10, fontweight="bold")
        ax.set_title(subtitle, fontsize=10.5, pad=10)

        thresh = 0.5
        for i in range(n_rows):
            for j in range(n_cols):
                val = arr[i, j]
                text_val = f"{val * 100:.1f}%" if val > 0 else "0.0%"
                color = "white" if val > thresh else "black"
                ax.text(j, i, text_val, ha="center", va="center", color=color, fontsize=8, fontweight="semibold")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_bootstrap_macro_f1(
    summary_data: dict[str, Any],
    title: str = "Held-Out Test Attribution Performance (95% Clustered Bootstrap CI)",
    output_path: str | Path | None = None,
) -> Any:
    """Plot Macro-F1 comparison with error bars representing 95% bootstrap confidence intervals."""
    _check_matplotlib()

    diagnostics = summary_data.get("diagnostic_results", {})
    names = []
    means = []
    err_lowers = []
    err_uppers = []

    display_names = {
        "baseline_majority_cause": "Majority Baseline",
        "baseline_random": "Random Baseline",
        "d1_retrieval_rules": "D1 Retrieval Rules",
        "d2_lexical_support": "D2 Lexical Support",
        "d4_combined_policy": "D4 Combined Policy",
    }

    colors = ["#7f7f7f", "#bcbd22", "#1f77b4", "#ff7f0e", "#2ca02c"]

    for key, d in diagnostics.items():
        name = display_names.get(key, key)
        bs = d.get("bootstrap", {})
        f1_data = bs.get("macro_f1", {})
        mean = f1_data.get("mean", d.get("metrics", {}).get("macro_f1", 0.0))
        ci_lower = f1_data.get("ci_lower", mean)
        ci_upper = f1_data.get("ci_upper", mean)

        names.append(name)
        means.append(mean)
        err_lowers.append(max(0.0, mean - ci_lower))
        err_uppers.append(max(0.0, ci_upper - mean))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_pos = np.arange(len(names))
    yerr = [err_lowers, err_uppers]

    bars = ax.bar(x_pos, means, yerr=yerr, capsize=5, color=colors[:len(names)], alpha=0.85, edgecolor="black", linewidth=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9.5)
    ax.set_ylabel("Attribution Macro-F1", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=12)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.annotate(
            f"{mean:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def generate_all_evaluation_plots(
    regime_a_summary_path: str | Path = "results/test_regime_a/evaluation_summary.json",
    regime_b_summary_path: str | Path | None = "results/test_regime_b/evaluation_summary.json",
    output_dir: str | Path = "results/plots",
) -> dict[str, str]:
    """Generate all standard evaluation plots from saved JSON summaries.

    Returns:
        Dict mapping plot descriptions to saved file paths.
    """
    _check_matplotlib()

    p_a = Path(regime_a_summary_path)
    if not p_a.exists():
        raise FileNotFoundError(f"Regime A summary file not found at: {p_a}")

    with open(p_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = {}

    # 1. Regime A D4 Confusion Matrix
    d4_a = data_a.get("diagnostic_results", {}).get("d4_combined_policy", {})
    if d4_a:
        metrics_a = d4_a.get("metrics", {})
        norm_cm_a = metrics_a.get("normalized_confusion_matrix")
        labels = metrics_a.get("confusion_matrix_labels", ["F0", "F1", "F2", "F3", "F4", "UNKNOWN"])
        if norm_cm_a:
            p_cm = out_dir / "regime_a_d4_confusion_matrix.png"
            plot_confusion_matrix(
                norm_cm_a,
                labels,
                title="D4 Combined Policy Confusion Matrix (Regime A: Candidate Pool Visible)",
                output_path=p_cm,
            )
            generated["regime_a_d4_cm"] = str(p_cm)

    # 2. Bootstrap Macro-F1 Bar Chart
    p_bar = out_dir / "regime_a_macro_f1_bootstrap.png"
    plot_bootstrap_macro_f1(data_a, output_path=p_bar)
    generated["regime_a_bootstrap_bar"] = str(p_bar)

    # 3. Regime Comparison (if Regime B summary exists)
    if regime_b_summary_path and Path(regime_b_summary_path).exists():
        with open(regime_b_summary_path, "r", encoding="utf-8") as f:
            data_b = json.load(f)

        d4_b = data_b.get("diagnostic_results", {}).get("d4_combined_policy", {})
        if d4_a and d4_b:
            cm_a = d4_a.get("metrics", {}).get("normalized_confusion_matrix")
            cm_b = d4_b.get("metrics", {}).get("normalized_confusion_matrix")
            labels = d4_a.get("metrics", {}).get("confusion_matrix_labels", ["F0", "F1", "F2", "F3", "F4", "UNKNOWN"])

            p_comp = out_dir / "regime_comparison_d4.png"
            plot_regime_comparison(cm_a, cm_b, labels, output_path=p_comp)
            generated["regime_comparison_d4"] = str(p_comp)

    return generated
