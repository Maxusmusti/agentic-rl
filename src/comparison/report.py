"""Generate markdown report and matplotlib plots from comparison results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_report(
    results_dir: str = "results",
    output_dir: str = "results/comparison",
) -> str:
    """Generate a markdown report with plots comparing ART and AGL.

    Args:
        results_dir: Root results directory.
        output_dir: Where to save the report and plots.

    Returns:
        Path to the generated markdown report.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all results
    comparison = _load_json(Path(results_dir) / "comparison" / "comparison.json")
    art_training = _load_json(Path(results_dir) / "art" / "training_results.json")
    agl_training = _load_json(Path(results_dir) / "agl" / "training_results.json")
    baseline = _load_json(Path(results_dir) / "baseline" / "metrics.json")
    art_post_eval = _load_json(Path(results_dir) / "art" / "post_eval" / "metrics.json")
    agl_post_eval = _load_json(Path(results_dir) / "agl" / "post_eval" / "metrics.json")

    # Generate plots
    _generate_plots(art_training, agl_training, baseline, output_path,
                    art_post_eval=art_post_eval, agl_post_eval=agl_post_eval)

    # Generate markdown
    report = _build_markdown(comparison, art_training, agl_training, baseline)

    report_path = output_path / "report.md"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info("Report generated: %s", report_path)
    return str(report_path)


def _build_markdown(
    comparison: dict,
    art_training: dict,
    agl_training: dict,
    baseline: dict,
) -> str:
    """Build the markdown report content."""
    lines = [
        "# Agentic RLVR PoC: ART vs Agent Lightning",
        "",
        "## Summary",
        "",
    ]

    # Baseline
    if baseline:
        lines.extend([
            "### Baseline (Pre-Training)",
            "",
            f"- **pass@1**: {baseline.get('pass_at_1', 'N/A'):.3f}"
            if isinstance(baseline.get('pass_at_1'), (int, float))
            else f"- **pass@1**: {baseline.get('pass_at_1', 'N/A')}",
            f"- **pass@k**: {baseline.get('pass_at_k', 'N/A'):.3f}"
            if isinstance(baseline.get('pass_at_k'), (int, float))
            else f"- **pass@k**: {baseline.get('pass_at_k', 'N/A')}",
            f"- **Mean reward**: {baseline.get('mean_reward', 'N/A')}",
            "",
        ])

    # Comparison table
    lines.extend([
        "### Post-Training Comparison",
        "",
        "| Metric | ART (OpenPipe) | Agent Lightning (Microsoft) |",
        "|--------|---------------|----------------------------|",
    ])

    art_data = comparison.get("art", {})
    agl_data = comparison.get("agl", {})

    metrics = [
        ("pass@1", "post_pass_at_1"),
        ("pass@k", "post_pass_at_k"),
        ("Mean reward", "post_mean_reward"),
        ("pass@1 improvement", "pass_at_1_improvement"),
        ("Training time (s)", "training_time_s"),
        ("Peak VRAM (MB)", "peak_vram_mb"),
        ("Rollouts/second", "rollouts_per_second"),
        ("Avg turns", "avg_turns"),
    ]

    for label, key in metrics:
        art_val = art_data.get(key, "N/A")
        agl_val = agl_data.get(key, "N/A")
        if isinstance(art_val, float):
            art_val = f"{art_val:.3f}"
        if isinstance(agl_val, float):
            agl_val = f"{agl_val:.3f}"
        lines.append(f"| {label} | {art_val} | {agl_val} |")

    lines.extend([
        "",
        "### Training Details",
        "",
        "#### ART (OpenPipe)",
        "",
    ])

    if art_training:
        lines.extend([
            f"- **Iterations**: {art_training.get('num_iterations', 'N/A')}",
            f"- **Group size**: {art_training.get('group_size', 'N/A')}",
            f"- **Learning rate**: {art_training.get('learning_rate', 'N/A')}",
            f"- **Total rollouts**: {art_training.get('total_rollouts', 'N/A')}",
            f"- **Final mean reward**: {art_training.get('final_mean_reward', 'N/A')}",
        ])

    lines.extend(["", "#### Agent Lightning (Microsoft)", ""])

    if agl_training:
        lines.extend([
            f"- **Total epochs**: {agl_training.get('total_epochs', 'N/A')}",
            f"- **Batch size**: {agl_training.get('train_batch_size', 'N/A')}",
            f"- **Learning rate**: {agl_training.get('learning_rate', 'N/A')}",
            f"- **N runners**: {agl_training.get('n_runners', 'N/A')}",
        ])

    lines.extend([
        "",
        "### Plots",
        "",
        "![Reward Curves](reward_curves.png)",
        "",
        "![Pass@1 Comparison](pass_at_1_comparison.png)",
        "",
        "![Training Efficiency](training_efficiency.png)",
        "",
    ])

    return "\n".join(lines)


def _generate_plots(
    art_training: dict,
    agl_training: dict,
    baseline: dict,
    output_path: Path,
    art_post_eval: dict | None = None,
    agl_post_eval: dict | None = None,
):
    """Generate matplotlib comparison plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plots")
        return

    # 1. Reward curves
    fig, ax = plt.subplots(figsize=(10, 6))

    art_rewards = art_training.get("reward_history", [])
    if art_rewards:
        ax.plot(range(1, len(art_rewards) + 1), art_rewards, label="ART", color="blue")

    # AGL may not have per-iteration rewards, but plot if available
    agl_rewards = agl_training.get("reward_history", [])
    if agl_rewards:
        ax.plot(range(1, len(agl_rewards) + 1), agl_rewards, label="Agent Lightning", color="red")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Training Reward Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "reward_curves.png", dpi=150)
    plt.close(fig)

    # 2. Pass@1 comparison bar chart — use post-eval if available
    fig, ax = plt.subplots(figsize=(8, 5))

    baseline_p1 = baseline.get("pass_at_1", 0)
    art_p1 = (art_post_eval or {}).get("pass_at_1", art_training.get("final_mean_reward", 0))
    agl_p1 = (agl_post_eval or {}).get("pass_at_1", agl_training.get("final_mean_reward", 0))

    bars = ax.bar(
        ["Baseline", "ART", "Agent Lightning"],
        [baseline_p1, art_p1, agl_p1],
        color=["gray", "blue", "red"],
        alpha=0.7,
    )
    ax.set_ylabel("pass@1")
    ax.set_title("pass@1 Comparison")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, [baseline_p1, art_p1, agl_p1]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
        )
    fig.tight_layout()
    fig.savefig(output_path / "pass_at_1_comparison.png", dpi=150)
    plt.close(fig)

    # 3. Training efficiency scatter
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, data, color in [
        ("ART", art_training, "blue"),
        ("Agent Lightning", agl_training, "red"),
    ]:
        time_s = data.get("total_time_seconds", 0)
        vram = data.get("peak_vram_mb", 0)
        if time_s > 0 or vram > 0:
            ax.scatter(time_s, vram, label=name, color=color, s=100, zorder=5)
            ax.annotate(name, (time_s, vram), textcoords="offset points", xytext=(10, 10))

    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel("Peak VRAM (MB)")
    ax.set_title("Training Efficiency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path / "training_efficiency.png", dpi=150)
    plt.close(fig)


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate comparison report")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output-dir", default="results/comparison", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    report_path = generate_report(args.results_dir, args.output_dir)
    print(f"Report generated: {report_path}")


if __name__ == "__main__":
    main()
