"""Side-by-side comparison of ART and Agent Lightning training results."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FrameworkResult:
    """Results from a single framework's training + evaluation."""

    name: str
    training: dict[str, Any] = field(default_factory=dict)
    baseline_eval: dict[str, Any] = field(default_factory=dict)
    post_eval: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Side-by-side comparison of two frameworks."""

    art: FrameworkResult
    agl: FrameworkResult
    summary: dict[str, Any] = field(default_factory=dict)


def load_results(results_dir: str = "results") -> ComparisonResult:
    """Load results from all framework directories.

    Expected structure:
        results/
        ├── baseline/metrics.json
        ├── art/training_results.json
        ├── art/post_eval/metrics.json
        ├── agl/training_results.json
        └── agl/post_eval/metrics.json
    """
    base = Path(results_dir)

    baseline = _load_json(base / "baseline" / "metrics.json")

    art = FrameworkResult(
        name="ART (OpenPipe)",
        training=_load_json(base / "art" / "training_results.json"),
        baseline_eval=baseline,
        post_eval=_load_json(base / "art" / "post_eval" / "metrics.json"),
    )

    agl = FrameworkResult(
        name="Agent Lightning (Microsoft)",
        training=_load_json(base / "agl" / "training_results.json"),
        baseline_eval=baseline,
        post_eval=_load_json(base / "agl" / "post_eval" / "metrics.json"),
    )

    comparison = ComparisonResult(art=art, agl=agl)
    comparison.summary = _compute_summary(comparison)

    return comparison


def _compute_summary(comp: ComparisonResult) -> dict[str, Any]:
    """Compute summary comparison metrics."""
    summary: dict[str, Any] = {}

    # Baseline metrics (shared)
    baseline = comp.art.baseline_eval
    if baseline:
        summary["baseline"] = {
            "pass_at_1": baseline.get("pass_at_1", 0),
            "pass_at_k": baseline.get("pass_at_k", 0),
            "mean_reward": baseline.get("mean_reward", 0),
        }

    for fw_name, fw in [("art", comp.art), ("agl", comp.agl)]:
        fw_summary: dict[str, Any] = {"name": fw.name}

        # Training metrics
        if fw.training:
            fw_summary["training_time_s"] = fw.training.get("total_time_seconds", 0)
            fw_summary["peak_vram_mb"] = fw.training.get("peak_vram_mb", 0)
            fw_summary["total_rollouts"] = fw.training.get("total_rollouts", 0)

            # Reward curve (final value)
            rewards = fw.training.get("reward_history", [])
            fw_summary["final_train_reward"] = rewards[-1] if rewards else 0

        # Post-training eval
        if fw.post_eval:
            fw_summary["post_pass_at_1"] = fw.post_eval.get("pass_at_1", 0)
            fw_summary["post_pass_at_k"] = fw.post_eval.get("pass_at_k", 0)
            fw_summary["post_mean_reward"] = fw.post_eval.get("mean_reward", 0)
            fw_summary["avg_turns"] = fw.post_eval.get("avg_turns", 0)

        # Improvement over baseline
        if baseline and fw.post_eval:
            base_p1 = baseline.get("pass_at_1", 0)
            post_p1 = fw.post_eval.get("pass_at_1", 0)
            fw_summary["pass_at_1_improvement"] = post_p1 - base_p1

        summary[fw_name] = fw_summary

    # Rollout throughput comparison
    for fw_name, fw in [("art", comp.art), ("agl", comp.agl)]:
        if fw.training:
            total_rollouts = fw.training.get("total_rollouts", 0)
            total_time = fw.training.get("total_time_seconds", 1)
            summary[fw_name]["rollouts_per_second"] = total_rollouts / total_time

    return summary


def save_comparison(comp: ComparisonResult, output_dir: str = "results/comparison"):
    """Save comparison results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "comparison.json", "w") as f:
        json.dump(comp.summary, f, indent=2)

    logger.info("Comparison saved to %s", output_path / "comparison.json")


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file, returning empty dict if not found."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    logger.warning("Results file not found: %s", path)
    return {}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare ART vs AGL results")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output-dir", default="results/comparison", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    comp = load_results(args.results_dir)
    save_comparison(comp, args.output_dir)

    # Print summary table
    print("\n" + "=" * 60)
    print("COMPARISON: ART vs Agent Lightning")
    print("=" * 60)
    for key, val in comp.summary.items():
        if isinstance(val, dict):
            print(f"\n{key}:")
            for k, v in val.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {val}")


if __name__ == "__main__":
    main()
