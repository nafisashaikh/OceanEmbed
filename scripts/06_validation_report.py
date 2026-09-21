"""Generate the OceanEmbed ARGO validation report and plots."""

from __future__ import annotations

from pathlib import Path
import sys
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.validation.argo_validation import validate


def _taylor_plot(records: pd.DataFrame, depths: list[float], output: Path) -> None:
    figure = plt.figure(figsize=(7, 6))
    axis = figure.add_subplot(111, polar=True)
    axis.set_theta_zero_location("E")
    axis.set_thetamin(0)
    axis.set_thetamax(90)
    for depth in depths:
        subset = records[records["depth"] == depth]
        if len(subset) < 2 or subset["observed"].std() == 0 or subset["model"].std() == 0:
            continue
        correlation = subset["observed"].corr(subset["model"])
        ratio = subset["model"].std() / subset["observed"].std()
        axis.scatter(np.arccos(np.clip(correlation, -1, 1)), ratio, label=f"{depth:g} m")
    axis.set_title("Taylor diagram: model vs real ARGO")
    axis.set_xlabel("angle = arccos(correlation); radius = std(model) / std(ARGO)")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, fontsize="small", bbox_to_anchor=(1.35, 1.05))
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main(config_path: str | Path = "config.yaml") -> int:
    config = load_config(config_path)
    summary = validate(config_path)
    output_dir = Path(config["paths"]["sanity_output"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    depths = [float(depth) for depth in config["depths"]["depths_m"]]
    overall = summary["overall"]
    table = pd.DataFrame([
        {
            "depth": depth,
            **overall[str(depth)],
            "glorys_RMSE": summary["glorys_ceiling"][str(depth)]["RMSE"],
            "climatology_RMSE": summary["climatology"][str(depth)]["RMSE"],
            "confidence": "low" if summary["overall"][str(depth)]["n_matches"] < 30 else "adequate",
            "model_beats_climatology_p<0.05": summary["paired_ttest_by_depth"][str(depth)]["model_significantly_better"],
        }
        for depth in depths
    ])
    table.to_csv(output_dir / "argo_validation_summary.csv", index=False)
    print(table.to_string(index=False))
    print("\nPaired t-test:", summary["paired_ttest"])
    print("Platform audit:", summary["platform_audit"])
    print("Per-depth paired tests:", summary["paired_ttest_by_depth"])
    records = pd.DataFrame(summary["matched_records"])
    _taylor_plot(records, depths, output_dir / "argo_taylor_diagram.png")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(table["depth"], table["RMSE"], marker="o", label="model")
    axis.set(xlabel="depth (m)", ylabel="RMSE (C)", title="ARGO validation RMSE by depth")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "argo_rmse_by_depth.png", dpi=150)
    plt.close(figure)
    print(f"\nPlots saved under {output_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    raise SystemExit(main(parser.parse_args().config))