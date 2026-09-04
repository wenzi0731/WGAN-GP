from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from baseline3.runtime import load_config, resolve_path


EXPECTED_CONFIGURATIONS = 6


def load_search_space(path: str | Path) -> list[dict[str, Any]]:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    configurations = payload.get("configurations") if isinstance(payload, dict) else None
    if not isinstance(configurations, list):
        raise TypeError("Search space must contain a configurations list.")
    if len(configurations) != EXPECTED_CONFIGURATIONS:
        raise ValueError(
            f"The fairness budget is exactly {EXPECTED_CONFIGURATIONS} configurations; "
            f"found {len(configurations)}."
        )
    identifiers = [str(item.get("id", "")) for item in configurations]
    if any(not identifier for identifier in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError("Every search configuration needs a unique non-empty id.")
    for item in configurations:
        if not isinstance(item.get("overrides"), dict):
            raise TypeError(f"Configuration {item['id']} requires an overrides mapping.")
    return configurations


def flatten_overrides(values: dict[str, Any], prefix: str = "") -> list[str]:
    flattened: list[str] = []
    for key, value in values.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.extend(flatten_overrides(value, dotted))
        else:
            flattened.append(f"{dotted}={json.dumps(value)}")
    return flattened


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed six-configuration budget.")
    parser.add_argument("--config", default="baseline3/configs/heew.yaml")
    parser.add_argument(
        "--search-space", default="baseline3/configs/search_space.yaml"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="experiments/baseline3/tuning_budget_6")
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configurations = load_search_space(args.search_space)
    base_config = load_config(args.config)
    output_root = resolve_path(base_config["run"]["output_root"])
    summary_dir = resolve_path(args.output_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for specification in configurations:
        identifier = str(specification["id"])
        run_name = f"cwgangp_tune_{identifier}"
        run_dir = output_root / f"{run_name}_seed{args.seed}"
        metrics_path = run_dir / "best_metrics.json"
        command = [
            sys.executable,
            "-m",
            "baseline3.train",
            "--config",
            str(resolve_path(args.config)),
            "--run-name",
            run_name,
            "--set",
            f"run.seed={args.seed}",
        ]
        for override in flatten_overrides(specification["overrides"]):
            command.extend(["--set", override])
        if not (args.skip_completed and metrics_path.exists()):
            subprocess.run(command, cwd=resolve_path("."), check=True)
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append(
            {
                "config_id": identifier,
                "description": specification.get("description", ""),
                "overrides": json.dumps(specification["overrides"], sort_keys=True),
                **metrics,
                "run_dir": str(run_dir),
                "checkpoint": str(run_dir / "best.pt"),
            }
        )

    rows.sort(key=lambda row: (float(row["val_macro_nCRPS"]), str(row["config_id"])))
    best = rows[0]
    with (summary_dir / "tuning_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (summary_dir / "best_config.json").open("w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2)
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
