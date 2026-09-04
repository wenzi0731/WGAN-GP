from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline3.data import HEEWConditionDataset
from baseline3.metrics import (
    save_global_pearson_comparison,
    save_global_pearson_plot,
    save_random_timeseries_plots,
    summarize,
)
from baseline3.models import build_models
from baseline3.runtime import choose_device, load_checkpoint, resolve_path, seed_everything
from baseline3.train import generate_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate conditional joint WGAN-GP.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--energy-path", default=None)
    parser.add_argument("--weather-path", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-days", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    seed = int(args.seed if args.seed is not None else config["run"]["seed"])
    seed_everything(seed)
    device = choose_device(args.device)
    data_config = config["data"]
    dataset = HEEWConditionDataset(
        energy_path=resolve_path(args.energy_path or data_config["energy_path"]),
        weather_path=resolve_path(args.weather_path or data_config["weather_path"]),
        split="test",
        weather_feature_set=data_config.get("weather_feature_set", "pv10"),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    generator, _ = build_models(config, dataset.condition_channels)
    generator.load_state_dict(checkpoint["generator"])
    generator.to(device).eval()
    latent_dim = int(config["model"]["latent_dim"])
    output_dir = (
        resolve_path(args.outdir)
        if args.outdir
        else checkpoint_path.parent / "evaluation_test"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    condition_batches: list[np.ndarray] = []
    dates: list[str] = []
    noise_generator = torch.Generator(device=device.type).manual_seed(seed + 90_000)
    elapsed = 0.0
    processed = 0
    with torch.no_grad():
        for conditions, pv_year, target, batch_dates in loader:
            if args.max_days is not None and processed >= args.max_days:
                break
            take = len(batch_dates)
            if args.max_days is not None:
                take = min(take, args.max_days - processed)
            conditions = conditions[:take].to(device)
            pv_year = pv_year[:take].to(device)
            target = target[:take].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            generated_z = generate_scenarios(
                generator,
                conditions,
                pv_year,
                args.scenarios,
                latent_dim,
                noise_generator,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - start
            generated = dataset.denormalize(generated_z)
            target_physical = dataset.denormalize(target)
            generated[:, :, 3].clamp_(min=0.0)
            generated_batches.append(generated.cpu().numpy())
            target_batches.append(target_physical.cpu().numpy())
            condition_batches.append(conditions.cpu().numpy())
            dates.extend(list(batch_dates[:take]))
            processed += take

    generated_values = np.concatenate(generated_batches)
    target_values = np.concatenate(target_batches)
    condition_values = np.concatenate(condition_batches)
    labels = list(dataset.target_cols)
    metrics = summarize(
        generated_values,
        target_values,
        labels,
        dataset.target_mean,
        dataset.target_std,
        seed=seed,
    )
    metrics.update(
        save_global_pearson_comparison(
            target_values, generated_values, output_dir / "pearson", labels
        )
    )
    metrics["sampling_seconds"] = elapsed
    metrics["seconds_per_1000_scenarios"] = (
        elapsed * 1000.0 / (len(target_values) * args.scenarios)
    )
    metrics["num_test_days"] = int(len(target_values))
    metrics["scenarios_per_day"] = int(args.scenarios)
    metrics["seed"] = seed
    metrics["sampler"] = "one_generator_forward_pass"

    save_global_pearson_plot(
        target_values, output_dir / "global_pearson.png", labels
    )
    save_random_timeseries_plots(
        target_values,
        generated_values,
        dates,
        output_dir / "random_timeseries_50",
        labels,
        seed=seed,
        n_plots=50,
    )
    np.savez_compressed(
        output_dir / "baseline3_scenarios.npz",
        scenarios=generated_values,
        targets=target_values,
        conditions=condition_values,
        dates=np.asarray(dates),
        target_mean=dataset.target_mean,
        target_std=dataset.target_std,
    )
    with (output_dir / "global_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_dir / "global_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
