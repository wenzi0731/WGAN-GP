from __future__ import annotations

import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline3.data import HEEWConditionDataset
from baseline3.reporting import require_empty_output, write_report
from baseline3.models import build_models
from baseline3.runtime import choose_device, load_checkpoint, resolve_path, seed_everything
from baseline3.train import generate_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate conditional joint WGAN-GP.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--energy-path", default=None)
    parser.add_argument("--weather-path", default=None)
    parser.add_argument("--outdir", "--output-dir", default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scenarios", "--num-scenarios", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-days", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scenarios < 1 or args.batch_size < 1 or (args.max_days is not None and args.max_days < 1):
        raise ValueError("Scenarios, batch size and max-days must be positive")
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
    require_empty_output(output_dir)

    generated_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    condition_batches: list[np.ndarray] = []
    year_batches: list[np.ndarray] = []
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
            if not torch.isfinite(generated_z).all():
                raise FloatingPointError("Nonfinite generated scenarios before denormalization")
            generated = dataset.denormalize(generated_z)
            target_physical = dataset.denormalize(target)
            generated[:, :, 3].clamp_(min=0.0)
            generated_batches.append(generated.cpu().numpy())
            target_batches.append(target_physical.cpu().numpy())
            condition_batches.append(conditions.cpu().numpy())
            year_batches.append(pv_year.cpu().numpy())
            dates.extend(list(batch_dates[:take]))
            processed += take

    generated_values = np.concatenate(generated_batches)
    target_values = np.concatenate(target_batches)
    condition_values = np.concatenate(condition_batches)
    labels = list(dataset.target_cols)
    archive = output_dir / "baseline3_scenarios.npz"
    metrics = write_report(
        generated_values, target_values, dataset.target_mean, dataset.target_std,
        dates, labels, output_dir, seed=seed, no_plots=args.no_plots,
        precision_recall_k=int(config.get("evaluation", {}).get("precision_recall_k", 5)),
        max_precision_samples=int(config.get("evaluation", {}).get("max_precision_samples", 10_000)),
        metadata={
            "method": "conditional-wgan-gp-joint-baseline3",
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "split": "test-2022",
            "partial_test": len(target_values) != len(dataset),
            "training_seed": int(config["run"]["seed"]),
            "sample_seed": seed + 90_000,
            "sampler": "one_generator_forward_pass",
            "training_timesteps": None,
            "sampling_steps": None,
            "generator_forward_passes_per_scenario": 1,
            "latent_dim": latent_dim,
            "sampling_batch_size": args.batch_size,
            "sampling_seconds": elapsed,
            "seconds_per_1000_scenarios": elapsed * 1000 / (len(target_values) * args.scenarios),
            "archive": str(archive),
        },
    )
    np.savez_compressed(
        archive,
        scenarios=generated_values,
        targets=target_values,
        conditions=condition_values,
        pv_year=np.concatenate(year_batches),
        dates=np.asarray(dates),
        channel_names=np.asarray(labels),
        target_mean=dataset.target_mean,
        target_std=dataset.target_std,
        seed=seed,
        sample_seed=seed + 90_000,
        sampler="one_generator_forward_pass",
        num_scenarios=args.scenarios,
        metadata_json=json.dumps(metrics, allow_nan=False),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
