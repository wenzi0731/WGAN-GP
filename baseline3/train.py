from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader

from baseline3.data import HEEWConditionDataset
from baseline3.metrics import crps_map
from baseline3.models import build_models, checkpoint_payload, gradient_penalty
from baseline3.runtime import (
    choose_device,
    load_config,
    resolve_path,
    seed_everything,
    seed_worker,
)


def build_dataloaders(
    config: dict[str, Any], seed: int
) -> tuple[DataLoader, DataLoader, HEEWConditionDataset, HEEWConditionDataset]:
    data = config["data"]
    common = {
        "energy_path": resolve_path(data["energy_path"]),
        "weather_path": resolve_path(data["weather_path"]),
        "weather_feature_set": data.get("weather_feature_set", "pv10"),
    }
    train_dataset = HEEWConditionDataset(split="train", **common)
    val_dataset = HEEWConditionDataset(split="val", **common)
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(data.get("num_workers", 0)),
        pin_memory=bool(data.get("pin_memory", False)),
        worker_init_fn=seed_worker,
        generator=loader_generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["evaluation"].get("batch_size", 64)),
        shuffle=False,
        num_workers=int(data.get("num_workers", 0)),
        pin_memory=bool(data.get("pin_memory", False)),
        worker_init_fn=seed_worker,
    )
    if train_dataset.condition_channels != val_dataset.condition_channels:
        raise RuntimeError("Train/validation condition dimensions do not match.")
    return train_loader, val_loader, train_dataset, val_dataset


@torch.no_grad()
def generate_scenarios(
    generator: torch.nn.Module,
    conditions: torch.Tensor,
    pv_year: torch.Tensor,
    scenario_count: int,
    latent_dim: int,
    noise_generator: torch.Generator | None = None,
) -> torch.Tensor:
    batch_size = conditions.shape[0]
    expanded_conditions = conditions[:, None].expand(
        batch_size, scenario_count, *conditions.shape[1:]
    )
    expanded_pv_year = pv_year[:, None].expand(
        batch_size, scenario_count, *pv_year.shape[1:]
    )
    noise = torch.randn(
        batch_size * scenario_count,
        latent_dim,
        device=conditions.device,
        generator=noise_generator,
    )
    generated = generator(
        noise,
        expanded_conditions.reshape(-1, *conditions.shape[1:]),
        expanded_pv_year.reshape(-1, *pv_year.shape[1:]),
    )
    return generated.reshape(batch_size, scenario_count, *generated.shape[1:])


@torch.no_grad()
def validate(
    generator: torch.nn.Module,
    loader: DataLoader,
    dataset: HEEWConditionDataset,
    device: torch.device,
    scenario_count: int,
    latent_dim: int,
    seed: int,
) -> dict[str, float]:
    generator.eval()
    noise_generator = torch.Generator(device=device.type).manual_seed(seed)
    generated_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    for conditions, pv_year, target, _ in loader:
        conditions = conditions.to(device)
        pv_year = pv_year.to(device)
        generated_z = generate_scenarios(
            generator,
            conditions,
            pv_year,
            scenario_count,
            latent_dim,
            noise_generator,
        )
        generated = dataset.denormalize(generated_z)
        target_physical = dataset.denormalize(target.to(device))
        generated[:, :, 3].clamp_(min=0.0)
        generated_batches.append(generated.cpu().numpy())
        target_batches.append(target_physical.cpu().numpy())

    generated_values = np.concatenate(generated_batches)
    target_values = np.concatenate(target_batches)
    scores = crps_map(generated_values, target_values)
    channel_ncrps = [
        float(
            np.sum(scores[:, channel])
            / (np.sum(np.abs(target_values[:, channel])) + 1e-8)
        )
        for channel in range(target_values.shape[1])
    ]
    lower = np.quantile(generated_values, 0.05, axis=1)
    upper = np.quantile(generated_values, 0.95, axis=1)
    coverage = float(np.mean((target_values >= lower) & (target_values <= upper)))
    return {
        "val_macro_nCRPS": float(np.mean(channel_ncrps)),
        "val_mean_nCRPS": float(
            np.sum(scores) / (np.sum(np.abs(target_values)) + 1e-8)
        ),
        "val_Coverage90": coverage,
        "val_Coverage90_error": abs(coverage - 0.90),
    }


def train(config: dict[str, Any], run_name: str | None = None) -> Path:
    seed = int(config["run"]["seed"])
    seed_everything(seed)
    device = choose_device(str(config["run"].get("device", "auto")))
    run_label = run_name or str(config["run"].get("name", "cwgangp_joint"))
    run_dir = resolve_path(config["run"]["output_root"]) / f"{run_label}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(
        config, seed
    )
    generator, critic = build_models(config, train_dataset.condition_channels)
    generator.to(device)
    critic.to(device)

    training = config["training"]
    beta1, beta2 = (float(value) for value in training.get("betas", [0.0, 0.9]))
    generator_optimizer = Adam(
        generator.parameters(), lr=float(training["lr_g"]), betas=(beta1, beta2)
    )
    critic_optimizer = Adam(
        critic.parameters(), lr=float(training["lr_d"]), betas=(beta1, beta2)
    )
    epochs = int(training["epochs"])
    critic_steps = int(training["critic_steps"])
    gp_weight = float(training["gp_lambda"])
    val_every = int(training.get("val_every", 5))
    patience = int(training.get("early_stopping_patience", 12))
    gradient_clip = float(training.get("gradient_clip", 0.0))
    latent_dim = int(config["model"]["latent_dim"])
    val_scenarios = int(config["evaluation"].get("val_scenarios", 20))
    history: list[dict[str, float | int]] = []
    best_score = float("inf")
    validations_without_improvement = 0
    best_path = run_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        generator.train()
        critic.train()
        critic_losses: list[float] = []
        generator_losses: list[float] = []
        penalties: list[float] = []
        for conditions, pv_year, real, _ in train_loader:
            conditions = conditions.to(device, non_blocking=True)
            pv_year = pv_year.to(device, non_blocking=True)
            real = real.to(device, non_blocking=True)
            batch_size = real.shape[0]

            for _ in range(critic_steps):
                critic_optimizer.zero_grad(set_to_none=True)
                noise = torch.randn(batch_size, latent_dim, device=device)
                with torch.no_grad():
                    fake = generator(noise, conditions, pv_year)
                real_score = critic(real, conditions, pv_year).mean()
                fake_score = critic(fake, conditions, pv_year).mean()
                penalty = gradient_penalty(
                    critic, real, fake, conditions, pv_year
                )
                critic_loss = fake_score - real_score + gp_weight * penalty
                critic_loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), gradient_clip)
                critic_optimizer.step()
                critic_losses.append(float(critic_loss.detach()))
                penalties.append(float(penalty.detach()))

            generator_optimizer.zero_grad(set_to_none=True)
            for parameter in critic.parameters():
                parameter.requires_grad_(False)
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake = generator(noise, conditions, pv_year)
            generator_loss = -critic(fake, conditions, pv_year).mean()
            generator_loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), gradient_clip)
            generator_optimizer.step()
            for parameter in critic.parameters():
                parameter.requires_grad_(True)
            generator_losses.append(float(generator_loss.detach()))

        record: dict[str, float | int] = {
            "epoch": epoch,
            "critic_loss": float(np.mean(critic_losses)),
            "generator_loss": float(np.mean(generator_losses)),
            "gradient_penalty": float(np.mean(penalties)),
        }
        if epoch % val_every == 0 or epoch == epochs:
            validation = validate(
                generator,
                val_loader,
                val_dataset,
                device,
                val_scenarios,
                latent_dim,
                seed + 50_000,
            )
            record.update(validation)
            score = validation["val_macro_nCRPS"]
            if score < best_score:
                best_score = score
                validations_without_improvement = 0
                torch.save(
                    checkpoint_payload(
                        config, generator, critic, epoch, validation
                    ),
                    best_path,
                )
                with (run_dir / "best_metrics.json").open(
                    "w", encoding="utf-8"
                ) as handle:
                    json.dump(
                        {"epoch": epoch, **validation}, handle, indent=2
                    )
            else:
                validations_without_improvement += 1
        history.append(record)
        with (run_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        print(json.dumps(record, ensure_ascii=False))
        if validations_without_improvement >= patience:
            print(f"Early stopping after epoch {epoch}.")
            break

    if not best_path.exists():
        raise RuntimeError("Training finished without a validation checkpoint.")
    return best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train conditional joint WGAN-GP.")
    parser.add_argument("--config", default="baseline3/configs/heew.yaml")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    checkpoint = train(config, args.run_name)
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
