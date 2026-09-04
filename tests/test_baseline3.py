from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from baseline3.data import HEEWConditionDataset
from baseline3.metrics import crps_map, summarize
from baseline3.models import build_models, gradient_penalty
from baseline3.sweep import EXPECTED_CONFIGURATIONS, load_search_space
from baseline3.train import train


def tiny_config() -> dict:
    return {
        "data": {"seq_len": 24},
        "model": {"latent_dim": 16, "hidden_dim": 16, "num_blocks": 1},
    }


def test_joint_model_shapes_and_gradient_penalty() -> None:
    config = tiny_config()
    generator, critic = build_models(config, condition_channels=18)
    noise = torch.randn(4, 16)
    conditions = torch.randn(4, 18, 24)
    pv_year = torch.randn(4, 1, 24)
    real = torch.randn(4, 4, 24)
    fake = generator(noise, conditions, pv_year)
    assert fake.shape == (4, 4, 24)
    assert critic(fake, conditions, pv_year).shape == (4,)
    penalty = gradient_penalty(critic, real, fake.detach(), conditions, pv_year)
    assert penalty.ndim == 0
    assert torch.isfinite(penalty)
    penalty.backward()


def test_crps_and_metric_contract() -> None:
    random = np.random.default_rng(42)
    target = random.normal(size=(10, 4, 24)).astype(np.float32)
    samples = target[:, None] + random.normal(
        scale=0.2, size=(10, 20, 4, 24)
    ).astype(np.float32)
    scores = crps_map(samples, target)
    assert scores.shape == target.shape
    assert np.all(scores >= -1e-10)
    result = summarize(
        samples,
        target,
        ["Electricity", "Heat", "Cooling", "PV"],
        np.zeros(4),
        np.ones(4),
    )
    for label in ("Electricity", "Heat", "Cooling", "PV"):
        for metric in (
            "RMSE",
            "MAE",
            "RMSE_Z",
            "MAE_Z",
            "CRPS",
            "nCRPS",
            "Coverage90",
            "IntervalWidth90",
            "Precision_Z",
            "Recall_Z",
            "CR",
            "IW",
        ):
            assert f"{label}_{metric}" in result
    assert "macro_nCRPS" in result


def test_search_budget_is_exactly_six() -> None:
    configurations = load_search_space("baseline3/configs/search_space.yaml")
    assert len(configurations) == EXPECTED_CONFIGURATIONS == 6
    assert len({item["id"] for item in configurations}) == 6


def test_dataset_and_one_epoch_training_smoke(tmp_path) -> None:
    timestamps = pd.DatetimeIndex(
        [
            pd.Timestamp(year=year, month=1, day=1, hour=hour)
            for year in range(2014, 2023)
            for hour in range(24)
        ]
    )
    timestamp_columns = {
        "Year": timestamps.year,
        "Month": timestamps.month,
        "Day": timestamps.day,
        "Hour": timestamps.hour,
    }
    hour = timestamps.hour.to_numpy(dtype=np.float32)
    year = timestamps.year.to_numpy(dtype=np.float32)
    energy = pd.DataFrame(
        {
            **timestamp_columns,
            "Electricity": 100 + hour + 0.1 * year,
            "Heat": 60 + np.cos(hour / 24 * 2 * np.pi) * 10,
            "Cooling": 30 + np.sin(hour / 24 * 2 * np.pi) * 5,
            "PV": np.maximum(0, np.sin((hour - 6) / 12 * np.pi)) * 20,
        }
    )
    weather = pd.DataFrame(
        {
            **timestamp_columns,
            "Temperature": 15 + np.sin(hour / 24 * 2 * np.pi),
            "Dew Point": 8 + 0.1 * hour,
            "Humidity": 50 + hour,
            "Wind Speed": 2 + 0.01 * hour,
            "Pressure": 1000 + hour,
            "Precip": np.zeros_like(hour),
            "ALLSKY_SFC_SW_DWN": np.maximum(0, np.sin((hour - 6) / 12 * np.pi)),
            "CLRSKY_SFC_SW_DWN": np.maximum(0, np.sin((hour - 6) / 12 * np.pi)),
            "PV_CLEARNESS_RATIO": np.ones_like(hour),
            "PV_IS_DAYLIGHT": ((hour >= 6) & (hour <= 18)).astype(np.float32),
        }
    )
    energy_path = tmp_path / "energy.csv"
    weather_path = tmp_path / "weather.csv"
    energy.to_csv(energy_path, index=False)
    weather.to_csv(weather_path, index=False)
    dataset = HEEWConditionDataset(energy_path, weather_path, "test")
    assert dataset[0][0].shape == (18, 24)
    assert dataset[0][2].shape == (4, 24)

    config = {
        "run": {
            "seed": 42,
            "device": "cpu",
            "output_root": str(tmp_path / "experiments"),
        },
        "data": {
            "energy_path": str(energy_path),
            "weather_path": str(weather_path),
            "weather_feature_set": "pv10",
            "seq_len": 24,
            "num_workers": 0,
        },
        "model": {"latent_dim": 8, "hidden_dim": 8, "num_blocks": 1},
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "lr_g": 0.0001,
            "lr_d": 0.0001,
            "betas": [0.0, 0.9],
            "critic_steps": 1,
            "gp_lambda": 10.0,
            "val_every": 1,
            "early_stopping_patience": 1,
        },
        "evaluation": {"batch_size": 2, "val_scenarios": 3},
    }
    checkpoint = train(config, "smoke")
    assert checkpoint.exists()
