from __future__ import annotations

import numpy as np
import torch

from baseline3.metrics import crps_map, summarize
from baseline3.models import build_models, gradient_penalty
from baseline3.sweep import EXPECTED_CONFIGURATIONS, load_search_space


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
