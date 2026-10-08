"""Tiny actual GAN checkpoints: unchanged samples and baseline5 output schema."""
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from baseline3 import evaluate, score_npz
from baseline3.data import HEEWConditionDataset, BASE_WEATHER_COLUMNS, PV_WEATHER_COLUMNS
from baseline3.models import build_models, checkpoint_payload
from baseline3.runtime import seed_everything
from baseline3.train import generate_scenarios


def make_csvs(tmp_path):
    stamps = pd.DatetimeIndex([pd.Timestamp(year=y, month=1, day=d, hour=h)
                              for y in range(2014, 2023) for d in (1, 2) for h in range(24)])
    base = {"Year": stamps.year, "Month": stamps.month, "Day": stamps.day, "Hour": stamps.hour}
    time = np.arange(len(stamps), dtype=np.float32)
    energy = pd.DataFrame({**base, "Electricity": 100 + time, "Heat": 10 + time / 10,
                           "Cooling": 50 + time / 2,
                           "PV": np.maximum(0, np.sin((stamps.hour - 6) * np.pi / 12)) * 20})
    weather = pd.DataFrame({**base, **{col: time * (i + 1) / 100
                          for i, col in enumerate(BASE_WEATHER_COLUMNS + PV_WEATHER_COLUMNS)}})
    ep, wp = tmp_path / "energy.csv", tmp_path / "weather.csv"
    energy.to_csv(ep, index=False)
    weather.to_csv(wp, index=False)
    return ep, wp


@pytest.mark.parametrize("seed", [42, 3407])
def test_legacy_sampling_and_rescore_parity(tmp_path, monkeypatch, seed):
    ep, wp = make_csvs(tmp_path)
    config = {"run": {"seed": seed}, "data": {"seq_len": 24, "energy_path": str(ep),
              "weather_path": str(wp), "weather_feature_set": "pv10"},
              "model": {"latent_dim": 8, "hidden_dim": 8, "num_blocks": 1}}
    dataset = HEEWConditionDataset(ep, wp, "test")
    seed_everything(seed)
    generator, critic = build_models(config, dataset.condition_channels)
    generator.eval()
    checkpoint = tmp_path / "best.pt"
    torch.save(checkpoint_payload(config, generator, critic, 3, {}), checkpoint)
    noise_generator = torch.Generator(device="cpu").manual_seed(seed + 90_000)
    expected = []
    with torch.no_grad():
        for conditions, year, target, dates in DataLoader(dataset, batch_size=1, shuffle=False):
            x = generate_scenarios(generator, conditions, year, 3, 8, noise_generator)
            x = dataset.denormalize(x)
            x[:, :, 3].clamp_(min=0)
            expected.append(x.numpy())
    out = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", str(checkpoint), "--outdir", str(out),
                                     "--scenarios", "3", "--batch-size", "1", "--device", "cpu", "--no-plots"])
    evaluate.main()
    with np.load(out / "baseline3_scenarios.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["scenarios"], np.concatenate(expected))
        np.testing.assert_array_equal(saved["target_std"], dataset.target_std)
        assert saved["conditions"].shape == (2, 18, 24)
        assert saved["sample_seed"].item() == seed + 90_000
    result = json.loads((out / "global_metrics.json").read_text())
    assert result["sampling_steps"] is None and result["training_timesteps"] is None
    assert result["generator_forward_passes_per_scenario"] == 1
    assert result["VS_pair_count"] == 3456 and result["seed"] == seed
    rescored = tmp_path / "rescored"
    monkeypatch.setattr(sys, "argv", ["score_npz", "--npz", str(out / "baseline3_scenarios.npz"),
                                     "--outdir", str(rescored), "--no-plots"])
    score_npz.main()
    offline = json.loads((rescored / "global_metrics.json").read_text())
    for key in ("ES", "VS", "macro_nCRPS", "PV_R2", "PV_IS", "PV_RMSE_Z", "sample_seed"):
        assert offline[key] == result[key]


def test_legacy_sidecar_seed_and_explicit_override(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(2, 3, 4, 24)), rng.normal(size=(2, 4, 24))
    source = tmp_path / "baseline3_scenarios.npz"
    np.savez(source, scenarios=x, targets=y, dates=["2022-01-01", "2022-01-02"],
             target_mean=np.zeros(4), target_std=np.ones(4))
    out = tmp_path / "rescored"
    command = ["score_npz", "--npz", str(source), "--outdir", str(out), "--no-plots"]
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(SystemExit):
        score_npz.main()
    monkeypatch.setattr(sys, "argv", ["score_npz", "--npz", str(source), "--outdir", str(tmp_path / "explicit"),
                                     "--seed", "123", "--no-plots"])
    score_npz.main()
    explicit = json.loads((tmp_path / "explicit" / "global_metrics.json").read_text())
    assert explicit["seed"] == 123 and explicit["sample_seed"] is None
    monkeypatch.setattr(sys, "argv", command)
    sidecar = tmp_path / "global_metrics.json"
    sidecar.write_text(json.dumps({"seed": 3407, "sampler": "one_generator_forward_pass",
                                  "num_test_days": 2, "scenarios_per_day": 3}))
    score_npz.main()
    result = json.loads((out / "global_metrics.json").read_text())
    assert result["seed"] == 3407 and result["sample_seed"] == 93407
    monkeypatch.setattr(sys, "argv", ["score_npz", "--npz", str(source), "--outdir", str(tmp_path / "override"),
                                     "--seed", "42", "--no-plots"])
    score_npz.main()
    result = json.loads((tmp_path / "override" / "global_metrics.json").read_text())
    assert result["seed"] == 42 and result["sample_seed"] == 93407
    assert result["original_evaluation_seed"] == 3407


@pytest.mark.parametrize("args", [["--scenarios", "0"], ["--batch-size", "0"], ["--max-days", "0"]])
def test_invalid_counts_fail_early(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", "unused.pt", *args])
    with pytest.raises(ValueError, match="positive"):
        evaluate.main()
