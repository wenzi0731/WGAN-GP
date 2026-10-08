# Baseline 3: Conditional Joint WGAN-GP for HEEW

This directory adapts the repository's original WGAN-GP implementation into a
reproducible baseline for **condition-only joint source-load scenario
generation**. The original examples remain unchanged. New code lives under
`baseline3/`.

## Task definition

- Input: all 24 target-day weather values plus cyclical calendar/time features.
- No historical energy observations are used.
- Output: one joint daily trajectory with shape `[4, 24]` in the fixed order
  `Electricity, Heat, Cooling, PV`.
- Both the generator and critic receive exactly the same conditions. This makes
  the model a conditional WGAN-GP rather than an unconditional WGAN-GP.
- Split: 2014--2020 train, 2021 validation, 2022 test.
- Normalization: channel statistics are fitted on 2014--2020 only.

The model is single-stage. It does not require OOF predictions. A 128-D random
vector represents scenario uncertainty; one generator forward pass produces a
complete four-channel day. The critic has no sigmoid and no batch
normalization, and is regularized with the standard interpolation gradient
penalty.

## Data files

Set the paths in `baseline3/configs/heew.yaml`, or override them on the command
line. The energy CSV must contain:

```text
Year, Month, Day, Hour, Electricity, Heat, Cooling, PV
```

The weather CSV must contain the timestamp columns and the selected weather
features. The default `pv10` set uses the six base HEEW weather columns plus
`ALLSKY_SFC_SW_DWN`, `CLRSKY_SFC_SW_DWN`, `PV_CLEARNESS_RATIO`, and
`PV_IS_DAYLIGHT`.

## Environment

Run all commands from the repository root:

```bash
pip install -r requirements_baseline3.txt
python -m baseline3.train --config baseline3/configs/heew.yaml
```

Running the module form avoids import errors caused by starting a package file
from inside `baseline3/`.

## Fixed six-configuration tuning budget

The search space in `baseline3/configs/search_space.yaml` contains exactly six
predeclared configurations:

| ID | latent dim | lr G / lr D | critic steps | GP lambda |
|---|---:|---:|---:|---:|
| c01_base | 128 | 1e-4 / 1e-4 | 5 | 10 |
| c02_z64 | 64 | 1e-4 / 1e-4 | 5 | 10 |
| c03_z256 | 256 | 1e-4 / 1e-4 | 5 | 10 |
| c04_lr_low | 128 | 5e-5 / 5e-5 | 5 | 10 |
| c05_lr_high | 128 | 2e-4 / 2e-4 | 5 | 10 |
| c06_critic3 | 128 | 1e-4 / 1e-4 | 3 | 10 |

Run the complete budget once with tuning seed 42:

```bash
python -m baseline3.sweep \
  --config baseline3/configs/heew.yaml \
  --search-space baseline3/configs/search_space.yaml \
  --seed 42
```

Each configuration uses the same architecture capacity, epoch ceiling, early
stopping rule, batch size, data, and validation scenario count. The winner is
the lowest 2021 validation `macro_nCRPS`; the script writes
`tuning_results.csv` and `best_config.json`. Do not add trials after examining
2022 test results.

After choosing the winner, retrain that configuration with final seeds
`42, 123, 777, 2024, 3407`. For example, if `c01_base` wins:

```bash
python -m baseline3.train --config baseline3/configs/heew.yaml --run-name cwgangp_final --set run.seed=42
python -m baseline3.train --config baseline3/configs/heew.yaml --run-name cwgangp_final --set run.seed=123
python -m baseline3.train --config baseline3/configs/heew.yaml --run-name cwgangp_final --set run.seed=777
python -m baseline3.train --config baseline3/configs/heew.yaml --run-name cwgangp_final --set run.seed=2024
python -m baseline3.train --config baseline3/configs/heew.yaml --run-name cwgangp_final --set run.seed=3407
```

If another configuration wins, append its overrides from `best_config.json` to
all five commands. This is deliberate: the six configurations tune
hyperparameters on one fixed seed, while five final seeds estimate uncertainty
for the selected method.

## Test evaluation

Generate 100 scenarios for every 2022 test day:

```bash
python -m baseline3.evaluate \
  --checkpoint experiments/baseline3/cwgangp_final_seed42/best.pt \
  --scenarios 100
```

The evaluator now uses the same metric definitions as CSDI baseline5
(`17c217b`, paper VS definition `cross_channel_unit_weights_v1`).
This output-only update does not change the generator, critic, gradient
penalty, training, checkpoint selection, latent sampling, or the **six-trial
tuning budget**. Existing checkpoints remain valid; no retraining is needed.

Install the additional scoring dependency if necessary:

```bash
python -m pip install "scipy>=1.10"
```

Use a **new empty output directory** when evaluating updated metrics:

```bash
python -m baseline3.evaluate \
  --checkpoint experiments/baseline3/cwgangp_final_seed42/best.pt \
  --outdir evaluation_results/baseline3_seed42_aligned
```

Defaults remain 100 scenarios and batch size 32, as in the previous evaluator.
Keep scenarios, batch size and seed fixed to reproduce a previous draw.
The default evaluation seed is the checkpoint's run.seed. Generator noise uses
a dedicated RNG with **sample_seed = seed + 90000**, unchanged from before.
`--output-dir` / `--num-scenarios` are aliases of `--outdir` / `--scenarios`;
`--no-plots` skips figures and Pearson comparison scores, as in baseline5.

### Output files

- `channel_metrics.csv`: four channel rows with MAE, RMSE, MAE_Z, RMSE_Z,
  R2, CRPS, nCRPS, Precision_Z, Recall_Z, CR, IW, IS, IS_Z.
- `global_metrics.json`, `metrics_summary.json`, and `global_metrics.csv`:
  all metrics, joint ES/VS and reproducibility metadata. The CSV retains
  the previous **metric,value** format. `global_metrics_wide.csv` additionally
  provides one row with metrics as columns.
- `daily_scores.csv`: date, daily ES_Z/VS_Z and per-channel CRPS, IS, IS_Z.
- `metric_definitions.json`: score formulas, normalization and pair conventions.
- `baseline3_scenarios.npz`: physical scenarios [day, scenario, channel, hour],
  physical targets, normalized conditions and PV-year controls, dates,
  training target_mean/target_std, channel names, evaluation seed, actual
  sample_seed, sampler and metadata JSON. Existing array names are retained.
- `global_pearson.png`: real-data global Pearson heatmap.
- `pearson/`: real/generated Pearson heatmaps and absolute differences.
- `random_timeseries_50/`: up to 50 distinct random test days.

### Metric conventions

- Point estimates are the **scenario mean**. R2 pools all test days and hours
  separately per channel. Constant-target R2 is undefined (JSON null / CSV blank).
- MAE_Z/RMSE_Z and joint scores use **training-channel z-scores**, not min-max.
- nCRPS is sum(CRPS) / (sum(abs(truth)) + 1e-8). `macro_nCRPS` averages the
  four channel nCRPS values; `mean_nCRPS` retains the legacy pooled score.
- CR, IW and IS use central **95%** intervals (linear 2.5%/97.5% quantiles).
  IS_Z divides physical IS by the training std. IS95 aliases IS.
  Legacy Coverage90/IntervalWidth90 remain separate global fields.
- ES/ES_Z evaluate standardized 96-dimensional complete daily trajectories
  using the empirical ensemble formula with the S-squared denominator.
- VS/VS_Z use p=0.5 and **3456 cross-channel unordered pairs**, including
  both same-hour and different-hour pairs. Weight 1 across channels and
  0 within a channel. **Sum pairs, then average days; do not divide by 3456.**
- Precision_Z/Recall_Z retain the same nearest-center-radius approximation
  as baseline5, **not the full union-of-kNN-balls estimator**. Defaults:
  k=5, cap=10000 generated daily trajectories per channel, subsampling seed
  = evaluation seed + 1000 + channel index. Optional overrides are
  `evaluation.precision_recall_k` and `evaluation.max_precision_samples`.
- Generated PV retains physical nonnegative clipping before scoring.
- WGAN-GP has **no diffusion T or reverse sampling steps**. Metadata records
  those fields as null and generator_forward_passes_per_scenario=1;
  it does not invent a diffusion-step count for GAN.

### Rescore saved scenarios without training or sampling

For an existing archive, keep its original `global_metrics.json` beside it:

```bash
python -m baseline3.score_npz \
  --npz evaluation_results/baseline3_seed42/baseline3_scenarios.npz \
  --outdir evaluation_results/baseline3_seed42_rescored
```

New archives contain seed and normalization metadata. **Old baseline3 archives
already contain training mean/std, but did not contain the seed**: the rescorer
recovers it from the accompanying JSON. If that JSON was moved, pass
`--metrics-json /path/to/original/global_metrics.json`. If seed provenance is
unavailable, explicitly supply `--seed` for metric subsampling and plot
selection; it does not resample scenarios or change their original generation
seed. Directory names are never interpreted as authoritative seed metadata.

For an older/custom archive lacking training statistics, additionally supply
`--checkpoint` and the original CSVs (optionally via `--energy-path` and
`--weather-path`). Statistics are fitted only on 2014–2020, and saved test
dates/targets are checked. Legacy checkpoints lack training-data hashes, so
unchanged historical CSV rows cannot be cryptographically verified.
Never use test-set normalization.

Rescoring writes only new metric/plot outputs and leaves the source NPZ
untouched. Existing nonempty output directories are rejected.

Report table entries as mean ± standard deviation over training seeds.
Daily scores support paired comparisons on the same dates. Tests include
paper ES/VS formulas, legacy sampling equality at two seeds, output schemas,
seed recovery, offline rescoring, and the existing six-trial/training tests.

## Fair comparison checklist

1. Use identical target columns, weather/time conditions, split, and train-only
   normalization for all methods.
2. Tune every method with exactly six declared configurations on validation
   data only.
3. Use 20 scenarios per validation day and 100 scenarios per test day.
4. Select checkpoints with the same primary criterion: validation
   `macro_nCRPS`.
5. Evaluate all methods with the same shared metric implementation where
   possible; never tune on the 2022 test set.
6. Distinguish training budget from sampling budget: WGAN-GP has no diffusion
   sampling steps and produces each scenario in one forward pass.
