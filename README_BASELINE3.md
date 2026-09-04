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

The evaluator writes:

- `global_metrics.csv` and `global_metrics.json`;
- physical-unit RMSE, MAE, CRPS, nCRPS, 90% coverage and interval width;
- per-channel normalized RMSE/MAE and normalized precision/recall;
- per-channel 95% CR and IW;
- real/generated Pearson correlation plots and errors;
- `random_timeseries_50/` visual comparisons;
- `baseline3_scenarios.npz` with scenarios shaped `[day, scenario, channel, hour]`;
- sampling time for the one-pass GAN generator.

Report each final table entry as mean ± standard deviation over the five seeds.
For paired significance testing, retain the per-day outputs in every `.npz` and
compare methods on the same test days and seeds.

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
