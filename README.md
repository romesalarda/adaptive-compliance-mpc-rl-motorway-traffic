# Adaptive Compliance Model Predictive Control with Reinforcement learning for Motorway Traffic Control

Model predictive control (MPC) and tabular Q-learning for motorway traffic management. The controllers pick a variable speed limit (VSL), ramp metering (RM), or both (coordinated VSL + RM). The trade-off between travel time and safety is set by `alpha_compliance` in [0, 1]. Each control scheme can run in two settings:

- **Macroscopic**: an open-loop METANET-style model solved with `sym_metanet` and `casadi`.
- **Microscopic**: closed-loop SUMO simulation, where the MPC runs asynchronously over TraCI and can be driven by a Q-learning agent that picks `alpha_compliance` online.

## Contents

- [Components](#components)
- [Installation](#installation)
- [Repository layout](#repository-layout)
- [Scenarios](#scenarios)
- [Macroscopic runs: `macro_main.py`](#macroscopic-runs-macro_mainpy)
- [Microscopic runs: `micro_main.py`](#microscopic-runs-micro_mainpy)
- [RL training and evaluation](#rl-training-and-evaluation)
- [Alpha sweep: `alpha_sweep.py`](#alpha-sweep-alpha_sweeppy)
- [Bootstrap confidence intervals: `bootstrap.py`](#bootstrap-confidence-intervals-bootstrappy)
- [RL overview: `rl.py`](#rl-overview-rlpy)
- [Report: `report.main`](#report-reportmain)
- [Output reference](#output-reference)
- [Reproducibility](#reproducibility)
- [Known limitations](#known-limitations)
- [License](#license)

## Components

| Entry point | Purpose |
|---|---|
| `macro_main.py` | Solve a single scenario with the macroscopic MPC (or batch all control types). |
| `micro_main.py` | Closed-loop SUMO simulation with the async MPC. Handles single runs, repeats, batch, fixed alpha, and RL training. |
| `alpha_sweep.py` | Open-loop macroscopic sweep of `alpha_compliance` for the coordinated controller, written to CSV. |
| `bootstrap.py` | Case-resampling bootstrap over a folder of repeated micro runs, with a Plotly HTML dashboard. |
| `rl.py` | Aggregates RL training runs for one experiment and renders the RL overview (learning curves, Q-table, policy). |
| `report/main.py` | Builds the report figures and tables from the `data/` folder. Depends on gitignored data. |

## Installation

Requirements: Python (use the version your `casadi`/`numpy` pins support) and [Eclipse SUMO](https://sumo.dlr.de/) 1.24.0, matching the pinned `eclipse-sumo`/`traci` versions. Only the microscopic runs need SUMO.

```bash
python -m venv venv
venv\Scripts\activate          # Windows (use source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Set `SUMO_HOME` to the SUMO installation directory before running any microscopic command. `micro_main.py` exits with an error if it is missing.

```bash
set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo    # Windows cmd
export SUMO_HOME=/usr/share/sumo                     # Linux/macOS
```

The microscopic network files (`micronet/`) and any local `data/` folder are gitignored. Each micro scenario expects `micronet/<network>/motorway.sumocfg` to exist, so place the network files there before running micro simulations.

## Repository layout

```
macro_main.py          macroscopic single/batch entry point
micro_main.py          microscopic entry point (SUMO + async MPC + RL)
alpha_sweep.py         macroscopic alpha sweep -> CSV
bootstrap.py           bootstrap CI over repeated micro runs
rl.py                  RL overview for one experiment
macroscopic/           METANET model, MPC solvers, run statistics
microscopic/           SUMO simulation, async controller, Q-learning agent,
                       bootstrap service, RL visualisation
scenarios/             one folder per scenario (scenario.json + outputs)
report/                report generator (main.py, report_config.json)
sweeps/                tracked outputs from the alpha sweep
requirements.txt       pinned dependencies
```

## Scenarios

Each folder under `scenarios/` contains a `scenario.json` with these blocks:

- `parameters`: link and traffic parameters for the METANET model.
- `control_parameters`: per-controller settings (`solve_interval_steps`, `Nc`, and so on). Keys: `vsl`, `rm`, `coordinated`.
- `demands`: origin demand profile (interpolated from a descriptor such as `hourly-peak`).
- `micro`: SUMO settings (`step_length`, `duration_steps`, `n_segments`, `sumo_config`).
- `rl.configs`: named RL configurations (state features, actions, reward, epsilon schedule).

Available scenarios:

| Scenario | Heterogeneity | Notes |
|---|---|---|
| `s1_motorway_high` / `s1_motorway_medium` / `s1_motorway_low` | high / medium / low | A3 motorway, three heterogeneity levels |
| `s2_motorway_medium` | medium | |
| `s3_motorway_medium` | medium | |
| `s4_motorway_medium` | medium | |
| `a1_motorway` | | Macroscopic only (no `rl` block) |

Run `python -c "from scenarios import Scenario; print(Scenario.list_scenarios())"` to list scenarios from the current checkout.

Always pass `--scenario` explicitly. The default values in `micro_main.py`, `rl.py`, and the `bootstrap.py` docstring name scenarios that do not exist in the repository.

## Macroscopic runs: `macro_main.py`

Runs the macroscopic MPC on one scenario. Horizon settings are fixed in the script: `Np=7`, `Nc=3` (`Nc=5` for coordinated), `M=6`, `function_compactness=2`.

```bash
# single control type
python macro_main.py --scenario s1_motorway_medium --control coordinated

# all control types in sequence (vsl, rm, base, coordinated)
python macro_main.py --scenario s1_motorway_medium --batch

# interactive: prompts for any missing value
python macro_main.py
```

| Option | Description |
|---|---|
| `--scenario NAME` | Scenario folder name. Prompted if omitted. |
| `--control {vsl,rm,base,coordinated}` | Single control type. Mutually exclusive with `--batch`. |
| `--batch` | Run every control type in sequence. |

Results go to `scenarios/<scenario>/macro-sim-runs/<scenario>/`. Each run calls `plt.show()` at the end, so a plot window opens and blocks until you close it. Close it before the next run in a batch starts.

## Microscopic runs: `micro_main.py`

Runs a closed-loop SUMO simulation. The MPC runs asynchronously in a worker thread, and the SUMO loop pushes state and receives controls.

### Single run

```bash
python micro_main.py --scenario s1_motorway_medium --solver coordinated --no-gui --no-wait
```

Drop `--no-gui` to watch the run in `sumo-gui`. Use `--no-wait` to skip the pre-run confirmation prompt.

### Repeats (same scenario, same solver)

```bash
python micro_main.py --scenario s1_motorway_medium --solver coordinated \
    --no-gui --no-wait --repeat 20 --experiment 0.01-runs
```

Repeats are only applied to a single solver and no `--batch`. With multiple solvers, `--repeat` is forced to 1. Each run is written to its own folder under the experiment.

### Fixed alpha

```bash
python micro_main.py --scenario s1_motorway_medium --solver coordinated \
    --alpha 0.032 --no-gui --no-wait --experiment fixed-0.032
```

`--alpha` builds the MPC with a constant `alpha_compliance` for any solver type.

### Multiple solvers (batch)

```bash
python micro_main.py --scenario s1_motorway_medium --solver vsl rm coordinated --no-gui --no-wait
```

Passing several values to `--solver` triggers batch mode, which runs the solvers in parallel. Batch mode does not currently support RL, `--alpha`, or `--no-wait` (see [Known limitations](#known-limitations)).

### Options

| Option | Default | Description |
|---|---|---|
| `--scenario NAME` | `a2_motorway` (does not exist, always pass this) | Scenario folder name. |
| `--solver {rm,vsl,coordinated,base} [...]` | `rm` | One or more controllers. Several values trigger batch. |
| `--repeat N` | `1` | Repeat the run N times (single solver only). |
| `--experiment NAME` | `default` | Sub-folder under `micro-sim-runs/`. Use one per experiment. |
| `--alpha FLOAT` | none | Fixed `alpha_compliance` instead of the scenario default. |
| `--no-gui` | off | Run SUMO headless. |
| `--no-wait` | off | Do not wait for keyboard input between runs. |
| `--port INT` | `8813` | Base TraCI port. |
| `--seed INT` | random | Seed for the `--scenarios` stratified shuffle. |
| `--output DIR` | none | Override the output root. Uses the legacy `runs/<scenario>/<solver>_<hex>/` layout. |
| `--redirect-detectors` | off | Redirect detector output to the run folder. |
| `--verbose-descriptor` | off | Print the full run descriptor. |
| `--no-async-controller` | off | Run the MPC synchronously in the SUMO loop instead of in the worker thread. |
| `--rl` | off | Use the Q-learning agent to choose alpha. See [RL training](#rl-training-and-evaluation). |
| `--rl-config-name NAME` | `default` | Which `rl.configs.<name>` block to use. |
| `--no-learning` | off | Keep the Q-table fixed and act greedily (evaluation). |
| `--scenarios A,B,...` | none | Pool of scenarios for RL training (comma-separated). Requires `--rl`. |
| `--pool-name NAME` | `+`-joined sorted names | Name of the pool folder. |
| `--workers INT` | `2` | Accepted but not used (see limitations). |

## RL training and evaluation

With `--rl`, a tabular Q-learning agent observes a discretised state every MPC solve interval (`solve_interval_steps`), picks an alpha from the configured action set, and gets a reward from the interval's travel time and collisions.

Configurations live in `scenarios/<s>/scenario.json` under `rl.configs.<name>`:

- **state**: a list of features. Each feature has a `source` (`mean_density`, `mean_speed`, `ramp_queue`, or `demand_ratio`), `thresholds`, and `labels` (one more label than thresholds).
- **actions**: `alpha_values`, the candidate values of alpha.
- **reward**: components with a `weight`, `scale`, and optional `clip`. Supported components are `tts` (travel time, negative) and `collisions`.
- **epsilon / learning**: exploration schedule (`epsilon_decay_mode`: `per_step` or `per_episode`), learning rate, and discount.

### Training

```bash
python micro_main.py --scenario s1_motorway_medium --rl --rl-config-name v2 \
    --no-gui --no-wait --repeat 50 --experiment rl-v2
```

### Evaluation (greedy, no learning)

```bash
python micro_main.py --scenario s1_motorway_medium --rl --rl-config-name v2 \
    --no-learning --no-gui --no-wait --experiment rl-v2-eval
```

### Q-table persistence

- Training saves `q_table.npy` under `scenarios/<s>/micro-sim-runs/<experiment>/<control>/` and a per-episode snapshot in `q-tables/q-table-ep-N.npy`.
- If a Q-table already exists when a run starts, it is loaded (warm start). Using the same `--experiment` continues training from the saved table.
- With `--no-learning`, the table is loaded and epsilon is fixed at 0.

### Pool training

```bash
python micro_main.py --scenarios s1_motorway_medium,s2_motorway_medium \
    --rl --rl-config-name v2 --seed 1 --no-gui --no-wait --repeat 20
```

Pool mode writes a pool descriptor and Q-tables under `scenarios/_pools/<pool-name>/`. **The current code overwrites the stratified schedule with a hardcoded list** (`micro_main.py:748-757`), so `--seed` has no effect and `--repeat N` uses the first N entries of that list. Do not use pool results until this is fixed.

## Alpha sweep: `alpha_sweep.py`

Open-loop macroscopic sweep of `alpha_compliance` for the coordinated controller. Each row is one scenario and one alpha.

```bash
python alpha_sweep.py --scenarios s1_motorway_medium s2_motorway_medium \
    --alphas 0.0 0.01 0.032 0.1 --out alpha_sweep.csv
```

| Option | Default |
|---|---|
| `--scenarios NAME ...` | `s1_motorway_medium s2_motorway_medium s4_motorway_medium` |
| `--alphas FLOAT ...` | `0.0 0.01 0.017 0.032 0.056 0.1 0.15 0.2` |
| `--out PATH` | `alpha_sweep.csv` (current directory) |

The CSV is rewritten after every row, so partial results survive an interruption. Columns:

| Column | Meaning |
|---|---|
| `scenario`, `alpha` | Inputs |
| `tts`, `wait`, `speed_var` | Total time spent, total wait time, total speed variance |
| `congestion_h`, `excess_cong` | Congested time and excess congestion time |
| `vsl_max`, `vsl_mean`, `vsl_min` | Speed limit statistics over the run (infinite values replaced with the maximum) |
| `vsl_tail_mean` | Mean speed limit over the last quarter of the run |
| `vsl_tail_frac_unrestricted` | Fraction of the last quarter at the unrestricted limit |
| `runtime_s` | Wall-clock time per row |

Tracked results are in `sweeps/alpha_sweep_s{1,2,4}.csv`. The matching `sweep_s*.log` and `.err` files hold the Ipopt output.

## Bootstrap confidence intervals: `bootstrap.py`

Resamples whole runs (case resampling) from a folder of repeated micro runs and produces confidence intervals for density, speed, queue, control, applied rate, and tracking RMSE.

```bash
python bootstrap.py --path scenarios/s1_motorway_medium/micro-sim-runs/0.01-runs/coordinated \
    --n-bootstrap 2000 --ci 0.95 --seed 42
```

| Option | Default | Description |
|---|---|---|
| `--path FOLDER` | required | Folder of runs from one scenario and one solver. |
| `--output-dir DIR` | `--path` | Where the dashboard and summary are written. |
| `--filename NAME` | `bootstrap_analysis.html` | Dashboard file name. |
| `--n-bootstrap INT` | `1000` | Number of resamples. |
| `--ci FLOAT` | `0.95` | Confidence interval width. |
| `--seed INT` | none | Seed for reproducible resampling. |

Requirements, enforced by the script:

- At least 2 runs, each with `simulation.h5`, `meta.json`, and `micro_config.json`.
- All runs must share the same `scenario_id` and `run_type`. Otherwise the script exits with code 1.

Outputs: the Plotly dashboard (`bootstrap_analysis.html`) and `bootstrap_summary.json`.

## RL overview: `rl.py`

Collects the training runs for one experiment (and control type) and renders the RL overview.

```bash
python rl.py --scenario s1_motorway_medium --experiment rl-v2 --control coordinated

# pool: must match the pool used for training
python rl.py --scenarios s1_motorway_medium,s2_motorway_medium --experiment rl-pool --control coordinated
```

| Option | Default |
|---|---|
| `--scenario NAME` | `a3_motorway` (does not exist, always pass this) |
| `--scenarios A,B,...` | none (pool) |
| `--pool-name NAME` | derived from the pool |
| `--experiment NAME` | `rl-tests` |
| `--control TYPE` | `coordinated` (matched against `run_type` in `index.json`) |

Output: `scenarios/<scenario>/micro-sim-runs/<experiment>/rl-overview/<control>/<YYYY-MM-DDTHH-MM-SS>/`. Each invocation creates a new timestamped folder.

## Report: `report.main`

Builds the figures and tables from the static (fixed-alpha) and RL runs listed in `report/report_config.json`.

```bash
python -m report.main            # uses cached bootstrap aggregation
python -m report.main --no-cache # recompute
```

Outputs go to `report/output/` (`figures/`, `tables/`, `cache/`). This command reads from `data/`, which is gitignored and not included in the repository. Before it runs, copy the static run folders and RL pool folders referenced in `report_config.json` into `data/`, or edit the config paths to point at your local runs.

## Output reference

| Path | Written by | Contents |
|---|---|---|
| `scenarios/<s>/macro-sim-runs/<s>/` | `macro_main.py` | Macroscopic run sets (`RunSet`) |
| `scenarios/<s>/micro-sim-runs/<exp>/<control>/<run>/` | `micro_main.py` | `simulation.h5`, `meta.json`, `micro_config.json`, `async_mpc_stats.h5` |
| `scenarios/<s>/micro-sim-runs/<exp>/index.json` | `micro_main.py` | Index of runs in the experiment |
| `scenarios/<s>/micro-sim-runs/<exp>/<control>/rl_stats.h5` | RL runs | Per-step RL records (state, action, reward, epsilon, Q-values) |
| `scenarios/<s>/micro-sim-runs/<exp>/<control>/q_table.npy` | RL runs | Q-table with `q_table` and `meta` keys |
| `scenarios/<s>/micro-sim-runs/<exp>/<control>/q-tables/` | RL runs | Q-table snapshot per episode |
| `scenarios/_pools/<pool>/` | pool runs | Schedule and pool descriptor |
| `Errors/` and `<run>/error.json` | any failed run | Crash information |
| `alpha_sweep.csv` | `alpha_sweep.py` | Sweep results |
| `bootstrap_analysis.html`, `bootstrap_summary.json` | `bootstrap.py` | Bootstrap dashboard and summary |
| `report/output/` | `report.main` | Figures, tables, cache |

Note that `scenarios/*/micro-sim-runs/`, `macro-sim-runs/`, `_pools/`, `data/`, and `Errors/` are gitignored, except for specific files already committed. Most result folders are local only.

## Reproducibility

- Pin the dependencies in `requirements.txt`, especially `eclipse-sumo`, `traci`, `casadi`, and `sym_metanet`. Solver results change with the Ipopt build.
- Use `--seed` for `bootstrap.py`, which is deterministic. Pool scheduling and RL exploration are not reproducible yet (see limitations).
- Record the exact `--experiment` name, `--rl-config-name`, and `alpha_sweep.py` arguments with each result. Each run's `meta.json` and `micro_config.json` store the scenario configuration used.

## Known limitations

These are known defects at the time of writing:

- **Pool schedule override**: `micro_main.py:757` replaces the schedule built by `_build_scenario_schedule` with a hardcoded list. `--seed` has no effect in pool mode.
- **Unused `--workers`**: parsed but never used.
- **Batch mode ignores some flags**: `--rl`, `--alpha`, and `--no-wait` are not applied in batch. Batch workers can block on `input()`.
- **Unsupported RL state source**: the default RL configs for `s1_motorway_medium`, `s2_motorway_medium`, `s3_motorway_medium`, and `s4_motorway_medium` use the `tts_trend` source, which is not implemented. `--rl` with those defaults fails during setup. Use the `v2` configs (`mean_density` + `demand_ratio`) or implement `tts_trend` in `microscopic/agents/state_features.py`.
- **Ignored reward keys**: `window_seconds` in the reward config is read but not used.
- **Unseeded RL**: Q-learning exploration uses an unseeded random generator, so training runs are not reproducible.
- **No tests**: the repository has no automated test suite.
- **Missing inputs**: `micronet/` and `data/` are not in the repository, so microscopic runs and the report need them supplied locally. - an example of a microscopic simulation setup i.e. the SUMO config files are included in scenarios\examples which you can copy and paste into the scenario folder along with network.py and scenario.json. data/ is not required to run the commands but is an output from the runs themselves.

## License

MIT. See [LICENSE](LICENSE).
