"""Open-loop sanity sweep: macroscopic coordinated MPC with a fixed alpha_compliance."""
import argparse
import csv
import time

import numpy as np

from macroscopic.solvers import MpcCoordinatedSimulation
from scenarios import Scenario

DEFAULT_ALPHAS = [0.0, 0.01, 0.017, 0.032, 0.056, 0.1, 0.15, 0.2]


def run_one(scenario_name: str, alpha: float) -> dict:
    scenario = Scenario(scenario_name)
    env = scenario.build_environment_with_alpha("coordinated", alpha)
    network = env.network
    solver = MpcCoordinatedSimulation(
        environment=env,
        Np=7,
        Nc=5,
        M=6,
        function_compactness=2,
        initial_conditions={network.origins_by_name["O1"]: {"v_ctrl": np.inf}},
        objective_weights=env.objective_weights,
    )
    solver.setup()
    solver.simulate()
    m = solver.overview_results
    res = solver.simulation_results
    v = np.asarray(res.v_ctrl, dtype=float)
    v = v.reshape(v.shape[0], -1) if v.ndim > 1 else v.reshape(-1, 1)
    vmax = float(np.max(v[np.isfinite(v)]))
    v = np.where(np.isfinite(v), v, vmax)
    # Fraction of the final quarter of the run (after congestion) at the unrestricted limit
    tail = v[-max(1, len(v) // 4):]
    return {
        "scenario": scenario_name,
        "alpha": alpha,
        "tts": m.total_time_spent,
        "wait": m.total_wait_time,
        "speed_var": m.total_speed_variance,
        "congestion_h": m.congestion_time,
        "excess_cong": m.excess_congestion_time,
        "vsl_max": vmax,
        "vsl_mean": float(v.mean()),
        "vsl_min": float(v.min()),
        "vsl_tail_mean": float(tail.mean()),
        "vsl_tail_frac_unrestricted": float((tail >= vmax - 1e-6).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["s1_motorway_medium", "s2_motorway_medium", "s4_motorway_medium"])
    ap.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    ap.add_argument("--out", default="alpha_sweep.csv")
    args = ap.parse_args()

    rows = []
    for sc in args.scenarios:
        for a in args.alphas:
            t0 = time.time()
            row = run_one(sc, a)
            row["runtime_s"] = round(time.time() - t0, 1)
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)


if __name__ == "__main__":
    main()
