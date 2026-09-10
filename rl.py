"""
RL training overview CLI.

Usage
-----
    python rl.py --scenario a3_motorway --experiment rl-tests --control coordinated

Output
------
    scenarios/<scenario>/micro-sim-runs/<experiment>/rl-overview/<control>/<YYYY-MM-DDTHH-MM-SS>/
        epsilon_decay.png
        total_reward.png
        q_table_heatmap.png
        exploration_rate.png
        tts_per_episode.png
        action_distribution.png
        reward_components.png   (only if reward_breakdown data exists)
        summary.json
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from microscopic.visualisation.rl_overview import RLOverviewVisualiser


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RL training overview plots")
    parser.add_argument("--scenario", default="a3_motorway", help="Scenario folder name")
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Comma-separated scenario pool to aggregate (must match the pool used for training)",
    )
    parser.add_argument(
        "--pool-name",
        dest="pool_name",
        default=None,
        help="Override the auto-generated pool folder name (default: sorted scenario names joined by '+')",
    )
    parser.add_argument("--experiment", default="rl-tests", help="Experiment sub-folder name")
    parser.add_argument("--control", default="coordinated",
                        help="Control type to filter (run_type in index.json)")
    args = parser.parse_args()

    if args.scenarios is not None:
        from scenarios import pool_name_for

        scenario_pool = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        pool_name = args.pool_name or pool_name_for(scenario_pool)
        experiment_dirs = [
            Path("scenarios") / s / "micro-sim-runs" / args.experiment for s in scenario_pool
        ]
        pool_dir = Path("scenarios") / "_pools" / pool_name / args.experiment
        pool_control_dir = pool_dir / args.control

        vis = RLOverviewVisualiser(pool_dir, args.control)
        vis.load_pool(experiment_dirs, pool_control_dir)

        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        out_dir = pool_dir / "rl-overview" / args.control / ts
        saved = vis.render(out_dir=out_dir)
    else:
        experiment_dir = (
            Path("scenarios") / args.scenario / "micro-sim-runs" / args.experiment
        )
        vis = RLOverviewVisualiser(experiment_dir, args.control)
        vis.load()
        saved = vis.render()

    print(f"\nRL overview saved to: {saved[0].parent}")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
