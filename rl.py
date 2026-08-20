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
from pathlib import Path

from microscopic.visualisation.rl_overview import RLOverviewVisualiser


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RL training overview plots")
    parser.add_argument("--scenario", default="a3_motorway", help="Scenario folder name")
    parser.add_argument("--experiment", default="rl-tests", help="Experiment sub-folder name")
    parser.add_argument("--control", default="coordinated",
                        help="Control type to filter (run_type in index.json)")
    args = parser.parse_args()

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
