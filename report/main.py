"""Entry point: generate all IEEE report figures + the overview table from existing run data.

Usage:
    python -m report.main [--no-cache]
"""

from __future__ import annotations

import argparse

from macroscopic.logger import LOGGER
from report.data_loader import (
    load_bootstrap,
    load_config,
    load_demand_profile,
    load_q_table_snapshot,
    load_rl_action_counts,
    load_runs_log_upto_episode,
    load_state_action_order,
)
from report.plots import (
    plot_demand_profiles,
    plot_q_table_heatmap,
    plot_rl_alpha_selection,
    plot_rmse,
    plot_rmse_full_horizon_combined,
    plot_tts_collisions,
)
from report.tables import build_overview_table, build_q_table_frame, write_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cache", action="store_true", help="Recompute bootstrap aggregation, ignoring cache.")
    args = parser.parse_args()
    use_cache = not args.no_cache

    cfg = load_config()
    LOGGER.info(f"Loaded report config for {len(cfg.scenarios)} scenarios")

    results = {}
    demand_profiles = {}
    action_counts = {}
    for sc in cfg.scenarios:
        LOGGER.info(f"--- {sc.label} ({sc.real_scenario_id}) ---")
        results[sc.id] = {
            "static": load_bootstrap(sc.static_path, cfg, use_cache=use_cache),
            "rl": load_bootstrap(sc.rl_path, cfg, use_cache=use_cache),
        }
        demand_profiles[sc.id] = load_demand_profile(sc.scenario_config_path)
        action_counts[sc.id] = load_rl_action_counts(sc.rl_path)

    plot_demand_profiles(cfg.scenarios, demand_profiles, cfg)
    plot_tts_collisions(results, cfg.scenarios, cfg)
    plot_rmse("density", results, cfg.scenarios, cfg)
    plot_rmse("speed", results, cfg.scenarios, cfg)
    plot_rmse_full_horizon_combined(results, cfg.scenarios, cfg)
    plot_rl_alpha_selection(action_counts, cfg.scenarios, cfg.action_values, cfg.static_alpha, cfg)

    overview_df = build_overview_table(results, cfg.scenarios, cfg)
    write_table(overview_df, cfg.tables_dir / "overview")

    # --- shared mixed-pool Q-table snapshot, cross-checked against the training log ---
    LOGGER.info(f"--- RL policy snapshot (episode {cfg.rl_policy.episode}) ---")
    load_runs_log_upto_episode(cfg.rl_policy)  # sanity-check log only, for console verification
    q_table, q_meta = load_q_table_snapshot(cfg.rl_policy)
    state_order, descriptor_action_values = load_state_action_order(cfg.rl_policy.descriptor_path)
    if descriptor_action_values != cfg.action_values:
        LOGGER.warning(
            f"rl_descriptor.json action_values {descriptor_action_values} differ from report_config.json "
            f"action_values {cfg.action_values}"
        )
    plot_q_table_heatmap(q_table, state_order, descriptor_action_values, cfg.rl_policy.episode, cfg)
    q_table_df = build_q_table_frame(q_table, state_order, descriptor_action_values, cfg.rl_policy.episode)
    write_table(q_table_df, cfg.tables_dir / "q_table")

    LOGGER.info(f"Done. Figures in {cfg.figures_dir}, tables in {cfg.tables_dir}")


if __name__ == "__main__":
    main()
