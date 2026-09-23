"""Load report_config.json, aggregate repeated-run folders (static + RL) via bootstrap,
and gather RL alpha-selection statistics and demand profiles for the IEEE report figures."""

from __future__ import annotations

import hashlib
import itertools
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from macroscopic.logger import LOGGER
from microscopic.services.bootstrap import RunFolderBootstrapper

REPORT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPORT_ROOT.parent


@dataclass
class ScenarioConfig:
    id: str
    label: str
    real_scenario_id: str
    static_path: Path
    rl_path: Path
    scenario_config_path: Path


@dataclass
class RLPolicyConfig:
    pool_path: Path
    runs_log_path: Path
    descriptor_path: Path
    q_tables_dir: Path
    filename_pattern: str
    episode: int


@dataclass
class ReportConfig:
    scenarios: list[ScenarioConfig]
    action_values: list[float]
    static_alpha: float
    rl_policy: RLPolicyConfig
    n_bootstrap: int
    ci: float
    seed: Optional[int]
    cache_enabled: bool
    cache_dir: Path
    output_dir: Path
    figures_dir: Path
    tables_dir: Path
    formats: list[str]
    dpi: int


def load_config(path: Path | str = REPORT_ROOT / "report_config.json") -> ReportConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    scenarios = [
        ScenarioConfig(
            id=s["id"],
            label=s["label"],
            real_scenario_id=s["real_scenario_id"],
            static_path=PROJECT_ROOT / s["static_path"],
            rl_path=PROJECT_ROOT / s["rl_path"],
            scenario_config_path=PROJECT_ROOT / s["scenario_config"],
        )
        for s in raw["scenarios"]
    ]

    rl_policy_raw = raw["rl_policy"]
    pool_path = PROJECT_ROOT / rl_policy_raw["pool_path"]
    rl_policy = RLPolicyConfig(
        pool_path=pool_path,
        runs_log_path=pool_path / rl_policy_raw["runs_log_file"],
        descriptor_path=pool_path / rl_policy_raw["descriptor_file"],
        q_tables_dir=pool_path / rl_policy_raw["q_tables_dir"],
        filename_pattern=rl_policy_raw["q_table_filename_pattern"],
        episode=rl_policy_raw["episode"],
    )

    return ReportConfig(
        scenarios=scenarios,
        action_values=raw["rl"]["action_values"],
        static_alpha=raw["rl"]["static_alpha"],
        rl_policy=rl_policy,
        n_bootstrap=raw["bootstrap"]["n_bootstrap"],
        ci=raw["bootstrap"]["ci"],
        seed=raw["bootstrap"].get("seed"),
        cache_enabled=raw["cache"]["enabled"],
        cache_dir=PROJECT_ROOT / raw["cache"]["dir"],
        output_dir=PROJECT_ROOT / raw["output"]["dir"],
        figures_dir=PROJECT_ROOT / raw["output"]["figures_dir"],
        tables_dir=PROJECT_ROOT / raw["output"]["tables_dir"],
        formats=raw["output"]["formats"],
        dpi=raw["output"]["dpi"],
    )


def _cache_key(folder: Path, n_bootstrap: int, ci: float, seed: Optional[int]) -> str:
    raw = f"{folder}|{n_bootstrap}|{ci}|{seed}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def load_bootstrap(folder: Path, cfg: ReportConfig, *, use_cache: bool = True) -> dict[str, Any]:
    """Aggregate a repeated-run folder via RunFolderBootstrapper, with optional on-disk caching."""
    cache_path = None
    if use_cache and cfg.cache_enabled:
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cfg.cache_dir / f"{_cache_key(folder, cfg.n_bootstrap, cfg.ci, cfg.seed)}.pkl"
        if cache_path.exists():
            LOGGER.info(f"Using cached bootstrap results for {folder}")
            with cache_path.open("rb") as fh:
                return pickle.load(fh)

    LOGGER.info(f"Processing run folder: {folder}")
    bootstrapper = RunFolderBootstrapper(folder, n_bootstrap=cfg.n_bootstrap, ci=cfg.ci, seed=cfg.seed)
    aggregated = bootstrapper.aggregate()
    LOGGER.info(f"Finished processing {folder} ({aggregated['n_runs']} runs)")

    if cache_path is not None:
        with cache_path.open("wb") as fh:
            pickle.dump(aggregated, fh)

    return aggregated


def load_demand_profile(scenario_config_path: Path) -> dict[str, Any]:
    """Read the mainline/on-ramp demand profile (xp/fp keypoints) from a scenario.json file."""
    with scenario_config_path.open(encoding="utf-8") as fh:
        scenario = json.load(fh)
    return scenario["demands"]


def load_runs_log_upto_episode(rl_policy: RLPolicyConfig) -> list[dict[str, Any]]:
    """Filter the mixed-pool runs_log.json down to entries with total_episodes <= the configured episode."""
    with rl_policy.runs_log_path.open(encoding="utf-8") as fh:
        log = json.load(fh)
    all_runs = log["runs"]
    entries = [r for r in all_runs if r.get("total_episodes", -1) <= rl_policy.episode]
    LOGGER.info(
        f"Filtered {rl_policy.runs_log_path.name} to {len(entries)}/{len(all_runs)} entries "
        f"with total_episodes <= {rl_policy.episode}"
    )
    return entries


def _q_table_regex(pattern: str) -> re.Pattern:
    escaped = re.escape(pattern).replace(re.escape("{episode}"), r"(\d+)")
    return re.compile(f"^{escaped}$")


def verify_q_table_naming(q_tables_dir: Path, filename_pattern: str) -> dict[int, Path]:
    """Map episode number -> file for all snapshots matching the standardised naming pattern, warning on outliers."""
    regex = _q_table_regex(filename_pattern)
    episodes: dict[int, Path] = {}
    unmatched: list[str] = []
    for f in q_tables_dir.iterdir():
        if not f.is_file():
            continue
        m = regex.match(f.name)
        if m:
            episodes[int(m.group(1))] = f
        else:
            unmatched.append(f.name)

    if unmatched:
        LOGGER.warning(f"{len(unmatched)} file(s) in {q_tables_dir} don't match pattern {filename_pattern!r}: {unmatched}")
    if episodes:
        LOGGER.info(f"Found {len(episodes)} standardised Q-table snapshots in {q_tables_dir} (episodes {min(episodes)}-{max(episodes)})")
    else:
        LOGGER.warning(f"No Q-table snapshots matching {filename_pattern!r} found in {q_tables_dir}")
    return episodes


def load_q_table_snapshot(rl_policy: RLPolicyConfig) -> tuple[dict[tuple, np.ndarray], dict[str, Any]]:
    """Load and verify the Q-table snapshot for the configured episode, cross-checking snapshot metadata."""
    episodes = verify_q_table_naming(rl_policy.q_tables_dir, rl_policy.filename_pattern)
    if rl_policy.episode not in episodes:
        available = sorted(episodes)
        raise FileNotFoundError(
            f"No Q-table snapshot for episode {rl_policy.episode} in {rl_policy.q_tables_dir}. "
            f"Available episodes: {available}"
        )

    path = episodes[rl_policy.episode]
    LOGGER.info(f"Loading Q-table snapshot: {path}")
    payload = np.load(path, allow_pickle=True).item()
    q_table = payload.get("q_table", payload)
    meta = payload.get("meta", {})

    if meta.get("total_episodes") != rl_policy.episode:
        LOGGER.warning(
            f"Q-table file name implies episode {rl_policy.episode} but meta.total_episodes="
            f"{meta.get('total_episodes')} — verify this is the intended snapshot"
        )
    return q_table, meta


def load_state_action_order(descriptor_path: Path) -> tuple[list[tuple[str, ...]], list[float]]:
    """Canonical (state, action) ordering from rl_descriptor.json, for consistent heatmap/table axes."""
    with descriptor_path.open(encoding="utf-8") as fh:
        descriptor = json.load(fh)
    label_lists = [f["labels"] for f in descriptor["state"]["features"]]
    state_order = list(itertools.product(*label_lists))
    action_values = descriptor["actions"]["alpha_values"]
    return state_order, action_values


def load_rl_action_counts(rl_path: Path) -> dict[str, int]:
    """Sum RL action_counts (alpha selection frequency) across all episodes in an RL run folder."""
    episode_dirs = sorted(d for d in rl_path.iterdir() if d.is_dir() and (d / "rl_stats.json").exists())
    LOGGER.info(f"Aggregating RL action counts across {len(episode_dirs)} episodes in {rl_path}")

    counts: dict[str, int] = {}
    for ep_dir in episode_dirs:
        with (ep_dir / "rl_stats.json").open(encoding="utf-8") as fh:
            stats = json.load(fh)
        for alpha, n in stats["summary"]["action_counts"].items():
            counts[alpha] = counts.get(alpha, 0) + n
    return counts
