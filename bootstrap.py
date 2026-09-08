"""
Bootstrap statistical analysis across a folder of repeated micro-sim runs.

The folder must contain repeated runs of the SAME scenario + solver type,
each in its own subdirectory with simulation.h5/meta.json/micro_config.json
(the layout produced by micro_main.py), e.g.:

    scenarios/a3_motorway/micro-sim-runs/default/rm/
        rm_001_.../
        rm_002_.../
        rm_003_.../

Usage
-----
    python bootstrap.py --path scenarios/a3_motorway/micro-sim-runs/default/rm

Output: bootstrap_analysis.html + bootstrap_summary.json written into the
input folder (or --output-dir if given).
"""

from __future__ import annotations

import argparse
import sys

from microscopic.services.bootstrap import (
    InsufficientRunsError,
    RunFolderBootstrapper,
    ScenarioMismatchError,
)
from macroscopic.logger import LOGGER

if __name__ == "__main__":
    # py bootstrap.py --path path/to/folder/of/runs

    parser = argparse.ArgumentParser(description="Bootstrap analysis across a folder of repeated micro-sim runs.")
    parser.add_argument("--path", required=True, metavar="FOLDER", help="Folder of repeated runs (same scenario + solver)")
    parser.add_argument("--output-dir", default=None, metavar="DIR", help="Where to write the dashboard/summary (default: --path)")
    parser.add_argument("--filename", default="bootstrap_analysis.html", help="Dashboard HTML filename")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap resamples (default: 1000)")
    parser.add_argument("--ci", type=float, default=0.95, help="Confidence interval width, e.g. 0.95 (default: 0.95)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible bootstrap resampling")
    args = parser.parse_args()

    bootstrapper = RunFolderBootstrapper(
        folder=args.path,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
        seed=args.seed,
    )

    try:
        html_path = bootstrapper.run(output_dir=args.output_dir, filename=args.filename)
    except (ScenarioMismatchError, InsufficientRunsError, FileNotFoundError) as exc:
        LOGGER.error(str(exc))
        sys.exit(1)

    LOGGER.info(f"Done. Open {html_path} to view the bootstrap dashboard.")
