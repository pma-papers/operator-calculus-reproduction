"""Extract the trajectory subset behind the dynamic-contrast figures from a complete run.

reproduce_paper_figures.py reads results/trajectories/dynamic_traces_N32.npz, which is
not shipped (16 MB).  This script builds it from the raw output of
trajectory_study.py --run (about 13 minutes): the traces of all 128 runs, eight
assemblies, 201 generations and four metrics, order S then R, population N=32, for
the unequal-wells and quartic 4D instances with both initial families, copied
without modification from the raw files.  The SHA-256 of every raw file is checked
against the run summary and recorded in the output.  The output is created
exclusively; move an existing file aside first.

    python code/extract_dynamic_traces.py [--input results/trajectories/finite-population-full]
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/"results/trajectories/dynamic_traces_N32.npz"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT/"results/trajectories/finite-population-full")
    folder = parser.parse_args().input
    summary = json.loads((folder/"summary.json").read_text())
    assert summary["status"] == "full" and summary["generations"] == 200 and summary["runs_per_cell"] == 128
    arrays, provenance = {}, {}
    for instance in ("wells_d4", "quartic_d4"):
        for family in ("broad", "clustered"):
            key = instance+"__"+family+"__N32"
            path = folder/(key+".npz")
            assert sha256(path) == summary["cases"][key]["output_sha256"], "raw file differs from the summary: "+key
            with np.load(path) as saved:
                orders = saved["orders"].tolist()
                arrays[key] = saved["traces"][:, :, orders.index("MRS")].copy()
                assert arrays[key].shape == (128, 8, 201, 4), key
                assemblies, metrics = saved["assemblies"], saved["metrics"]
            provenance[key] = summary["cases"][key]["output_sha256"]
    with OUTPUT.open("xb") as stream:
        np.savez_compressed(stream, assemblies=assemblies, metrics=metrics,
                            source_raw_sha256=json.dumps(provenance), **arrays)
    print("Wrote", OUTPUT, "from four hash-checked raw files")


if __name__ == "__main__":
    main()
