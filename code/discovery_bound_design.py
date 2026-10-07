"""Table S11 of the Online Appendix (the practitioner's guide): floor of the measured
discovery bound and the probe count needed to avoid an inadequate-mass flag.

Uses the functions of the frozen inference module code/discovery_inference.py (read only);
writes results/analysis/discovery_bound_design.json.  The floors for 128 repetitions are
asserted equal to the audited values of the batch-size table of the trajectory study.
Runs in about a second.

    python code/discovery_bound_design.py
"""
from pathlib import Path
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
from scipy.stats import binom

import discovery_inference as inference

M, BATCHES, REPETITIONS, PROBES = 200, (1, 4, 16, 32), (64, 128, 256), (64, 128, 256, 512, 1024)


def main():
    floors = {}
    for batch in BATCHES:
        for repetitions in REPETITIONS:
            result = inference.discovery_bound(np.zeros(M, dtype=int), repetitions, batch, M)
            floors[f"N_k={batch},R={repetitions}"] = {"floor": float(result["upper"]),
                                                     "m_rho": int(result["retained_generations"])}
    audited = {1: .196, 4: .057, 16: .030, 32: .024}       # batch-size table of the trajectory study
    for batch, value in audited.items():
        assert round(floors[f"N_k={batch},R=128"]["floor"], 3) == value, batch
    thresholds = {}
    for probes in PROBES:
        lower = inference.cp_lower(np.arange(probes+1), probes, inference.ETA)
        smallest = int(np.argmax(lower >= inference.Q0))
        thresholds[str(probes)] = {"smallest_unflagged_count": smallest, "fraction": smallest/probes,
                                   "probability_no_flag_at_mass": {str(p): float(binom.sf(smallest-1, probes, p))
                                                                    for p in (.3, .35, .4, .45, .5)}}
    out = {"endpoint": M, "alpha": inference.ALPHA, "eta_diag": inference.ETA, "threshold": inference.Q0,
           "floors": floors, "probe_thresholds": thresholds}
    path = HERE.parent/"results/analysis/discovery_bound_design.json"
    path.write_text(json.dumps(out, indent=1)+"\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
