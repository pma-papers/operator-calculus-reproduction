"""Reference-run estimate of the state-law approximation modulus (illustration).

For the frozen trajectory study, compute at generation 200 the exact squared
2-Wasserstein distance between the empirical population of size N and the
population of size N_ref = 128 started from the same seed (nested starts), for
each run, assembly and order; report seed averages with bootstrap intervals.
Because 128 is a multiple of N, each N-point is replicated 128/N times and the
distance is an assignment problem, which is exact for uniform weights.
Also reports the i.i.d. baseline: W_2^2 between N i.i.d. draws from the N_ref
population and that population, i.e. the empirical-sampling term alone.
Writes results/analysis/reference_run_modulus.json.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"results/trajectories/finite-population-full"
OUT = ROOT/"results/analysis/reference_run_modulus.json"
CASES = ("separable_pl_d4__broad", "quartic_d4__broad", "wells_d4__broad")
N_REF = 128
SIZES = (16, 32, 64)
ASSEMBLY, ORDER = "S1_R1_H1", "MRS"
BOOTSTRAP, SEED = 2000, 4172600001


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def w2_squared(points, reference):
    """Exact W_2^2 between uniform empirical measures; len(reference) is a multiple of len(points)."""
    rep = len(reference)//len(points)
    cost = cdist(np.repeat(points, rep, axis=0), reference, metric="sqeuclidean")
    rows, cols = linear_sum_assignment(cost)
    return cost[rows, cols].mean()


def main():
    summary = json.loads((DATA/"summary.json").read_text())
    ai = [a["id"] for a in summary["assemblies"]].index(ASSEMBLY)
    oi = summary["orders"].index(ORDER)
    rng = np.random.default_rng(SEED)
    out = {"generation": 200, "assembly": ASSEMBLY, "order": ORDER, "N_ref": N_REF, "cases": {}}
    for case in CASES:
        ref_key = f"{case}__N{N_REF}"
        assert sha256(DATA/(ref_key+".npz")) == summary["cases"][ref_key]["output_sha256"]
        with np.load(DATA/(ref_key+".npz")) as d:
            ref = d["final_points"][:, ai, oi]          # runs, N_ref, d
        result = {}
        for N in SIZES:
            key = f"{case}__N{N}"
            assert sha256(DATA/(key+".npz")) == summary["cases"][key]["output_sha256"]
            with np.load(DATA/(key+".npz")) as d:
                pop = d["final_points"][:, ai, oi]       # runs, N, d
            paired = np.array([w2_squared(pop[r], ref[r]) for r in range(len(pop))])
            iid = np.array([w2_squared(ref[r][rng.choice(N_REF, N, replace=False)], ref[r]) for r in range(len(pop))])
            boot = np.array([paired[rng.integers(0, len(paired), len(paired))].mean() for _ in range(BOOTSTRAP)])
            result[str(N)] = {"mean_W2sq_vs_reference": float(paired.mean()),
                              "bootstrap_95pct": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                              "median": float(np.median(paired)),
                              "iid_subsample_baseline_mean": float(iid.mean()), "runs": int(len(paired))}
        means = np.array([result[str(N)]["mean_W2sq_vs_reference"] for N in SIZES])
        slope = np.polyfit(np.log(SIZES), np.log(means), 1)[0]
        result["fitted_decay_exponent_16_to_64"] = float(-slope)
        out["cases"][case] = result
        print(case, {N: round(result[str(N)]["mean_W2sq_vs_reference"], 4) for N in SIZES}, "exponent", round(-slope, 2))
    out["sources_sha256"] = {"script": sha256(__file__), "summary": sha256(DATA/"summary.json")}
    OUT.write_text(json.dumps(out, indent=2)+"\n")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
