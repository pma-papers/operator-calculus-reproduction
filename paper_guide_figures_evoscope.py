"""The practitioner's-guide figures of the paper (Figures S7-S8), drawn with EvoScope from the
arrays of the paper's study.

Needs EvoScope on the Python path and the regenerated arrays of this package: the one-step arrays
(`python code/regenerate_arrays.py`, seconds) and the trajectory files of unequal wells
(`python code/trajectory_study.py --run`, about 13 minutes).  The bounds are recomputed
from the raw probe counts and the drift coefficients from the stored population points,
both with EvoScope; the script checks them against the study's own analysis.

    PYTHONPATH=<evoscope>/src python paper_guide_figures_evoscope.py [--one-step DIR] [--trajectories DIR]

Writes paper_guide_figures.pdf next to this script.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/"evoscope/src"))
from evoscope import attribution, discovery, problems, viz             # noqa: E402
from evoscope.operators import paper_assembly                          # noqa: E402
from evoscope.population import Population                             # noqa: E402

PAPER = HERE/"results"
parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--one-step", type=Path, default=PAPER/"one-step")
parser.add_argument("--trajectories", type=Path, default=PAPER/"trajectories")
args = parser.parse_args()

viz.paper_style()
fig, axes = plt.subplots(2, 3, figsize=(9, 6.2), layout="constrained")

# ---- Figure S7: measured discovery bound on unequal wells, broad starts, full assembly, order M R S
analysis = np.load(args.trajectories/"finite_population_analysis.npz")
case = list(analysis["cases"]).index("wells_d4__broad")
for N, color in ((16, "#c9b27c"), (64, "#246f96"), (128, "#1b2f44")):
    data = np.load(args.trajectories/f"finite-population-full/wells_d4__broad__N{N}.npz")
    ai, oi = list(data["assemblies"]).index("S1_R1_H1"), list(data["orders"]).index("MRS")
    ei = list(data["epsilons"]).index(.1)
    counts = data["probe_success_counts"][:, ai, oi, :, ei]
    stored = analysis["upper"][case, list(analysis["populations"]).index(N), ai, oi, 0, list(analysis["epsilons"]).index(.1)]
    ours = [float(discovery.measured_bound(counts, 128, N, endpoint=int(m))["upper"]) for m in analysis["endpoints"]]
    assert np.allclose(ours, stored, rtol=0, atol=1e-15), (N, ours, stored)
    observed = 1-data["offspring_cumulative_hit"][:, ai, oi, 1:, ei].mean(axis=0)
    viz.plot_discovery(counts, 128, N, axes=[axes[0, 0] if N == 16 else None, axes[0, 1], axes[0, 2]],
                       observed_non_discovery=observed, label=f"N = {N}", color=color, heatmap=N == 16)
    print(f"N = {N:3d}: bound at m = 200: {ours[-1]:.3f} (matches the study's analysis)")
axes[0, 0].set_title("(a) probes, N = 16")
axes[0, 1].set_title("(b) flagged fraction")
axes[0, 2].set_title("(c) bound by endpoint")
axes[0, 1].legend(frameon=False)

# ---- Figure S8: drift-coefficient estimate, components, residual form
cal = np.load(args.one_step/"canonical-calibration/canonical_arrays.npz")
val = np.load(args.one_step/"canonical-validation/canonical_arrays.npz")
populations = lambda arrays, pid, fams: [Population.uniform(x) for f in fams for x in arrays[f"{pid}__{f}_input_points"]]  # noqa: E731
groups = {}
for pid, label in (("separable_pl_d4", "PL 4D"), ("wells_d4", "Wells 4D")):
    problem = problems.get(pid)
    assembly = paper_assembly(problem)
    c = attribution.coefficients(assembly, populations(cal, pid, ("broad", "clustered")), problem)
    v = attribution.coefficients(assembly, populations(val, pid, ("broad", "clustered")), problem)
    groups[label] = (c["total"], v["total"])
    if pid == "separable_pl_d4":
        viz.plot_component_coefficients(c, ax=axes[1, 1])
    lam = attribution.estimate_drift_coefficient(c["total"])
    print(f"{label}: lambda_cal = {lam:.3f}, validation below: {attribution.validate(lam, v['total'], 64)['violations']}/256")
viz.plot_drift_estimates(groups, ax=axes[1, 0])
problem = problems.get("quartic_d4")
assembly = paper_assembly(problem)
c = attribution.coefficients(assembly, populations(cal, "quartic_d4", ("clustered",)), problem)
v = attribution.coefficients(assembly, populations(val, "quartic_d4", ("clustered",)), problem)
const = attribution.residual_constant(c["generator"], c["V"], 1.)
viz.plot_residual_form(c["generator"], c["V"], 1., const, v["generator"], v["V"], ax=axes[1, 2])
print(f"quartic, two-cluster: residual constant c = {const:.3f} at lambda = 1")
axes[1, 0].set_title("(a) estimate, validation")
axes[1, 1].set_title("(b) components, PL 4D")
axes[1, 2].set_title("(c) residual form")
out = HERE/"paper_guide_figures.pdf"
fig.savefig(out)
print("wrote", out)
