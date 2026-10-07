"""Figures S7 and S8 of the Online Appendix (the practitioner's guide).

Figure S7 (measured discovery bound): the steps of the computation on three cells of the
trajectory study (unequal wells 4D, broad starts, full assembly, order M R S).
Figure S8 (drift coefficient): the generator-based estimate, its components, and the
residual form on populations of the one-step study.

Reads the arrays of the one-step study (regenerate with `python code/regenerate_arrays.py`)
and the trajectory arrays of unequal wells with broad starts (regenerate with
`python code/trajectory_study.py --run`, or pass their folder with --trajectories); nothing
is written there.  The recomputed bounds are asserted equal to the stored analysis of the
study.  Lettering is at least 9 pt at the printed width of 6 in.  Writes
figures/guide_discovery_bound.pdf, figures/guide_lambda_estimate.pdf and
results/analysis/guide_figures_audit.json.

    python code/guide_figures.py [--trajectories DIR]
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

import discovery_inference as inference

FIGURES, RESULTS = ROOT/"figures", ROOT/"results/analysis"
parser = argparse.ArgumentParser(description="Figures S7 and S8 of the Online Appendix.")
parser.add_argument("--trajectories", type=Path, default=ROOT/"results/trajectories/finite-population-full",
                    help="folder with wells_d4__broad__N{16,64,128}.npz")
WIDTH, MIN_PT = 6.0, 9.0
ASSEMBLY, ORDER, EPSILON = "S1_R1_H1", "MRS", .1
N_COLORS = {16: "#c9b27c", 64: "#246f96", 128: "#1b2f44"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, name):
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_text().strip()]
    assert min(sizes) >= MIN_PT and abs(fig.get_size_inches()[0]-WIDTH) < 1e-9
    path = FIGURES/name
    fig.savefig(path, metadata={"Title": path.stem, "CreationDate": None, "ModDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return sha256(path)


def discovery_bound_figure(audit):
    analysis = np.load(ROOT/"results/trajectories/finite_population_analysis.npz")
    case = list(analysis["cases"]).index("wells_d4__broad")
    eps_a = list(analysis["epsilons"]).index(EPSILON)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.5), layout="constrained")
    generations = np.arange(1, 201)
    record = {}
    for N, color in N_COLORS.items():
        data = np.load(ARGS.trajectories/f"wells_d4__broad__N{N}.npz")
        ai, oi = list(data["assemblies"]).index(ASSEMBLY), list(data["orders"]).index(ORDER)
        ei = list(data["epsilons"]).index(EPSILON)
        hits = data["probe_success_counts"][:, ai, oi, :, ei]              # repetitions x generations
        flags = inference.low_mass_flags(hits)
        counts = flags.sum(axis=0)
        bound = np.array([float(inference.discovery_bound(counts, hits.shape[0], N, m)["upper"]) for m in generations])
        stored = analysis["upper"][case, list(analysis["populations"]).index(N), ai, oi, 0, eps_a, :]
        for endpoint, value in zip(analysis["endpoints"], stored):          # offspring stream of the study
            assert np.isclose(bound[endpoint-1], value), (N, endpoint)
        observed = 1-data["offspring_cumulative_hit"][:, ai, oi, 1:, ei].mean(axis=0)
        if N == 16:
            order = np.argsort(hits[:, -50:].mean(axis=1))
            image = axes[0].imshow(hits[order]/inference.PROBES, aspect="auto", origin="lower", cmap="viridis",
                                   vmin=0, vmax=1, extent=(.5, 200.5, .5, hits.shape[0]+.5), interpolation="nearest")
            fig.colorbar(image, ax=axes[0], location="bottom", pad=.03, shrink=.9, label="probe success fraction")
        axes[1].plot(generations, counts/hits.shape[0], color=color, lw=1.4)
        axes[2].plot(generations, bound, color=color, lw=1.4)
        axes[2].plot(generations, observed, color=color, lw=1.1, ls=":")
        record[f"N={N}"] = {"bound_at_25_50_100_200": [float(bound[m-1]) for m in (25, 50, 100, 200)],
                            "flag_frequency_at_200": float(counts[-1]/hits.shape[0]),
                            "observed_offspring_non_discovery_at_200": float(observed[-1]),
                            "equals_stored_analysis": True}
    axes[0].set(title="(a) Probes, $N=16$", xlabel="Generation $j$", ylabel="Repetition (sorted)")
    axes[1].set(title="(b) Flag frequency", xlabel="Generation $j$", ylim=(-.02, 1.02))
    axes[2].axhline(.05, color="0.4", lw=.8, ls="--")
    axes[2].set(title="(c) Bound by endpoint", xlabel="Endpoint $m$", ylim=(-.02, 1.02))
    for ax in axes[1:]:
        ax.grid(alpha=.18, lw=.5); ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([0], [0], color=c, lw=1.6, label=f"$N={N}$") for N, c in N_COLORS.items()]
    handles += [Line2D([0], [0], color="0.3", lw=1.4, label="measured bound"),
                Line2D([0], [0], color="0.3", lw=1.1, ls=":", label="observed non-discovery"),
                Line2D([0], [0], color="0.4", lw=.8, ls="--", label="level 0.05")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    audit["discovery_bound"] = {"cells": record, "figure_sha256": save(fig, "guide_discovery_bound.pdf")}


def lambda_figure(audit):
    cal = np.load(ROOT/"results/one-step/canonical-calibration/canonical_arrays.npz")
    val = np.load(ROOT/"results/one-step/canonical-validation/canonical_arrays.npz")
    summary = json.loads((ROOT/"results/one-step/canonical-validation/canonical_summary.json").read_text())
    ai = [a["id"] for a in summary["assemblies"]].index(ASSEMBLY)
    families = ("broad", "clustered")
    pooled = lambda arrays, inst, key: np.concatenate([arrays[f"{inst}__{f}_{key}"] for f in families])
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.4), layout="constrained")
    rng = np.random.default_rng(0)                                     # horizontal jitter only
    record = {}
    ax = axes[0]
    for xi, (inst, name) in enumerate((("separable_pl_d4", "PL 4D"), ("wells_d4", "Wells 4D"))):
        c = pooled(cal, inst, "effective_decay_coefficient")[:, ai]
        v = pooled(val, inst, "effective_decay_coefficient")[:, ai]
        lam = c.min(); below = v < lam
        ax.plot(xi-.2+rng.uniform(-.09, .09, len(c)), c, "o", ms=3, color="#1b2f44", alpha=.8)
        ax.plot(xi+.2+rng.uniform(-.09, .09, len(v))[~below], v[~below], "o", ms=2.4, color="#7fb3d5", alpha=.7)
        ax.plot(xi+.2+rng.uniform(-.09, .09, int(below.sum())), v[below], "x", ms=5, mew=1.2, color="#c0392b")
        ax.hlines(lam, xi-.42, xi+.42, color="k", lw=1.2)
        by_family = {f: int((val[f"{inst}__{f}_effective_decay_coefficient"][:, ai] < lam).sum()) for f in families}
        record[inst] = {"lambda_cal": float(lam), "calibration_populations": int(len(c)),
                        "validation_populations": int(len(v)), "validation_below_lambda_cal": int(below.sum()),
                        "validation_below_lambda_cal_by_family": by_family}
    ax.set_xticks([0, 1], ["PL 4D", "Wells 4D"]); ax.set_xlim(-.6, 1.6)
    ax.set(title="(a) Estimate, validation"); ax.set_ylabel(r"$-G[\mu](f)/V(\mu)$", labelpad=1)
    ax = axes[1]
    comp = pooled(cal, "separable_pl_d4", "component_effective_decay_coefficients")[:, ai, :]
    total = pooled(cal, "separable_pl_d4", "effective_decay_coefficient")[:, ai]
    assert np.allclose(comp.sum(axis=1), total)
    for ci in range(4):
        ax.plot(ci+rng.uniform(-.14, .14, len(comp)), comp[:, ci], "o", ms=2.6, color="#246f96", alpha=.75)
        ax.hlines(comp[:, ci].min(), ci-.32, ci+.32, color="k", lw=1.2)
    ax.plot(4+rng.uniform(-.14, .14, len(total)), total, "o", ms=2.6, color="#1b2f44", alpha=.8)
    ax.hlines(total.min(), 3.68, 4.32, color="k", lw=1.2)
    ax.axhline(0, color="0.4", lw=.6)
    ax.set_xticks(range(5), ["D", "H", "S", "R", "sum"])
    ax.set(title="(b) Components, PL 4D")
    assert comp.min(axis=0).sum() <= total.min()
    record["separable_pl_d4"].update(component_minima=comp.min(axis=0).tolist(), sum_of_component_minima=float(comp.min(axis=0).sum()))
    ax = axes[2]
    lam = 1.
    G, V = cal["quartic_d4__clustered_generator_action"][:, ai], cal["quartic_d4__clustered_V"]
    Gv, Vv = val["quartic_d4__clustered_generator_action"][:, ai], val["quartic_d4__clustered_V"]
    c_cal = float(np.maximum(G+lam*V, 0).max())
    exceed = Gv+lam*Vv > c_cal
    ax.plot(Vv[~exceed], Gv[~exceed], "o", ms=2.4, color="#7fb3d5", alpha=.7)
    ax.plot(Vv[exceed], Gv[exceed], "x", ms=5, mew=1.2, color="#c0392b")
    ax.plot(V, G, "o", ms=3, color="#1b2f44", alpha=.85)
    grid = np.linspace(min(V.min(), Vv.min())*.9, max(V.max(), Vv.max())*1.05, 50)
    ax.plot(grid, -lam*grid+c_cal, color="k", lw=1.2)
    ax.set(title="(c) Residual form", xlabel=r"$V(\mu)$"); ax.set_ylabel(r"$G[\mu](f)$", labelpad=1)
    record["quartic_d4__clustered"] = {"lambda_chosen": lam, "c_cal": c_cal, "floor_c_over_lambda": c_cal/lam,
                                       "calibration_populations": int(len(G)), "validation_populations": int(len(Gv)),
                                       "validation_above_line": int(exceed.sum()),
                                       "V_range_calibration": [float(V.min()), float(V.max())],
                                       "all_generator_actions_positive": bool((G > 0).all() and (Gv > 0).all())}
    for ax in axes:
        ax.grid(alpha=.18, lw=.5); ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([0], [0], marker="o", ms=4, lw=0, color="#1b2f44", label="calibration populations"),
               Line2D([0], [0], marker="o", ms=4, lw=0, color="#7fb3d5", label="validation populations"),
               Line2D([0], [0], marker="x", ms=6, mew=1.2, lw=0, color="#c0392b", label="validation violations"),
               Line2D([0], [0], color="k", lw=1.2, label="estimate from calibration")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    audit["drift_coefficient"] = {"cases": record, "figure_sha256": save(fig, "guide_lambda_estimate.pdf")}


def main():
    global ARGS
    ARGS = parser.parse_args()
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 9,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    audit = {"assembly": ASSEMBLY, "order": ORDER, "epsilon": EPSILON}
    discovery_bound_figure(audit)
    lambda_figure(audit)
    audit["script_sha256"] = sha256(__file__)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS/"guide_figures_audit.json").write_text(json.dumps(audit, indent=1)+"\n")
    print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
