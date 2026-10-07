"""Archive non-discovery as a function of objective calls (derived analysis).

Reads the frozen finite-population trajectories and, for every
instance/family/N, computes the fraction of runs (over all eight assemblies,
both orders, 128 runs) whose archive still has no epsilon-good point when the
operational objective-call count reaches each budget on a log grid.  Budget
measurement follows the trajectory study's rule: the last fully completed generation within
the budget, counting all operational calls.  The four audited budgets are checked
against results/trajectories/finite_population_analysis.npz.  Writes
results/analysis/budget_curves.npz, an audit, and the appendix figure.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"results/trajectories/finite-population-full"
OUT = ROOT/"results/analysis/budget_curves.npz"
AUDIT = ROOT/"results/analysis/budget_curves_audit.json"
FIGURE = ROOT/"figures/canonical_budget_curves.pdf"
TABLE_BUDGETS = (8192, 32768)
BUDGETS = np.union1d(np.round(np.logspace(np.log10(128), np.log10(240000), 80)).astype(int), TABLE_BUDGETS)
NAMES = {"separable_pl": "PL", "rotated_anisotropic_pl": "Rot. PL", "quartic": "Quartic",
         "rastrigin": "Rastrigin", "rosenbrock": "Rosenbrock", "wells": "Unequal wells", "periodic": "Periodic"}
COLORS = {16: "#c9b27c", 32: "#7fb3d5", 64: "#246f96", 128: "#1b2f44"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def failure_curves(calls, record, eps):
    """calls, record: (runs, assemblies, orders, generations+1). Returns failure per budget."""
    out = np.empty(len(BUDGETS))
    for bi, budget in enumerate(BUDGETS):
        last = np.sum(calls <= budget, axis=-1)-1
        reached = last >= 0
        rec = np.take_along_axis(record, np.maximum(last, 0)[..., None], axis=-1)[..., 0]
        failed = np.where(reached, rec > eps, True)   # before the first completed generation, nothing is archived
        out[bi] = failed.mean()
    return out


def main():
    summary = json.loads((DATA/"summary.json").read_text())
    assert summary["status"] == "full"
    with np.load(ROOT/"results/trajectories/finite_population_analysis.npz") as a:
        analysis_cases, analysis_pops, analysis_budgets = a["cases"].tolist(), a["populations"].tolist(), a["budgets"].tolist()
        analysis_budget_failure = a["archive_budget_failure"]   # case, N, assembly, order, eps, budget
    eps_index = summary["epsilons"].index(.1)
    curves, table, checked = {}, {}, 0
    for spec in summary["instances"]:
        for family in summary["families"]:
            case = spec["id"]+"__"+family
            for N in summary["populations"]:
                key = f"{case}__N{N}"
                path = DATA/(key+".npz")
                assert sha256(path) == summary["cases"][key]["output_sha256"], key
                with np.load(path) as d:
                    calls = d["evolution_cumulative_calls_by_category"].sum(axis=-1)
                    record = d["traces"][..., 0]
                    assert summary["metrics"][0] == "record_gap"
                curve = failure_curves(calls, record, .1)
                curves[key] = curve
                # Cross-check against the audited trajectory-study budget columns.
                ci, ni = analysis_cases.index(case), analysis_pops.index(N)
                for bj, budget in enumerate(analysis_budgets):
                    last = np.sum(calls <= budget, axis=-1)-1
                    rec = np.take_along_axis(record, last[..., None], axis=-1)[..., 0]
                    assert np.isclose(np.mean(rec > .1), analysis_budget_failure[ci, ni, :, :, eps_index, bj].mean()), (key, budget)
                    checked += 1
                for budget in TABLE_BUDGETS:
                    last = np.sum(calls <= budget, axis=-1)-1
                    rec = np.take_along_axis(record, last[..., None], axis=-1)[..., 0]
                    table[f"{key}__{budget}"] = float(np.mean(rec > .1))
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, budgets=BUDGETS, **curves)
    by_N = {str(N): {str(b): float(np.mean([table[f"{s['id']}__{f}__N{N}__{b}"] for s in summary["instances"] for f in summary["families"]]))
                     for b in TABLE_BUDGETS} for N in summary["populations"]}
    # Appendix figure: one panel per instance, both families, four N.
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6,
                         "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 5, figsize=(7.6, 3.4), layout="constrained", sharex=True, sharey=True)
    for ax, spec in zip(axes.flat, summary["instances"]):
        for family, ls in (("broad", "-"), ("clustered", "--")):
            for N in summary["populations"]:
                ax.plot(BUDGETS, curves[f"{spec['id']}__{family}__N{N}"], color=COLORS[N], ls=ls, lw=1.1)
        ax.set_xscale("log")
        ax.set_title(f"{NAMES[spec['family']]} {spec['dimension']}D")
        ax.grid(alpha=.18, lw=.5)
    for ax in axes[1]:
        ax.set_xlabel("Objective calls")
    fig.supylabel(r"Archive non-discovery ($\varepsilon=0.1$)", fontsize=7)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=COLORS[N], lw=1.3, label=f"$N={N}$") for N in summary["populations"]]
    handles += [Line2D([0], [0], color="0.3", ls="-", lw=1.1, label="broad start"), Line2D([0], [0], color="0.3", ls="--", lw=1.1, label="clustered start")]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, frameon=False)
    fig.savefig(FIGURE, metadata={"Title": FIGURE.stem, "CreationDate": None, "ModDate": None})
    fig.savefig(FIGURE.with_suffix(".png"), dpi=200)
    plt.close(fig)
    audit = {"source_summary_sha256": sha256(DATA/"summary.json"), "script_sha256": sha256(__file__),
             "curves_sha256": sha256(OUT), "figure_sha256": sha256(FIGURE),
             "trajectory_budget_columns_cross_checked": checked, "budget_rule": "last fully completed generation within the budget; all operational calls",
             "table_failure_by_N_and_budget": by_N, "table_cells": table}
    AUDIT.write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(by_N, indent=1))
    print("PASS:", checked, "audited budget columns reproduced; wrote", OUT, FIGURE)


if __name__ == "__main__":
    main()
