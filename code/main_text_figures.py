"""Main-text results figure: two selected problems across all studies.

Reads only saved results of the frozen one-step, exploration, trajectory and verified-instance experiments and writes
figures/main_results.pdf plus an audit JSON with every plotted number
checked against the existing per-study audits.  No experiment is rerun.
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import one_step_study as model
import dynamic_contrasts as contrasts
import verified_instances as verified


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/"figures/main_results.pdf"
OUTPUT_CERTIFIED = ROOT/"figures/certified_rates.pdf"
AUDIT = ROOT/"results/analysis/main_results_figure_audit.json"
PROBLEMS = {"quartic_d4": "Quartic 4D", "wells_d4": "Unequal wells 4D"}
FAMILIES = {"broad": "broad", "clustered": "clustered"}
COLORS = {("quartic_d4", "broad"): "#246f96", ("quartic_d4", "clustered"): "#7fb3d5",
          ("wells_d4", "broad"): "#c16c23", ("wells_d4", "clustered"): "#e8b48a"}
COMPONENTS = ("drift", "diffusion", "selection", "recombination")
COMPONENT_LABELS = ("D", "H", "S", "R")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label(instance, family):
    return PROBLEMS[instance]+", "+FAMILIES[family]


def consistency(ax, audit):
    """Worst normalized remainder versus step; order differences stay in the appendix figure."""
    folder = ROOT/"results/one-step/canonical-validation"
    summary = json.loads((folder/"canonical_summary.json").read_text())
    reference = json.loads((ROOT/"results/one-step/one_step_figure_audit.json").read_text())["refinement_summary"]
    with np.load(folder/"canonical_arrays.npz") as arrays:
        for instance in PROBLEMS:
            for family in FAMILIES:
                key = instance+"__"+family
                remainder = arrays[key+"_normalized_weak_remainder"].max(axis=(0, 1, 2))
                ax.loglog(model.TAUS, remainder, color=COLORS[instance, family], marker="o", ms=2.5, lw=1.1)
                audit["consistency"][key] = {"worst_remainder_by_tau": remainder.tolist(),
                                             "remainder_reduction": float(remainder[0]/remainder[-1])}
    assert summary["taus"] == model.TAUS.tolist()
    audit["consistency"]["reference_refinement_summary"] = reference
    anchor = max(v["worst_remainder_by_tau"][0] for k, v in audit["consistency"].items() if "__" in k)
    ax.loglog(model.TAUS, anchor*1.6*model.TAUS/model.TAUS[0], color="0.45", ls=":", lw=.9, label=r"slope 1 ($\propto\tau$)")
    ax.set(title="(a) Composition: remainder vs. step", xlabel=r"Macro-step $\tau$")
    ax.set_ylabel(r"$\max\,|D_{\tau,o}-G[\mu](f)|/(1+V)$", fontsize=6.6)
    ax.legend(frameon=False, loc="lower right", fontsize=5.8)


def components(ax, audit):
    with (ROOT/"results/one-step/one_step_component_rates.csv").open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["instance"] in PROBLEMS]
    groups = [(i, f) for i in PROBLEMS for f in FAMILIES]
    width = .19
    for gi, (instance, family) in enumerate(groups):
        for ci, component in enumerate(COMPONENTS):
            row = next(r for r in rows if r["instance"] == instance and r["population_family"] == family
                       and r["component"] == component)
            med, q10, q90 = (float(row[k]) for k in ("median", "q10", "q90"))
            x = ci+(gi-1.5)*width
            ax.bar(x, med, width=width*.92, color=COLORS[instance, family], edgecolor="none")
            ax.plot([x, x], [q10, q90], color="k", lw=.6)
            audit["components"][instance+"__"+family+"__"+component] = {"median": med, "q10": q10, "q90": q90}
    ax.axhline(0, color="k", lw=.6)
    ax.set_yscale("symlog", linthresh=.1, linscale=.6)
    ax.set_xticks(range(4), COMPONENT_LABELS)
    ax.set_yticks([-10, -1, -.1, 0, .1, 1, 10])
    ax.set_ylim(-20, 20)
    ax.set(title="(b) Component coefficients")
    ax.set_ylabel(r"median $-G_j[\mu](f)/V(\mu)$", fontsize=6.6)
    ax.set_xlabel("Component (bars: 10--90% percentiles)", fontsize=6.6)


def dynamics(ax, audit):
    data = ROOT/"results/trajectories/finite-population-full"
    summary = json.loads((data/"summary.json").read_text())
    reference = json.loads((ROOT/"results/trajectories/dynamic_contrast_audit.json").read_text())
    generations = np.arange(201)
    instance = "quartic_d4"
    for fi, family in enumerate(FAMILIES):
        key = instance+"__"+family+"__N32"
        path = data/(key+".npz")
        assert sha256(path) == summary["cases"][key]["output_sha256"], key
        with np.load(path) as saved:
            assemblies, orders, metrics = (saved[k].tolist() for k in ("assemblies", "orders", "metrics"))
            traces = saved["traces"][:, :, orders.index("MRS")].copy()
        mean, low, high, _ = contrasts.paired_summary(traces, assemblies, metrics,
                                                      contrasts.BOOTSTRAP_SEED+10*contrasts.CASES.index(instance)+fi)
        case = reference["cases"][key]
        for aj, component in enumerate(("S", "R")):
            mj = contrasts.METRICS.index("mean_gap")
            for k in contrasts.CHECKPOINTS:
                saved_k = case["omissions"][component]["metrics"]["mean_gap"][str(k)]
                assert saved_k["paired_mean_off_minus_full"] == float(mean[aj, k, mj])
            color = COLORS[instance, family]
            ax.fill_between(generations, low[aj, :, mj], high[aj, :, mj], color=color, alpha=.18, lw=0)
            ax.plot(generations, mean[aj, :, mj], color=color, lw=1.2, ls="-" if component == "S" else "-.")
            audit["dynamics"][key+"__"+component] = {str(k): float(mean[aj, k, mj]) for k in contrasts.CHECKPOINTS}
    ax.axhline(0, color=".25", lw=.6, ls=":")
    ax.set_xscale("function", functions=(np.log1p, np.expm1))
    ax.set_xlim(0, 200)
    ax.set_xticks(contrasts.X_TICKS, [str(k) for k in contrasts.X_TICKS])
    ax.set(title="(c) Dynamic contrasts, quartic", xlabel="Generation (log(1+k) spacing)", ylim=(-1.3, 4.6))
    ax.set_ylabel("Mean gap, omitted $-$ full", fontsize=6.6)
    handles = [Line2D([0], [0], color="0.3", lw=1.2, label="without S"),
               Line2D([0], [0], color="0.3", lw=1.2, ls="-.", label="without R")]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=5.8)


def finite_population(ax, audit):
    """Archive non-discovery versus objective calls (unequal wells, broad start), from budget_curves."""
    curves = np.load(ROOT/"results/analysis/budget_curves.npz")
    reference = json.loads((ROOT/"results/analysis/budget_curves_audit.json").read_text())
    budgets = curves["budgets"]
    n_colors = {16: "#c9b27c", 32: "#7fb3d5", 64: "#246f96", 128: "#1b2f44"}
    for N, color in n_colors.items():
        key = f"wells_d4__broad__N{N}"
        y = curves[key]
        ax.plot(budgets, y, color=color, lw=1.2, label=f"$N={N}$")
        for budget in (8192, 32768):   # the audited table cells lie on the curve
            assert np.isclose(y[np.searchsorted(budgets, budget)], reference["table_cells"][f"{key}__{budget}"])
        audit["finite_population"][key] = {"budgets": budgets.tolist(), "archive_failure": y.tolist()}
    ax.set_xscale("log")
    ax.set(title="(d) Unequal wells vs. budget", xlabel="Objective calls", ylim=(-.03, 1.03))
    ax.set_ylabel(r"Non-discovery ($\varepsilon=0.1$)", fontsize=6.6)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, .62), fontsize=5.8, ncol=1, handlelength=1.4)


def certified(ax_cma, ax_res, audit):
    folder = ROOT/"results/verified-instances/verified-instances-full"
    summary = json.loads((folder/"summary.json").read_text())
    assert summary["status"] == "full" and summary["cma"]["flow_certificate_holds"]
    with np.load(folder/"arrays.npz") as data:
        times, L = data["cma_times"], data["cma_flow_L"]
        ratio = L/L[:, :1]
        for i, curve in enumerate(ratio):
            ax_cma.plot(times, curve, color="#246f96", lw=1.3 if i == 0 else .7, alpha=1 if i == 0 else .55)
        stochastic = data["cma_tau0.05_L"][0]
        steps = np.arange(stochastic.shape[1])*.05
        r = stochastic/stochastic[:, :1]
        ax_cma.plot(steps, np.median(r, axis=0), color="#c16c23", lw=1.)
        ax_cma.fill_between(steps, *np.quantile(r, [.1, .9], axis=0), color="#c16c23", alpha=.15, lw=0)
        ax_cma.plot(times, np.exp(-times/4), color="k", ls="--", lw=1.)
        audit["cma"] = {"max_ratio_to_certificate": float((ratio*np.exp(times/4)).max()),
                        "final_flow_L_over_L0": ratio[:, -1].tolist()}
        assert audit["cma"]["max_ratio_to_certificate"] <= 1+1e-8
        handles = [Line2D([0], [0], color="#246f96", lw=1.3, label="expected flow (5 starts)"),
                   Line2D([0], [0], color="#c16c23", lw=1., label=r"finite step $\tau=0.05$ (median)"),
                   Line2D([0], [0], color="k", ls="--", lw=1., label="certified $e^{-t/4}$")]
        ax_cma.legend(handles=handles, frameon=False, loc="lower left", fontsize=5.8)
        ax_cma.set(yscale="log", title=r"(a) CMA-ES-type flow, periodic $1-\cos y$ ($d=1$)", xlabel="Time $t$", xlim=(0, times[-1]))
        ax_cma.set_ylabel(r"$L(t)/L(0)$,  $L=\Upsilon_m+\Upsilon_c$", fontsize=6.6)
        for instance, color in (("quartic_d4", "#246f96"), ("separable_pl_d4", "#347c63")):
            gaps, envelope = data[f"res_{instance}_gaps"], data[f"res_{instance}_envelope"]
            scale = gaps[:, 0].mean()
            k = np.arange(gaps.shape[1])
            name = "Quartic 4D (box)" if instance == "quartic_d4" else "PL 4D"
            ax_res.plot(k, gaps.mean(axis=0)/scale, color=color, lw=1.3, label=name)
            ax_res.plot(k, envelope.mean(axis=0)/scale, color=color, lw=.9, ls="--")
            audit["res"][instance] = {"rho_RES": summary["res"][instance]["rho_RES"],
                                      "max_mean_over_envelope": summary["res"][instance]["max_mean_over_envelope"],
                                      "certificate_consistent": summary["res"][instance]["certificate_consistent"]}
            assert summary["res"][instance]["certificate_consistent"]
        rates = {i: summary["res"][i]["rho_RES"] for i in ("quartic_d4", "separable_pl_d4")}
        ax_res.plot([], [], color="0.3", ls="--", lw=.9, label="certified envelope")
        ax_res.text(k[-1]*.98, 4e-3, r"envelopes $\approx\rho_{\rm RES}^k$, $\rho_{\rm RES}=%.3f$ (quartic), $%.3f$ (PL)" % (rates["quartic_d4"], rates["separable_pl_d4"]),
                    ha="right", va="top", fontsize=5.6, color="0.25")
        ax_res.legend(frameon=False, loc="lower left", fontsize=5.8)
        ax_res.set(yscale="log", title="(b) Recombinative ES vs. certified envelope", xlabel="Generation $k$", xlim=(0, k[-1]))
        ax_res.set_ylabel(r"Mean gap $\bar{\mathcal{E}}_k/\bar{\mathcal{E}}_0$", fontsize=6.6)


def main():
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.2, "axes.labelsize": 6.8, "legend.fontsize": 6,
                         "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    audit = {"consistency": {}, "components": {}, "dynamics": {}, "finite_population": {}, "cma": {}, "res": {}}
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 1.65), layout="constrained")
    consistency(axes[0], audit)
    components(axes[1], audit)
    dynamics(axes[2], audit)
    finite_population(axes[3], audit)
    for ax in axes:
        ax.grid(alpha=.18, lw=.5)
    handles = [Line2D([0], [0], color=COLORS[i, f], lw=1.6, label=label(i, f)) for i in PROBLEMS for f in FAMILIES]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=6.4)
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, metadata={"Title": "main_results", "CreationDate": None, "ModDate": None})
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 1.55), layout="constrained")
    certified(axes[0], axes[1], audit)
    for ax in axes:
        ax.grid(alpha=.18, lw=.5)
    fig.savefig(OUTPUT_CERTIFIED, metadata={"Title": "certified_rates", "CreationDate": None, "ModDate": None})
    fig.savefig(OUTPUT_CERTIFIED.with_suffix(".png"), dpi=220)
    plt.close(fig)
    audit["sources_sha256"] = {"script": sha256(__file__), "figure": sha256(OUTPUT), "figure_certified": sha256(OUTPUT_CERTIFIED),
                               "one_step_arrays": sha256(ROOT/"results/one-step/canonical-validation/canonical_arrays.npz"),
                               "trajectory_analysis": sha256(ROOT/"results/trajectories/finite_population_analysis.npz"),
                               "verified_instances_arrays": sha256(ROOT/"results/verified-instances/verified-instances-full/arrays.npz")}
    AUDIT.parent.mkdir(exist_ok=True)
    AUDIT.write_text(json.dumps(audit, indent=2)+"\n")
    print("Wrote", OUTPUT, OUTPUT_CERTIFIED, "and", AUDIT)


if __name__ == "__main__":
    main()
