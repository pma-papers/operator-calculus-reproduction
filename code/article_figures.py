"""Figures 1-3 of the article, drawn at the text width of 6 in with all lettering at least 9 pt.

The figures are drawn at their final size (included at natural width), so the font sizes
set here are the printed sizes.  They reuse the plotting functions of the study scripts
(schematic_figures.py, main_text_figures.py, verified_instances.py) and read the same
arrays: the one-step and verified-instances arrays (`python code/regenerate_arrays.py`) and
the N=32 trajectory files (`python code/trajectory_study.py --run`).  The plotted numbers
are asserted equal to the audits of the study figures.  Writes
figures/article_landscape_operators.pdf, article_main_results.pdf,
article_verified_instances.pdf and results/analysis/article_figures_audit.json.

    python code/article_figures.py
"""
from pathlib import Path
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

import schematic_figures as schematic
import main_text_figures as results
import verified_instances as experiment

FIGURES, RESULTS = ROOT/"figures", ROOT/"results/analysis"
WIDTH, MIN_PT = 6.0, 9.0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def style():
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 9,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})


def enforce_min_font(fig):
    """Raise every text object of the figure to at least MIN_PT and return the smallest size left."""
    sizes = []
    for text in fig.findobj(Text):
        if text.get_text().strip():
            if text.get_fontsize() < MIN_PT:
                text.set_fontsize(MIN_PT)
            sizes.append(text.get_fontsize())
    return min(sizes)


def save(fig, name):
    smallest = enforce_min_font(fig)
    assert smallest >= MIN_PT and abs(fig.get_size_inches()[0]-WIDTH) < 1e-9
    path = FIGURES/name
    fig.savefig(path, metadata={"Title": path.stem, "CreationDate": None, "ModDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print("Wrote", path)
    return {"sha256": sha256(path), "smallest_font_pt": smallest}


def landscape_operators():
    """Figure 1: (a) landscape geometry, (b)-(d) one step of each canonical operator (schematic, no data)."""
    s = schematic
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.5), layout="constrained")
    ax = axes[0, 0]
    fy = s.f(s.X)
    ax.axvspan(s.Y_STAR-s.R0, s.Y_STAR+s.R0, color="#246f96", alpha=.09, lw=0)
    eps_set = s.X[fy <= s.EPS]
    ax.axvspan(eps_set.min(), eps_set.max(), color="#347c63", alpha=.25, lw=0)
    ax.plot(s.X, fy, color="k", lw=1.2)
    inside = np.abs(s.X-s.Y_STAR) <= s.R0
    ax.plot(s.X[inside], (s.ETA*np.abs(s.X[inside]-s.Y_STAR))**(1/s.NU), color="#c16c23", ls="--", lw=1.)
    ax.axhline(s.F_INF, color="#8a2b2b", ls=":", lw=.9)
    ax.axhline(s.EPS, color="#347c63", ls=":", lw=.9)
    ax.plot([s.Y_STAR], [0], "o", color="k", ms=3.5)
    ax.annotate("", xy=(s.Y_STAR+s.R0, .05), xytext=(s.Y_STAR, .05), arrowprops=dict(arrowstyle="->", lw=.8, color="#246f96"))
    ax.text(s.Y_STAR+s.R0*.62, .1, "$R_0$", color="#246f96", ha="center")
    ax.text(4.9, s.F_INF+.03, r"$f_\infty$", color="#8a2b2b", ha="right")
    ax.text(4.9, s.EPS+.03, r"$f_*+\varepsilon$", color="#347c63", ha="right")
    ax.text(-4.9, 1.12, r"$(\eta\|y-y_*\|)^{1/\nu}$", color="#c16c23", ha="left")
    ax.text(s.Y_STAR-.15, -.07, "$y_*$", ha="right", va="top")
    ax.text(.6, 1.12, r"$\mathcal{S}_*^{\varepsilon}$", color="#347c63", ha="left")
    ax.set(title="(a) Landscape and basin geometry", xlim=(-5, 5), ylim=(-.2, 1.3), yticks=[0, .5, 1])
    ax.set_ylabel("$f(y)-f_*$"); ax.set_xlabel("$y$", labelpad=1)
    rho = s.density(s.X)
    panels = (("(b) Mutation (drift, diffusion)", s.mutation(rho), "#246f96"),
              ("(c) Selection (reweighting)", s.selection(rho), "#8a2b2b"),
              ("(d) Recombination (midpoint)", s.recombination(rho), "#6a4c93"))
    for ax, (title, after, color) in zip((axes[0, 1], axes[1, 0], axes[1, 1]), panels):
        ax.fill_between(s.X, 0, rho, color="0.55", alpha=.35, lw=0)
        ax.plot(s.X, after, color=color, lw=1.4)
        ax.plot(s.X, fy*.62, color="k", lw=.6, alpha=.4)
        ax.axvline(s.Y_STAR, color="k", lw=.6, ls=":", alpha=.6)
        ax.set(title=title, xlim=(-5, 5), ylim=(0, .68), yticks=[])
        ax.set_xlabel("$y$", labelpad=1); ax.set_ylabel("density")
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.55", alpha=.35, lw=0, label=r"before: $\bar\mu$"),
               Line2D([0], [0], color="0.25", lw=1.4, label=r"after one $\tau$-step (colored)"),
               Line2D([0], [0], color="k", lw=.6, alpha=.6, label="objective $f$ (rescaled)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    return save(fig, "article_landscape_operators.pdf")


def main_results():
    """Figure 2: the four panels of the study's main-results figure in a 2 x 2 layout."""
    audit = {"consistency": {}, "components": {}, "dynamics": {}, "finite_population": {}}
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.4), layout="constrained")
    results.consistency(axes[0, 0], audit)
    results.components(axes[0, 1], audit)
    results.dynamics(axes[1, 0], audit)
    results.finite_population(axes[1, 1], audit)
    axes[0, 0].set_title("(a) Composition: remainder vs. step")
    axes[0, 1].set_xlabel("Component (bars: 10–90% range)")
    axes[1, 0].set_xlabel("Generation $k$ (spaced as $\\log(1+k)$)")
    axes[1, 1].legend(frameon=False, loc="upper right", handlelength=1.4)
    axes[1, 0].legend(handles=axes[1, 0].get_legend().legend_handles, frameon=False, loc="upper right")
    for ax in axes.ravel():
        ax.grid(alpha=.18, lw=.5)
    handles = [Line2D([0], [0], color=results.COLORS[i, f], lw=1.8, label=results.label(i, f))
               for i in results.PROBLEMS for f in results.FAMILIES]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, title="Panels (a)–(c)")
    reference = json.loads((ROOT/"results/analysis/main_results_figure_audit.json").read_text())
    for key in audit:                      # identical plotted numbers as the study figure
        assert json.loads(json.dumps(audit[key])) == reference[key], key
    out = save(fig, "article_main_results.pdf")
    out["plotted_numbers_equal_study_audit"] = True
    return out


def verified_instances():
    """Figure 3: the three verified instances; same arrays and curves as verified_instances_figures.py."""
    folder = ROOT/"results/verified-instances/verified-instances-full"
    summary = json.loads((folder/"summary.json").read_text())
    reference = json.loads((ROOT/"results/verified-instances/verified_instances_figure_audit.json").read_text())
    assert summary["status"] == "full" and sha256(folder/"arrays.npz") == reference["arrays_sha256"]
    res_labels = {"separable_pl_d4": "PL 4D", "rotated_anisotropic_pl_d4": "Rotated PL 4D", "quartic_d4": "Quartic 4D (box)"}
    res_colors = {"separable_pl_d4": "#246f96", "rotated_anisotropic_pl_d4": "#c16c23", "quartic_d4": "#347c63"}
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.4), layout="constrained")
    below = dict(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.24), handlelength=1.8, borderaxespad=0)
    with np.load(folder/"arrays.npz") as data:
        ax, cbo = axes[0], data["cbo"]
        t = np.arange(cbo.shape[1])*experiment.CBO["dt"]
        for j, (label, scale, color, ls) in enumerate((("$V_*$", 1, "#222222", "-"), (r"$2W_\alpha$", 2, "#246f96", "--"),
                                                      (r"$2\|v_\alpha-y_*\|^2$", 2, "#c16c23", "-."))):
            values = scale*cbo[:, :, j]
            ax.plot(t, np.median(values, axis=0), color=color, ls=ls, lw=1.3, label=label)
            if j == 0:
                ax.fill_between(t, *np.quantile(values, [.1, .9], axis=0), color=color, alpha=.12, lw=0)
        rate = summary["cbo"]["centered_coefficient"]
        ax.plot(t, np.median(cbo[:, 0, 0])*np.exp(-rate*t), color="0.5", ls=":", lw=1., label=r"slope $2\lambda-d\sigma^2$")
        ax.set(yscale="log", xlabel="Time $t$", title="(a) CBO, fixed $\\alpha$", xlim=(0, t[-1]), ylim=(1e-6, 60))
        ax.legend(**below)
        ax, times = axes[1], data["cma_times"]
        flow = data["cma_flow_L"]/data["cma_flow_L"][:, :1]
        for i, curve in enumerate(flow):
            ax.plot(times, curve, color="#246f96", lw=1.4 if i == 0 else .8, alpha=1 if i == 0 else .6,
                    label="Expected flow" if i == 0 else None)
        stochastic = data["cma_tau0.05_L"][0]
        steps = np.arange(stochastic.shape[1])*.05
        ratio = stochastic/stochastic[:, :1]
        ax.plot(steps, np.median(ratio, axis=0), color="#c16c23", lw=1.1, label=r"Finite step $\tau=0.05$")
        ax.fill_between(steps, *np.quantile(ratio, [.1, .9], axis=0), color="#c16c23", alpha=.15, lw=0)
        ax.plot(times, np.exp(-times/4), color="k", ls="--", lw=1.1, label="Certified $e^{-t/4}$")
        ax.set(yscale="log", xlabel="Time $t$", title="(b) CMA-ES-type flow", xlim=(0, times[-1]))
        ax.set_ylabel(r"$L(t)/L(0)$", labelpad=1)
        ax.legend(**below)
        ax = axes[2]
        for name in experiment.RES_INSTANCES:
            gaps, envelope = data[f"res_{name}_gaps"], data[f"res_{name}_envelope"]
            scale = gaps[:, 0].mean()
            k = np.arange(gaps.shape[1])
            ax.plot(k, gaps.mean(axis=0)/scale, color=res_colors[name], lw=1.4, label=res_labels[name])
            ax.plot(k, envelope.mean(axis=0)/scale, color=res_colors[name], lw=1., ls="--")
        ax.plot([], [], color="0.3", ls="--", lw=1., label="Certified envelope")
        ax.set(yscale="log", xlabel="Generation $k$", title="(c) Recombinative ES", xlim=(0, k[-1]))
        ax.set_ylabel(r"Mean gap $\bar{\mathcal{E}}_k/\bar{\mathcal{E}}_0$", labelpad=1)
        ax.legend(**below)
    for ax in axes:
        ax.grid(alpha=.18, lw=.5)
    out = save(fig, "article_verified_instances.pdf")
    out["arrays_sha256_equal_study_audit"] = True
    return out


def main():
    style()
    audit = {"requirement": "all lettering at least 9 pt at the printed width of 6 in",
             "article_landscape_operators": landscape_operators(),
             "article_main_results": main_results(),
             "article_verified_instances": verified_instances(),
             "script_sha256": sha256(__file__)}
    (RESULTS/"article_figures_audit.json").write_text(json.dumps(audit, indent=1)+"\n")
    print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
