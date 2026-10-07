"""Distribution of the estimated dissipation coefficients (appendix figure).

For every instance, population family and assembly of the frozen one-step
study (results/one-step), draw the validation distribution of the total coefficient
-G[mu](f)/V(mu) over the 128 validation populations (5-95% whisker, 25-75% box,
median dot, minimum cross) together with lambda_cal, the smallest coefficient
over the 64 pooled calibration populations of that instance and assembly.
Validation populations below lambda_cal are the drift violations of the
validation table.  Reads the audited arrays only; no new experiment.
Writes figures/dissipation_estimates.pdf and
results/analysis/dissipation_estimates_audit.json.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT/"results/one-step/canonical-calibration"
VAL = ROOT/"results/one-step/canonical-validation"
FIGURE = ROOT/"figures/dissipation_estimates.pdf"
AUDIT = ROOT/"results/analysis/dissipation_estimates_audit.json"
TITLES = {"separable_pl_d4": "PL 4D", "separable_pl_d8": "PL 8D", "rotated_anisotropic_pl_d4": "Rotated PL 4D",
          "rotated_anisotropic_pl_d8": "Rotated PL 8D", "quartic_d4": "Quartic 4D", "quartic_d8": "Quartic 8D",
          "rastrigin_d4": "Rastrigin 4D", "rosenbrock_d4": "Rosenbrock 4D", "wells_d4": "Unequal wells 4D",
          "periodic_d1": "Periodic 1D"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    audit = json.loads((ROOT/"results/one-step/one_step_figure_audit.json").read_text())
    for name, folder in (("calibration", CAL), ("validation", VAL)):
        for file, digest in audit["artifact_hashes"][name].items():
            assert sha256(folder/file) == digest, (name, file)
    summary = json.loads((VAL/"canonical_summary.json").read_text())
    assemblies = [a["id"] for a in summary["assemblies"]]
    labels = ["".join(s for s, on in (("S", a[1] == "1"), ("R", a[4] == "1"), ("H", a[7] == "1")) if on) or "D only"
              for a in assemblies]
    cal, val = np.load(CAL/"canonical_arrays.npz"), np.load(VAL/"canonical_arrays.npz")
    instances = [i["id"] for i in summary["instances"]]
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 5, figsize=(7.4, 4.6), layout="constrained")
    record, violations = {}, 0
    for ax, inst in zip(axes.ravel(), instances):
        pooled = np.concatenate([cal[f"{inst}__{fam}_effective_decay_coefficient"] for fam in ("broad", "clustered")])
        lam_cal = pooled.min(axis=0)                                          # per assembly
        record[inst] = {"lambda_cal": lam_cal.tolist()}
        for fam, dx, color in (("broad", -0.19, "#1f4e79"), ("clustered", 0.19, "#8fb3d9")):
            x = val[f"{inst}__{fam}_effective_decay_coefficient"]            # laws x assemblies
            q = np.quantile(x, [.05, .25, .5, .75, .95], axis=0)
            pos = np.arange(len(assemblies))+dx
            ax.vlines(pos, q[0], q[4], color=color, lw=.8)
            ax.vlines(pos, q[1], q[3], color=color, lw=3)
            ax.plot(pos, q[2], "o", color="white", mec=color, ms=3.2, mew=.9)
            ax.plot(pos, x.min(axis=0), "x", color=color, ms=3.6, mew=.8)
            below = int((x < lam_cal).sum()); violations += below
            record[inst][fam] = {"quantiles_05_25_50_75_95": q.tolist(), "min": x.min(axis=0).tolist(),
                                 "validation_below_lambda_cal": below}
        ax.hlines(lam_cal, np.arange(len(assemblies))-.36, np.arange(len(assemblies))+.36, color="black", lw=1)
        ax.axhline(0, color="0.4", lw=.5)
        lo, hi = ax.get_ylim()
        if lo < 0:                                     # shade the non-dissipative region
            ax.axhspan(lo, 0, color="#f2c6c6", alpha=.45, lw=0, zorder=0)
            ax.set_ylim(lo, hi)
        ax.set_title(TITLES[inst]); ax.set_xticks(range(len(assemblies)), labels, rotation=90, fontsize=6.8)
        ax.grid(axis="y", lw=.3, alpha=.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$-G[\mu](f)/V(\mu)$" + "\n" + r"($>0$: mean gap decreases)", fontsize=7.5)
    handles = [Line2D([], [], color="#1f4e79", lw=3, label="25\u201375% (broad starts)"),
               Line2D([], [], color="#8fb3d9", lw=3, label="25\u201375% (two-cluster starts)"),
               Line2D([], [], color="#1f4e79", lw=.8, label="5\u201395%"),
               Line2D([], [], marker="o", color="white", mec="#1f4e79", ms=4, lw=0, label="median"),
               Line2D([], [], marker="x", color="#1f4e79", ms=4, lw=0, label="minimum over 128 validation populations"),
               Line2D([], [], color="black", lw=1, label=r"$\lambda_{\rm cal}$: minimum over the 64 pooled calibration populations"),
               Patch(facecolor="#f2c6c6", alpha=.45, label="negative: no dissipation at that law")]
    fig.legend(handles=handles, ncol=4, loc="outside lower center", frameon=False, fontsize=7.2, handlelength=1.6, columnspacing=1.2)
    fig.savefig(FIGURE, metadata={"Title": FIGURE.stem, "CreationDate": None, "ModDate": None})
    fig.savefig(FIGURE.with_suffix(".png"), dpi=200)
    AUDIT.parent.mkdir(exist_ok=True)
    AUDIT.write_text(json.dumps({"sources": audit["artifact_hashes"], "figure_sha256": sha256(FIGURE),
                                 "script_sha256": sha256(__file__), "total_validation_below_lambda_cal": violations,
                                 "expected_from_validation_table": 287, "per_instance": record}, indent=1)+"\n")
    print("Wrote", FIGURE, "violations", violations)


if __name__ == "__main__":
    main()
