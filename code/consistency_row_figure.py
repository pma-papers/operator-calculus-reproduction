"""Composition-theorem checks as one row of four panels (single-row layout of the one-step figure).

Same data as one_step_figures.consistency_figure: for each instance
and family the maximum over 128 validation populations, eight assemblies and
both orders of the normalized remainder |D_{tau,o}-G[mu](f)|/(1+V) and of the
normalized order difference |D_{tau,1}-D_{tau,2}|/(1+V), against the step.
Reads the audited one-step arrays, checks the reduction factors against the one-step figure
audit, writes figures/canonical_consistency.pdf and
results/analysis/consistency_row_audit.json.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT/"results/one-step/canonical-validation"
FIGURE = ROOT/"figures/canonical_consistency.pdf"
AUDIT = ROOT/"results/analysis/consistency_row_audit.json"
LABELS = {"separable_pl_d4": "PL 4D", "separable_pl_d8": "PL 8D", "rotated_anisotropic_pl_d4": "Rotated PL 4D",
          "rotated_anisotropic_pl_d8": "Rotated PL 8D", "quartic_d4": "Quartic 4D", "quartic_d8": "Quartic 8D",
          "rastrigin_d4": "Rastrigin 4D", "rosenbrock_d4": "Rosenbrock 4D", "wells_d4": "Unequal wells 4D",
          "periodic_d1": "Periodic 1D"}
FAMILY = {"broad": "broad", "clustered": "two-cluster"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    audit = json.loads((ROOT/"results/one-step/one_step_figure_audit.json").read_text())
    for file, digest in audit["artifact_hashes"]["validation"].items():
        assert sha256(VAL/file) == digest, file
    summary = json.loads((VAL/"canonical_summary.json").read_text())
    arrays = np.load(VAL/"canonical_arrays.npz")
    taus = arrays["taus"]
    reference = {(r["instance"], r["family"]): r for r in audit["refinement_summary"]}
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.5), layout="constrained")
    colors = plt.get_cmap("tab10").colors
    record = []
    for k, (quantity, key, ylabel) in enumerate((("remainder", "normalized_weak_remainder", r"$\max\,|D_{\tau,o}-G[\mu](f)|/(1+V)$"),
                                                 ("order", "normalized_order_quotient", r"$\max\,|D_{\tau,1}-D_{\tau,2}|/(1+V)$"))):
        for c, family in enumerate(("broad", "clustered")):
            ax = axes[2*k+c]
            for j, spec in enumerate(summary["instances"]):
                data = arrays[f"{spec['id']}__{family}_{key}"]
                curve = data.max(axis=(0, 1, 2)) if quantity == "remainder" else data.max(axis=(0, 1))
                ax.loglog(taus, curve, color=colors[j], marker="o", ms=2.2, lw=.9, label=LABELS[spec["id"]])
                ref = reference[(spec["id"], family)]
                factor = float(curve[0]/curve[-1])
                assert abs(factor-ref[f"weak_remainder_reduction" if quantity == "remainder" else "order_quotient_reduction"]) < 1e-9
                record.append({"instance": spec["id"], "family": family, "quantity": quantity, "reduction": factor})
            ax.set_title(("Remainder, " if quantity == "remainder" else "Order diff., ")+FAMILY[family]+" starts")
            ax.set_xlabel(r"step $\tau$"); ax.grid(which="both", alpha=.2)
            if c == 0:
                ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="outside lower center", frameon=False, fontsize=6.5)
    fig.savefig(FIGURE, metadata={"Title": FIGURE.stem, "CreationDate": None, "ModDate": None})
    fig.savefig(FIGURE.with_suffix(".png"), dpi=200)
    AUDIT.write_text(json.dumps({"sources": audit["artifact_hashes"]["validation"], "figure_sha256": sha256(FIGURE),
                                 "script_sha256": sha256(__file__), "reductions_match_one_step_audit": True,
                                 "curves": record}, indent=1)+"\n")
    print("Wrote", FIGURE)


if __name__ == "__main__":
    main()
