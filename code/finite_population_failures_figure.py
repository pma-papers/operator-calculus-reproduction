"""Compact finite-population failure figure: only the cases with any failure.

Reads the audited analysis arrays of the trajectory study and draws, for each N, the
offspring-only non-discovery frequency by generation 200 for the four
instance/family cases on which any failure occurred (unequal wells and
Rastrigin, broad and clustered), with green dots marking cells whose pointwise
95% upper bound is at most 0.05.  All other sixteen cases have zero observed
failure in every cell; that fact is stated in the caption.  Writes
figures/canonical_finite_population_failures.pdf and an audit JSON.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT/"results/trajectories/finite_population_analysis.npz"
FIGURE = ROOT/"figures/canonical_finite_population_failures.pdf"
AUDIT = ROOT/"results/analysis/finite_population_compact_figure_audit.json"
ROWS = (("wells_d4__broad", "Wells / B"), ("wells_d4__clustered", "Wells / C"),
        ("rastrigin_d4__broad", "Rastrigin / B"), ("rastrigin_d4__clustered", "Rastrigin / C"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    audit = json.loads((ROOT/"results/trajectories/finite_population_audit.json").read_text())
    assert sha256(ANALYSIS) == audit["arrays_sha256"]
    summary = json.loads((ROOT/"results/trajectories/finite-population-full/summary.json").read_text())
    with np.load(ANALYSIS) as a:
        cases, pops = a["cases"].tolist(), a["populations"].tolist()
        failure = a["failure"][:, :, :, :, 0, 0, -1]   # case, N, assembly, order
        upper = a["upper"][:, :, :, :, 0, 0, -1]
    zero_cases = [c for i, c in enumerate(cases) if failure[i].max() == 0]
    assert len(zero_cases) == 16 and all(c not in zero_cases for c, _ in ROWS)
    columns = [a["id"].replace("S", "").replace("_R", "").replace("_H", "")+("a" if o == "MRS" else "b")
               for a in summary["assemblies"] for o in summary["orders"]]
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 1.75), layout="constrained", sharey=True)
    for ax, (ni, N) in zip(axes, enumerate(pops)):
        matrix = np.array([failure[cases.index(c), ni].reshape(16) for c, _ in ROWS])
        certified = np.array([upper[cases.index(c), ni].reshape(16) <= .05 for c, _ in ROWS])
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma_r", aspect="auto")
        y, x = np.nonzero(certified)
        ax.scatter(x, y, s=6, c="#218c36", marker="o")
        ax.set_title(f"$N={int(N)}$")
        ax.set_xticks(range(16), columns, rotation=90, fontsize=5.8)
        ax.set_yticks(range(4), [label for _, label in ROWS], fontsize=6.5)
        ax.set_xticks(np.arange(.5, 16, 2), minor=True)
        ax.grid(which="minor", axis="x", color="white", lw=.3)
    fig.colorbar(im, ax=axes, location="right", shrink=.9, label="Offspring-only failure\nby generation 200", pad=.01)
    fig.supxlabel("Switches SRH; a: S then R, b: R then S.  Green dot: pointwise 95% upper bound at most 0.05", fontsize=6.8)
    fig.savefig(FIGURE, metadata={"Title": FIGURE.stem, "CreationDate": None, "ModDate": None})
    fig.savefig(FIGURE.with_suffix(".png"), dpi=200)
    AUDIT.write_text(json.dumps({"arrays_sha256": sha256(ANALYSIS), "figure_sha256": sha256(FIGURE),
                                 "script_sha256": sha256(__file__), "cases_with_zero_failure": zero_cases,
                                 "rows": [c for c, _ in ROWS]}, indent=2)+"\n")
    print("Wrote", FIGURE)


if __name__ == "__main__":
    main()
