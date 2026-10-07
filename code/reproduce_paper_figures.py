"""Regenerate every data figure and the finite-population table of the manuscript.

This is the reviewer entry point.  It reads only the compact saved results that
ship with the supplementary material and calls the original, unmodified
plotting functions of the experiment scripts.  The manuscript's reader-facing
terminology (applied by relabel_trajectory_figures.py and relabel_dynamic_contrasts.py) is reproduced by
the same string substitutions; no plotted number depends on it.

Each regenerated figure is checked against the plotted-data digest recorded
when the manuscript figure was produced (positions, limits, and every line,
image, patch and collection array; text is excluded from the digest).  The
finite-population table is recomputed from the configuration-level arrays and
compared with the saved audit.  Nothing in results/ is written.

    python code/reproduce_paper_figures.py [--output DIR]
"""
from pathlib import Path
from contextlib import contextmanager
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

import one_step_figures as assembly_figures
import exploration_figures
import trajectory_figures as finite_figures
import dynamic_contrasts as contrasts
import relabel_trajectory_figures as population_labels
import relabel_dynamic_contrasts as dynamic_labels


ROOT = Path(__file__).resolve().parents[1]
ONE_STEP = ROOT/"results/one-step/canonical-validation"
TRAJECTORIES = ROOT/"results/trajectories"
LABELS = ROOT/"results/figure-labels"
TRACES = TRAJECTORIES/"dynamic_traces_N32.npz"
# Original script output stem -> file name used in the manuscript.
NAMES = {**population_labels.NAMES, **dynamic_labels.NAMES,
         "trajectory_failures": "trajectory_failures",
         "trajectory_discovery_bounds": "trajectory_discovery_bounds"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def routing(output, snapshots, digest):
    """Apply manuscript labels and redirect saves from figures/ to `output`."""
    original = Figure.text, Axes.set_title, Axes.set_ylabel, Figure.savefig

    def text(self, x, y, s, *args, **kwargs):
        return original[0](self, x, y, population_labels.changed_text(s), *args, **kwargs)

    def set_title(self, label, *args, **kwargs):
        if label in dynamic_labels.TITLES:
            label = dynamic_labels.TITLES[label]
            kwargs = {**kwargs, "fontsize": 8.5, "linespacing": 1.15}
        return original[1](self, label, *args, **kwargs)

    def set_ylabel(self, label, *args, **kwargs):
        return original[2](self, dynamic_labels.YLABELS.get(label, label), *args, **kwargs)

    def save(self, fname, *args, **kwargs):
        old = Path(fname)
        if old.stem not in NAMES:
            raise ValueError("Unexpected figure output: "+str(old))
        new = output/(NAMES[old.stem]+old.suffix)
        if old.suffix == ".pdf":
            self.canvas.draw()
            snapshots[NAMES[old.stem]] = digest(self)
        kwargs = {**kwargs, "metadata": {**kwargs.get("metadata", {}), "Title": NAMES[old.stem]}}
        original[3](self, new, *args, **kwargs)

    Figure.text, Axes.set_title, Axes.set_ylabel, Figure.savefig = text, set_title, set_ylabel, save
    try:
        yield
    finally:
        Figure.text, Axes.set_title, Axes.set_ylabel, Figure.savefig = original


@contextmanager
def redirected_hashes(module, attribute):
    """The original plotters hash the files they just wrote under figures/.
    Those files are now written to `output`, so skip hashes of the old paths."""
    original = getattr(module, attribute)
    figures = (ROOT/"figures").resolve()

    def tolerant(path):
        return None if Path(path).resolve().parent == figures else original(path)

    setattr(module, attribute, tolerant)
    try:
        yield
    finally:
        setattr(module, attribute, original)


def population_figures(output):
    """Figures one_step_consistency_relabeled, one_step_components_relabeled, exploration_noise_relabeled and exploration_components_relabeled."""
    snapshots = {}
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    summary = json.loads((ONE_STEP/"canonical_summary.json").read_text())
    with routing(output, snapshots, population_labels.numerical_snapshot):
        with np.load(ONE_STEP/"canonical_arrays.npz") as arrays:
            assembly_figures.consistency_figure(summary, arrays)
            assembly_figures.coefficient_figure(summary, arrays)
        with redirected_hashes(exploration_figures.experiment.model, "sha256"):
            exploration_figures.figures(population_labels.primary_rows())
    recorded = json.loads((LABELS/"population_figure_labels.json").read_text())["revised_label_pass"]
    return {new: snapshots[new] == recorded[old]["numerical_sha256"]
            for old, new in population_labels.NAMES.items()}


def dynamic_figures(output):
    """Figures canonical_dynamic_contrasts_wells and canonical_dynamic_contrasts_quartic."""
    audit = json.loads((TRAJECTORIES/"dynamic_contrast_audit.json").read_text())
    summaries = {}
    with np.load(TRACES) as saved:
        assemblies, metrics = saved["assemblies"].tolist(), saved["metrics"].tolist()
        for ii, instance in enumerate(contrasts.CASES):
            summaries[instance] = {}
            for fi, family in enumerate(contrasts.FAMILIES):
                key = instance+"__"+family+"__N32"
                traces = saved[key]
                assert traces.shape == (128, 8, 201, 4)
                # Same bootstrap seeds as the original analysis.
                result = contrasts.paired_summary(traces, assemblies, metrics,
                                                  contrasts.BOOTSTRAP_SEED+10*ii+fi)
                mean, low, high, _ = result
                case = audit["cases"][key]
                for aj, component in enumerate(contrasts.COMPONENTS):
                    for mj, metric in enumerate(contrasts.METRICS):
                        for k in contrasts.CHECKPOINTS:
                            saved_k = case["omissions"][component]["metrics"][metric][str(k)]
                            assert saved_k["paired_mean_off_minus_full"] == float(mean[aj, k, mj])
                            assert saved_k["pointwise_95pct_bootstrap_interval"] == \
                                [float(low[aj, k, mj]), float(high[aj, k, mj])]
                summaries[instance][family] = result
    snapshots = {}
    plt.rcdefaults()
    with routing(output, snapshots, dynamic_labels.numerical_snapshot):
        for instance in contrasts.CASES:
            name = "trajectory_contrasts_"+instance.replace("_d4", "")+".pdf"
            contrasts.plot_objective(instance, summaries[instance], ROOT/"figures"/name)
    recorded = json.loads((LABELS/"dynamic_contrast_labels.json").read_text())["revised_label_pass"]
    return {new: snapshots[new] == recorded[old]["numerical_sha256"]
            for old, new in dynamic_labels.NAMES.items()}


def finite_population(output):
    """Figures trajectory_failures / trajectory_discovery_bounds and the table."""
    audit = json.loads((TRAJECTORIES/"finite_population_audit.json").read_text())
    arrays_path = TRAJECTORIES/"finite_population_analysis.npz"
    assert sha256(arrays_path) == audit["arrays_sha256"], "Analysis arrays differ from the audited file"
    summary = json.loads((TRAJECTORIES/"finite-population-full/summary.json").read_text())
    with np.load(arrays_path) as a:
        failure, upper, joint = a["failure"], a["upper"], a["joint_primary"]
        archive, budget = a["archive_failure"], a["archive_budget_failure"]
        discounted, undiscounted = a["flag_discounted"], a["flag_undiscounted"]
    snapshots = {}
    plt.rcdefaults()
    with routing(output, snapshots, population_labels.numerical_snapshot), \
            redirected_hashes(finite_figures, "digest"):
        finite_figures.plot(summary, failure, upper, discounted, undiscounted, {"figures": {}})
    rows, agree = [], True
    for ni, saved in enumerate(audit["by_population"]):
        row = {"N": saved["N"],
               "offspring_failure_k200": float(failure[:, ni, :, :, 0, 0, -1].mean()),
               "archive_failure_k200": float(archive[:, ni, :, :, 0, -1].mean()),
               "archive_failure_8192_calls": float(budget[:, ni, :, :, 0, -1].mean()),
               "pointwise_05_cells": int((upper[:, ni, :, :, 0, 0, -1] <= .05).sum()),
               "joint_10_cells": int((joint[:, ni, :, :, 0, -1] <= .1).sum())}
        agree &= (np.isclose(row["offspring_failure_k200"], saved["offspring_failure_k200"])
                  and np.isclose(row["archive_failure_k200"], saved["archive_failure_k200"])
                  and np.isclose(row["archive_failure_8192_calls"], saved["archive_failure_by_budget"][-1])
                  and row["pointwise_05_cells"] == saved["pointwise_05_cells_by_stream"][0]
                  and row["joint_10_cells"] == saved["joint_10_cells_by_stream"][0])
        rows.append(row)
    return rows, agree


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT/"reproduced-figures",
                        help="Directory for the regenerated figures (default: reproduced-figures/)")
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checks = {}
    checks.update(population_figures(output))
    checks.update(dynamic_figures(output))
    table, table_ok = finite_population(output)
    print("Finite-population table (epsilon = 0.1; percentages averaged over 320 cells per N)")
    print(f"{'N':>4} {'offspring':>10} {'archive':>8} {'8192 calls':>11} {'pointwise':>10} {'joint':>6}")
    for r in table:
        print(f"{r['N']:>4} {100*r['offspring_failure_k200']:>10.2f} {100*r['archive_failure_k200']:>8.2f} "
              f"{100*r['archive_failure_8192_calls']:>11.2f} {r['pointwise_05_cells']:>6}/320 {r['joint_10_cells']:>3}/320")
    print()
    for name, ok in checks.items():
        print(("MATCH   " if ok else "DIFFERS ")+name+": plotted data vs. manuscript figure")
    print(("MATCH   " if table_ok else "DIFFERS ")+"finite-population table vs. saved audit")
    print("Figures written to", output)
    if not (all(checks.values()) and table_ok):
        raise SystemExit("Some outputs differ from the manuscript records (see above).")


if __name__ == "__main__":
    main()
