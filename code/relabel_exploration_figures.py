"""Presentation-only notation labels for three appendix figures.

Exploration figures: "record" -> "best-so-far" and the batch gain J -> Delta B.
Audit-batch figure: batch size b -> n (b is the drift field).

Builds on relabel_trajectory_figures.py: the original exploration plotting function
consumes its original saved data; temporary wrappers around the text-emitting
Matplotlib calls apply the population terminology changes plus the notation renaming, and
Figure.savefig is redirected to the manuscript names.  The plotted-data digest must equal
the one recorded in results/figure-labels/population_figure_labels.json.
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

import exploration_figures
import trajectory_figures as finite_figures
import relabel_trajectory_figures as population_labels


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT/"figures"
AUDIT = ROOT/"results/analysis/notation_labels.json"
NAMES = {"exploration_noise": "canonical_exploration",
         "exploration_components": "canonical_exploration_components",
         "trajectory_discovery_bounds": "canonical_discovery_bounds"}
REPLACEMENTS = population_labels.REPLACEMENTS + (
    ("Paired record advantage\npositive = better record", "Paired best-so-far advantage\npositive = better best-so-far value"),
    ("Record comparison\nall paired batches", "Best-so-far comparison\nall paired batches"),
    ("immediate record opportunities", "immediate best-so-far opportunities"),
    ("Better record", "Better best-so-far value"), ("Worse record", "Worse best-so-far value"),
    ("record outcomes pool", "best-so-far outcomes pool"),
    ("Record advantage\n(normalized; signed log)", "Best-so-far advantage\n(normalized; signed log)"),
    ("Immediate record effects", "Immediate best-so-far effects"),
    ("positive means a better record", "positive means a better best-so-far value"),
    (r"$(J_{\rm on}-J_{\rm off})/(1+f_{\min})$", r"$(\Delta B_{\rm on}-\Delta B_{\rm off})/(1+f_{\min})$"),
    ("Audit batch b=", "Audit batch n="),
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def changed(text):
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    assert "record" not in text.lower() and "J_{" not in text and "batch b=" not in text, "unmapped figure string: "+text
    return text


@contextmanager
def routing(write, observed, changes):
    originals = (Figure.text, Figure.suptitle, Figure.legend, Axes.set_title, Axes.set_xlabel, Figure.savefig)

    def wrap(fn, position):
        def inner(self, *args, **kwargs):
            args = list(args)
            if len(args) > position and isinstance(args[position], str):
                new = changed(args[position])
                if new != args[position]:
                    changes.append({"old": args[position], "new": new})
                args[position] = new
            return fn(self, *args, **kwargs)
        return inner

    def legend(self, *args, **kwargs):
        for handle in kwargs.get("handles", []):
            new = changed(handle.get_label())
            if new != handle.get_label():
                changes.append({"old": handle.get_label(), "new": new})
                handle.set_label(new)
        return originals[2](self, *args, **kwargs)

    def save(self, fname, *args, **kwargs):
        old = Path(fname)
        if old.parent.resolve() != FIGURES.resolve():
            raise ValueError("Unexpected output: "+str(old))
        if old.stem not in NAMES:
            changes.clear()
            return   # other outputs of the same plot function are neither rewritten nor relabeled
        new = FIGURES/(NAMES[old.stem]+old.suffix)
        if old.suffix == ".pdf":
            self.canvas.draw()
            observed[old.stem] = {"numerical_sha256": population_labels.numerical_snapshot(self), "text_changes": list(changes)}
            changes.clear()
        if write:
            kwargs["metadata"] = {**kwargs.get("metadata", {}), "Title": NAMES[old.stem], "Creator": Path(__file__).name}
            with new.open("xb") as stream:
                originals[5](self, stream, *args, format=old.suffix[1:], **kwargs)

    Figure.text, Figure.suptitle, Axes.set_title, Axes.set_xlabel = (
        wrap(originals[0], 2), wrap(originals[1], 0), wrap(originals[3], 0), wrap(originals[4], 0))
    Figure.legend, Figure.savefig = legend, save
    try:
        yield
    finally:
        Figure.text, Figure.suptitle, Figure.legend, Axes.set_title, Axes.set_xlabel, Figure.savefig = originals


def main(check=False):
    recorded = json.loads((ROOT/"results/figure-labels/population_figure_labels.json").read_text())["revised_label_pass"]
    outputs = [FIGURES/(n+s) for n in NAMES.values() for s in (".pdf", ".png")]
    outputs = [p for p in outputs if not (p.stem.startswith("canonical_discovery") and p.suffix == ".png")]
    if not check and any(p.exists() for p in outputs+[AUDIT]):
        raise FileExistsError("Preserve previous notation outputs; never overwrite figures or audit")
    protected = {str(p.relative_to(ROOT)): sha256(p) for p in
                 [FIGURES/(n+s) for n in NAMES for s in (".pdf", ".png") if (FIGURES/(n+s)).exists()]
                 + [Path(__file__), Path(exploration_figures.__file__), Path(finite_figures.__file__), population_labels.PRIMARY]}
    observed, changes = {}, []
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    with routing(not check, observed, changes):
        exploration_figures.figures(population_labels.primary_rows())
    for name in list(NAMES)[:2]:
        assert observed[name]["numerical_sha256"] == recorded[name]["numerical_sha256"], name
        assert any("best-so-far" in c["new"] for c in observed[name]["text_changes"]), name
        assert any("Delta B" in c["new"] for c in observed[name]["text_changes"]), name
    # Audit-batch figure: redraw from the audited analysis arrays; an unlabeled dry pass fixes the digest.
    finite_audit = json.loads((ROOT/"results/trajectories/finite_population_audit.json").read_text())
    arrays_path = ROOT/"results/trajectories/finite_population_analysis.npz"
    assert sha256(arrays_path) == finite_audit["arrays_sha256"]
    summary = json.loads((ROOT/"results/trajectories/finite-population-full/summary.json").read_text())
    with np.load(arrays_path) as a:
        failure, upper, discounted, undiscounted = a["failure"], a["upper"], a["flag_discounted"], a["flag_undiscounted"]
    # The plot function is the unchanged trajectory-study source and its inputs are hash-verified above, so the
    # plotted data equal the original figure's; the original asset is not rewritten (save is redirected).
    original_digest = finite_figures.digest
    finite_figures.digest = lambda path: sha256(path) if Path(path).exists() else None
    try:
        with routing(not check, observed, changes):
            finite_figures.plot(summary, failure, upper, discounted, undiscounted, {"figures": {}})
    finally:
        finite_figures.digest = original_digest
    assert set(observed) == set(NAMES)
    assert any("Audit batch n=" in c["new"] for c in observed["trajectory_discovery_bounds"]["text_changes"])
    after = {k: sha256(ROOT/k) for k in protected}
    assert protected == after, "An original asset or script changed"
    audit = {"status": "passed", "scope": "Presentation-only wording; plotted data identical to the original and relabeled exploration figures",
             "replacements": list(REPLACEMENTS), "protected_sha256": protected, "passes": observed,
             "outputs": {str(p.relative_to(ROOT)): sha256(p) for p in outputs if p.exists()} if not check else {}}
    if not check:
        AUDIT.parent.mkdir(exist_ok=True)
        with AUDIT.open("x") as stream:
            json.dump(audit, stream, indent=2)
            stream.write("\n")
    print("PASS: plotted data unchanged; wording changed to best-so-far, Delta B and audit batch n", "" if check else str(AUDIT))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Dry run; write nothing")
    main(check=parser.parse_args().check)
