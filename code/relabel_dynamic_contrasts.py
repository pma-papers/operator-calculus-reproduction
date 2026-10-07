"""Presentation-only manuscript labels for the two dynamic-contrast figures.

The original panel titles ("Mean objective gap", ...) read as levels of each
metric, although every curve is a difference between two assemblies
(omitted minus full).  This script recomputes the identical paired summaries
from the frozen trajectory data, calls the original plotting function, and only
rewrites the reader-facing title and row-label strings through temporary
Axes.set_title / Axes.set_ylabel wrappers.  Figure.savefig is redirected to
new names.  A dry original-label pass verifies that all plotted data,
limits and axes positions are identical.  No experiment, the analysis
main(), or any old output writer is called; the original assets stay untouched.
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

import dynamic_contrasts as contrasts
import trajectory_study as experiment


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
DATA = ROOT / "results/trajectories/finite-population-full"
CONTRAST_AUDIT = ROOT / "results/trajectories/dynamic_contrast_audit.json"
AUDIT = ROOT / "results/figure-labels/dynamic_contrast_labels.json"
NAMES = {
    "trajectory_contrasts_wells": "canonical_dynamic_contrasts_wells",
    "trajectory_contrasts_quartic": "canonical_dynamic_contrasts_quartic",
}
# Exact-string replacements. Every panel plots omitted-minus-full differences,
# so the titles now say so; the row labels name the initialization family with
# the manuscript's wording.
TITLES = {
    "Mean objective gap": "Mean objective gap:\nomitted minus full assembly",
    "Best-so-far objective gap": "Best-so-far objective gap:\nomitted minus full assembly",
    "Population variance": "Population variance:\nomitted minus full assembly",
}
YLABELS = {
    "Broad clouds\nOmitted minus full": "Broad starts\npaired difference",
    "Separated clusters\nOmitted minus full": "Clustered starts\npaired difference",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_signature(value):
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype.kind in "biufc":
        payload = np.ascontiguousarray(np.asarray(array, dtype=np.float64)).tobytes()
    else:
        payload = repr(array.tolist()).encode()
    return {"shape": list(array.shape), "sha256": hashlib.sha256(payload).hexdigest()}


def numerical_snapshot(fig):
    """No text values or text bounding boxes enter the plotted-data digest."""
    axes = []
    for ax in fig.axes:
        axes.append({
            "position": list(ax.get_position().bounds),
            "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
            "xscale": ax.get_xscale(), "yscale": ax.get_yscale(),
            "xticks": list(map(float, ax.get_xticks())),
            "lines": [{"x": array_signature(line.get_xdata()), "y": array_signature(line.get_ydata()),
                       "linewidth": line.get_linewidth(), "linestyle": line.get_linestyle(),
                       "color": str(line.get_color())} for line in ax.lines],
            "collections": [{"paths": [array_signature(path.vertices) for path in collection.get_paths()],
                             "facecolor": array_signature(collection.get_facecolor())}
                            for collection in ax.collections],
        })
    data = {"size_inches": fig.get_size_inches().tolist(), "axes": axes}
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


@contextmanager
def figure_routing(relabel, write, observed):
    original_title, original_ylabel, original_save = Axes.set_title, Axes.set_ylabel, Figure.savefig
    changes = []

    def set_title(self, label, *args, **kwargs):
        revised = TITLES.get(label, label) if relabel else label
        if revised != label:
            changes.append({"kind": "title", "old": label, "new": revised})
            kwargs = {**kwargs, "fontsize": 8.5, "linespacing": 1.15}
        return original_title(self, revised, *args, **kwargs)

    def set_ylabel(self, label, *args, **kwargs):
        revised = YLABELS.get(label, label) if relabel else label
        if revised != label:
            changes.append({"kind": "ylabel", "old": label, "new": revised})
        return original_ylabel(self, revised, *args, **kwargs)

    def save(self, fname, *args, **kwargs):
        old_path = Path(fname)
        if old_path.parent.resolve() != FIGURES.resolve() or old_path.stem not in NAMES:
            raise ValueError("Unexpected output from original plotting function: "+str(old_path))
        new_path = FIGURES/(NAMES[old_path.stem]+old_path.suffix)
        self.canvas.draw()
        renderer = self.canvas.get_renderer()
        clipped = [t.get_text() for ax in self.axes for t in (ax.title, ax.yaxis.label)
                   if t.get_text() and not self.bbox.contains(*t.get_window_extent(renderer).corners()[3])]
        observed[old_path.stem] = {"numerical_sha256": numerical_snapshot(self),
                                   "text_changes": list(changes),
                                   "text_outside_figure": clipped}
        changes.clear()
        if write:
            metadata = dict(kwargs.get("metadata", {}))
            metadata.update(Title=NAMES[old_path.stem], Creator=Path(__file__).name)
            kwargs["metadata"] = metadata
            with new_path.open("xb") as stream:   # exclusive creation: never overwrite
                original_save(self, stream, *args, format=old_path.suffix[1:], **kwargs)

    Axes.set_title, Axes.set_ylabel, Figure.savefig = set_title, set_ylabel, save
    try:
        yield
    finally:
        Axes.set_title, Axes.set_ylabel, Figure.savefig = original_title, original_ylabel, original_save


def load_summaries(summary):
    """The same loading, checks, order selection and bootstrap seeds as the dynamic_contrasts main()."""
    out = {}
    for ii, instance in enumerate(contrasts.CASES):
        summaries = {}
        for fi, family in enumerate(contrasts.FAMILIES):
            key = instance+"__"+family+"__N32"
            path = DATA/(key+".npz")
            assert sha256(path) == summary["cases"][key]["output_sha256"], key
            with np.load(path) as saved:
                assemblies = saved["assemblies"].tolist()
                orders = saved["orders"].tolist()
                metrics = saved["metrics"].tolist()
                assert assemblies == [a["id"] for a in summary["assemblies"]]
                assert metrics == summary["metrics"]
                traces = saved["traces"][:, :, orders.index("MRS")].copy()
                assert traces.shape == (128, 8, 201, 4)
            seed = contrasts.BOOTSTRAP_SEED+10*ii+fi
            summaries[family] = contrasts.paired_summary(traces, assemblies, metrics, seed)
        out[instance] = summaries
    return out


def plot_pass(summaries, relabel=False, write=False):
    snapshots = {}
    plt.rcdefaults()
    with figure_routing(relabel, write, snapshots):
        for instance in contrasts.CASES:
            output = FIGURES/("trajectory_contrasts_"+instance.replace("_d4", "")+".pdf")
            contrasts.plot_objective(instance, summaries[instance], output)
    assert set(snapshots) == set(NAMES)
    return snapshots


def main(check=False):
    summary_path = DATA/"summary.json"
    summary = json.loads(summary_path.read_text())
    contrast_audit = json.loads(CONTRAST_AUDIT.read_text())
    assert summary["status"] == "full"
    assert contrast_audit["source_summary_sha256"] == sha256(summary_path)
    assert contrast_audit["script_sha256"] == sha256(contrasts.__file__)
    assert summary["sources_sha256"]["runner"] == sha256(experiment.__file__)
    assert summary["sources_sha256"]["protocol"] == sha256(experiment.PROTOCOL)
    original_assets = [FIGURES/(name+".pdf") for name in NAMES]
    for path in original_assets:
        assert sha256(path) == contrast_audit["figures"][path.name], path.name
    protected = original_assets+[Path(__file__), Path(contrasts.__file__), Path(experiment.__file__),
                                 experiment.PROTOCOL, summary_path, CONTRAST_AUDIT]
    before = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    outputs = [FIGURES/(name+".pdf") for name in NAMES.values()]
    if not check and any(path.exists() for path in outputs+[AUDIT]):
        raise FileExistsError("Preserve previous relabeled results; never overwrite figure outputs or audit")
    summaries = load_summaries(summary)
    # The recomputed bootstrap contrasts must agree with the frozen contrast audit.
    for instance, per_family in summaries.items():
        for family, (mean, low, high, _) in per_family.items():
            case = contrast_audit["cases"][instance+"__"+family+"__N32"]
            for aj, component in enumerate(contrasts.COMPONENTS):
                for mj, metric in enumerate(contrasts.METRICS):
                    for k in contrasts.CHECKPOINTS:
                        saved = case["omissions"][component]["metrics"][metric][str(k)]
                        assert saved["paired_mean_off_minus_full"] == float(mean[aj, k, mj])
                        assert saved["pointwise_95pct_bootstrap_interval"] == [float(low[aj, k, mj]), float(high[aj, k, mj])]
    baseline = plot_pass(summaries)
    revised = plot_pass(summaries, relabel=True, write=not check)
    for name in NAMES:
        assert baseline[name]["numerical_sha256"] == revised[name]["numerical_sha256"], name
        assert len(revised[name]["text_changes"]) == 5, revised[name]["text_changes"]
        assert not revised[name]["text_outside_figure"], revised[name]["text_outside_figure"]
    after = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    assert before == after, "An original asset, script, or data file changed"
    audit = {"status": "passed",
             "scope": "Presentation-only panel titles and row labels; no experiment or numerical-analysis rerun",
             "reason": "The original titles named the metric only; every panel plots omitted-minus-full differences",
             "titles": TITLES, "row_labels": YLABELS,
             "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
             "protected_sha256_before": before, "protected_sha256_after": after,
             "all_original_files_unchanged": True,
             "numerical_artist_data_limits_positions_unchanged": True,
             "original_label_dry_pass": baseline, "revised_label_pass": revised,
             "visual_review": "Not asserted by the script; inspect the PDFs before delivery.",
             "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in outputs} if not check else {}}
    if not check:
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT.open("x") as stream:
            json.dump(audit, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print("PASS: both figures retain identical plotted numbers, limits and axes positions; "
          "all original assets/data/scripts unchanged")
    if not check:
        print("Audit:", AUDIT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Dry-render both label versions; write no outputs")
    main(check=parser.parse_args().check)
