"""Presentation-only manuscript labels for four existing one-step and exploration figures.

The original plotting functions consume their original saved data. A temporary
Figure.text wrapper changes only the specified reader-facing terminology;
Figure.savefig redirects to new names. A dry original-label pass verifies
that all plotted numerical data, limits, and axes positions are identical.
No experiment, data-analysis main(), or old output writer is called.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

import one_step_figures as assembly_figures
import exploration_figures


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
AUDIT = ROOT / "results/figure-labels/population_figure_labels.json"
NAMES = {
    "one_step_consistency": "one_step_consistency_relabeled",
    "one_step_components": "one_step_components_relabeled",
    "exploration_noise": "exploration_noise_relabeled",
    "exploration_components": "exploration_components_relabeled",
}
REPLACEMENTS = (
    ("held-out laws", "validation populations"),
    ("finite-law expectations", "finite-population expectations"),
    ("sampled-law maxima", "sampled-population maxima"),
    ("law by law", "population by population"),
    ("input laws", "initial populations"),
    ("law-setting-order cells", "population-setting-order cells"),
    ("problem-family cases", "problem/initial-distribution cases"),
)
VALIDATION = ROOT / "results/one-step/canonical-validation"
PRIMARY = ROOT / "results/exploration/exploration_primary.csv"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def changed_text(text):
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text


def primary_rows():
    with PRIMARY.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 60
    strings = {"instance", "short_label", "population_family", "switch"}
    return [{key: value if key in strings else float(value) if value else None
             for key, value in row.items()} for row in rows]


def array_signature(value):
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype.kind in "biufc":
        array = np.asarray(array, dtype=np.float64)
        payload = np.ascontiguousarray(array).tobytes()
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
            "lines": [{"x": array_signature(line.get_xdata()),
                       "y": array_signature(line.get_ydata()),
                       "linewidth": line.get_linewidth(), "linestyle": line.get_linestyle(),
                       "marker": line.get_marker(), "color": str(line.get_color())}
                      for line in ax.lines],
            "images": [{"data": array_signature(image.get_array()),
                        "extent": list(image.get_extent()), "clim": list(image.get_clim()),
                        "cmap": image.get_cmap().name} for image in ax.images],
            "collections": [{"array": array_signature(collection.get_array()),
                              "offsets": array_signature(collection.get_offsets()),
                              "paths": [array_signature(path.vertices) for path in collection.get_paths()]}
                             for collection in ax.collections],
            "patches": [{"vertices": array_signature(patch.get_path().vertices),
                         "transform": array_signature(patch.get_transform().get_matrix()),
                         "facecolor": list(patch.get_facecolor())} for patch in ax.patches],
        })
    data = {"size_inches": fig.get_size_inches().tolist(), "axes": axes}
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


@contextmanager
def figure_routing(relabel, write, observed):
    original_text, original_save = Figure.text, Figure.savefig

    def text(self, x, y, s, *args, **kwargs):
        revised = changed_text(s) if relabel else s
        artist = original_text(self, x, y, revised, *args, **kwargs)
        if revised != s:
            artist._population_label_original = s
        return artist

    def save(self, fname, *args, **kwargs):
        old_path = Path(fname)
        if old_path.parent.resolve() != FIGURES.resolve() or old_path.stem not in NAMES:
            raise ValueError("Unexpected output from original plotting function: "+str(old_path))
        new_path = FIGURES/(NAMES[old_path.stem]+old_path.suffix)
        if old_path.suffix == ".pdf":
            self.canvas.draw()
            changes = []
            for artist in self.texts:
                if hasattr(artist, "_population_label_original"):
                    box = artist.get_window_extent(self.canvas.get_renderer())
                    changes.append({"old": artist._population_label_original,
                                    "new": artist.get_text(),
                                    "font_size": artist.get_fontsize(),
                                    "bounds_figure_fraction": [box.x0/self.bbox.width,
                                        box.y0/self.bbox.height, box.x1/self.bbox.width,
                                        box.y1/self.bbox.height]})
            observed[old_path.stem] = {"numerical_sha256": numerical_snapshot(self),
                                       "text_changes": changes}
        if write:
            metadata = dict(kwargs.get("metadata", {}))
            metadata.update(Title=NAMES[old_path.stem], Creator=Path(__file__).name)
            kwargs["metadata"] = metadata
            # Exclusive creation is the final overwrite guard. Explicit format
            # is needed because the destination is a binary handle, not a path.
            with new_path.open("xb") as stream:
                original_save(self, stream, *args, format=old_path.suffix[1:], **kwargs)

    Figure.text, Figure.savefig = text, save
    try:
        yield
    finally:
        Figure.text, Figure.savefig = original_text, original_save


def plot_pass(relabel=False, write=False):
    snapshots = {}
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    summary = json.loads((VALIDATION/"canonical_summary.json").read_text())
    with figure_routing(relabel, write, snapshots):
        with np.load(VALIDATION/"canonical_arrays.npz") as arrays:
            assembly_figures.consistency_figure(summary, arrays)
            assembly_figures.coefficient_figure(summary, arrays)
        # This calls only the original plot function, not its analysis main().
        # Its return values still name old assets; our audit deliberately does
        # not use those return values and instead hashes the new paths directly.
        exploration_figures.figures(primary_rows())
    assert set(snapshots) == set(NAMES)
    return snapshots


def main(check=False):
    original_assets = [FIGURES/(name+"."+suffix) for name in NAMES for suffix in ("pdf", "png")]
    source_paths = [Path(__file__), Path(assembly_figures.__file__), Path(exploration_figures.__file__),
                    Path(assembly_figures.model.__file__), Path(assembly_figures.model.objectives.__file__),
                    Path(exploration_figures.experiment.__file__)]
    data_paths = [VALIDATION/"canonical_summary.json", VALIDATION/"canonical_arrays.npz", PRIMARY,
                  ROOT/"results/one-step/one_step_figure_audit.json",
                  ROOT/"results/exploration/exploration_analysis.json"]
    protected = original_assets+source_paths+data_paths
    before = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    one_step_audit = json.loads(data_paths[-2].read_text())
    exploration_audit = json.loads(data_paths[-1].read_text())
    assert before[str((VALIDATION/"canonical_arrays.npz").relative_to(ROOT))] == \
        one_step_audit["artifact_hashes"]["validation"]["canonical_arrays.npz"]
    primary_audit = next(row for row in exploration_audit["data_tables"] if row["path"] == str(PRIMARY.relative_to(ROOT)))
    assert sha256(PRIMARY) == primary_audit["sha256"]
    outputs = [FIGURES/(name+"."+suffix) for name in NAMES.values() for suffix in ("pdf", "png")]
    if not check and any(path.exists() for path in outputs+[AUDIT]):
        raise FileExistsError("Preserve previous relabeled results; never overwrite figure outputs or audit")
    baseline = plot_pass()
    revised = plot_pass(relabel=True, write=not check)
    for name in NAMES:
        assert baseline[name]["numerical_sha256"] == revised[name]["numerical_sha256"], name
        assert revised[name]["text_changes"], "No relevant terminology found in "+name
    after = {str(path.relative_to(ROOT)): sha256(path) for path in protected}
    assert before == after, "An original asset, script, or data file changed"
    audit = {"status": "passed", "scope": "Presentation-only terminology; no experiment or numerical-analysis rerun",
             "replacements": list(REPLACEMENTS), "protected_sha256_before": before,
             "protected_sha256_after": after, "all_original_files_unchanged": True,
             "numerical_artist_data_limits_positions_unchanged": True,
             "original_label_dry_pass": baseline, "revised_label_pass": revised,
             "visual_review": "Not asserted by the script; inspect the exported PNGs before delivery.",
             "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in outputs} if not check else {}}
    if not check:
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT.open("x") as stream:
            json.dump(audit, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print("PASS: four figures retain identical plotted numbers, limits and axes positions; "
          "all original assets/data/scripts unchanged")
    if not check:
        print("Audit:", AUDIT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Dry-render both label versions; write no outputs")
    main(check=parser.parse_args().check)
