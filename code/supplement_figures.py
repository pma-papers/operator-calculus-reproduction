"""Three figures of the supplement in the wording of the current version of the paper.

    supplement_budget_curves.pdf            archive failure frequency against function evaluations
                                            (drawn by budget_curves.py)
    supplement_dynamic_contrasts_wells.pdf  dynamic contrasts on unequal wells (relabel_dynamic_contrasts.py)
    supplement_exploration.pdf              exploration diagnostics of Gaussian mutation
                                            (exploration_figures.py with relabel_exploration_figures.py)

Each figure is drawn by its original script from the same data, in a temporary folder, so that
nothing under results/ and no earlier figure is written.  Only the lettering is changed when the
figure is saved (`wording.py`).  The plotted data are checked against the shipped results: the
budget curves against results/analysis/budget_curves.npz, and the two other figures against the
plotted-data digests in results/figure-labels/.

Needs the trajectory files (`python code/trajectory_study.py --run`), the one-step arrays
(`python code/regenerate_arrays.py`) and the exploration run (`python code/exploration_study.py --run`)
for the exploration figure.  Writes the three figures to figures/ and
results/analysis/supplement_figures_audit.json.

    python code/supplement_figures.py [budget_curves] [dynamic_contrasts] [exploration]
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
FIGURES, RESULTS = ROOT/"figures", ROOT/"results/analysis"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import wording

CHANGED = []
_savefig = Figure.savefig


def savefig(self, fname, *args, **kwargs):
    CHANGED.extend(wording.relabel(self))
    return _savefig(self, fname, *args, **kwargs)


Figure.savefig = savefig


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def take(work, name, new_name):
    FIGURES.mkdir(exist_ok=True)
    shutil.copy(work/name, FIGURES/new_name)
    changed = sorted({tuple(c) for c in CHANGED})
    CHANGED.clear()
    assert changed, name
    print("Wrote", FIGURES/new_name)
    return {"figure": new_name, "sha256": sha256(FIGURES/new_name), "relabelled": [list(c) for c in changed]}


def budget_curves(work):
    import budget_curves as bc
    bc.OUT, bc.AUDIT, bc.FIGURE = work/"budget_curves.npz", work/"budget_curves_audit.json", work/"budget_curves.pdf"
    bc.main()
    with np.load(work/"budget_curves.npz") as new, np.load(RESULTS/"budget_curves.npz") as old:
        assert set(new.files) == set(old.files) and all(np.array_equal(new[k], old[k]) for k in old.files)
    return {**take(work, "budget_curves.pdf", "supplement_budget_curves.pdf"), "curves_equal_shipped_arrays": True}


def dynamic_contrasts(work):
    import relabel_dynamic_contrasts as dc
    recorded = json.loads((ROOT/"results/figure-labels/dynamic_contrast_labels.json").read_text())["revised_label_pass"]
    dc.FIGURES = work
    dc.NAMES = {k: k.replace("trajectory_contrasts", "supplement_dynamic_contrasts") for k in dc.NAMES}
    summaries = dc.load_summaries(json.loads((dc.DATA/"summary.json").read_text()))
    baseline = dc.plot_pass(summaries)
    revised = dc.plot_pass(summaries, relabel=True, write=True)
    for name in dc.NAMES:
        assert baseline[name]["numerical_sha256"] == revised[name]["numerical_sha256"] == recorded[name]["numerical_sha256"], name
    return {**take(work, "supplement_dynamic_contrasts_wells.pdf", "supplement_dynamic_contrasts_wells.pdf"),
            "plotted_data_equal_shipped_digest": True}


def exploration(work):
    import exploration_figures as ef
    import relabel_exploration_figures as rn
    import relabel_trajectory_figures as population_labels
    recorded = json.loads((ROOT/"results/figure-labels/population_figure_labels.json").read_text())["revised_label_pass"]
    ef.FIGURES = rn.FIGURES = work

    def save_figure(fig, name):   # as in the original, without the record of the original file
        fig.savefig(work/(name+".pdf"), metadata={"Title": name, "Creator": "exploration_figures.py"})
        fig.savefig(work/(name+".png"), dpi=200)
        plt.close(fig)
        return {}

    ef.save_figure = save_figure
    rn.NAMES = {"exploration_noise": "supplement_exploration"}
    observed, changes = {}, []
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    with rn.routing(True, observed, changes):
        ef.figures(population_labels.primary_rows())
    assert observed["exploration_noise"]["numerical_sha256"] == recorded["exploration_noise"]["numerical_sha256"]
    return {**take(work, "supplement_exploration.pdf", "supplement_exploration.pdf"), "plotted_data_equal_shipped_digest": True}


if __name__ == "__main__":
    functions = {"budget_curves": budget_curves, "dynamic_contrasts": dynamic_contrasts, "exploration": exploration}
    chosen = sys.argv[1:] or list(functions)
    path = RESULTS/"supplement_figures_audit.json"
    audit = json.loads(path.read_text()) if sys.argv[1:] and path.exists() else {}
    audit["replacements"] = [list(r) for r in wording.REPLACEMENTS]
    with tempfile.TemporaryDirectory() as tmp:
        for name in chosen:
            work = Path(tmp)/name
            work.mkdir()
            audit[name] = functions[name](work)
    audit["script_sha256"] = sha256(__file__)
    path.write_text(json.dumps(audit, indent=1)+"\n")
    print("Wrote", path)
