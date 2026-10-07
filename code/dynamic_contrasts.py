"""Paired dynamic S/R contrasts; preserve all runs and the full time horizon.

The complete experiment also removes H.  H contrasts are reported numerically,
but omitted from these two figures to focus on S/R without crowding the panels.
Bands are pointwise percentile-bootstrap intervals for the mean contrast, not
simultaneous bands and not between-run quantile bands.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[1]
CASES = ("wells_d4", "quartic_d4")
FAMILIES = ("broad", "clustered")
FULL = "S1_R1_H1"
OMISSIONS = ("S0_R1_H1", "S1_R0_H1", "S1_R1_H0")
COMPONENTS = ("S", "R", "H")
METRICS = ("mean_gap", "record_gap", "variance")
CHECKPOINTS = (0, 1, 5, 10, 25, 50, 100, 200)
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 9472600000
COLORS = ("#246f96", "#c16c23")
X_TICKS = (0, 1, 5, 20, 50, 100, 200)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_summary(traces, assemblies, metrics, seed):
    """Summarize runwise differences, never differences of medians.

    Input has axes (independent run, assembly, generation, metric).  The same
    multinomial run resample is applied to every assembly, metric and time.
    Initial states must have been shared within each run across assemblies;
    this does not require subsequent random draws to have been shared.
    """
    assert traces.ndim == 4 and traces.shape[2] == 201
    assert traces.shape[0] == 128
    assert np.isfinite(traces).all()
    assert len(assemblies) == traces.shape[1]
    assert len(metrics) == traces.shape[3]
    ai = [assemblies.index(name) for name in OMISSIONS]
    fi = assemblies.index(FULL)
    mi = [metrics.index(name) for name in METRICS]
    selected = traces[..., mi]
    paired = selected[:, ai] - selected[:, fi, None]
    assert np.array_equal(paired[:, :, 0], np.zeros_like(paired[:, :, 0]))
    rng = np.random.default_rng(seed)
    n = len(traces)
    weights = rng.multinomial(n, np.full(n, 1/n), size=BOOTSTRAP_REPETITIONS)/n
    boot = (weights @ paired.reshape(n, -1)).reshape(
        BOOTSTRAP_REPETITIONS, *paired.shape[1:])
    low, high = np.quantile(boot, [.025, .975], axis=0)
    mean = paired.mean(axis=0)
    report = {
        "bootstrap_seed": seed,
        "runs": n,
        "reference_assembly": FULL,
        "initial_pairing_verified_by_equal_metrics": True,
        "reference": {},
        "omissions": {},
    }
    for mj, metric in enumerate(METRICS):
        report["reference"][metric] = {
            str(k): {
                "mean": float(selected[:, fi, k, mj].mean()),
                "quartiles": np.quantile(selected[:, fi, k, mj], [.25, .5, .75]).tolist(),
            } for k in CHECKPOINTS
        }
    for aj, (name, component) in enumerate(zip(OMISSIONS, COMPONENTS)):
        report["omissions"][component] = {"assembly": name, "metrics": {}}
        for mj, metric in enumerate(METRICS):
            report["omissions"][component]["metrics"][metric] = {
                str(k): {
                    "paired_mean_off_minus_full": float(mean[aj, k, mj]),
                    "pointwise_95pct_bootstrap_interval": [float(low[aj, k, mj]), float(high[aj, k, mj])],
                    "paired_quartiles": np.quantile(paired[:, aj, k, mj], [.25, .5, .75]).tolist(),
                    "fraction_off_minus_full_above_1e_minus_12": float(np.mean(paired[:, aj, k, mj] > 1e-12)),
                    "fraction_off_minus_full_below_minus_1e_minus_12": float(np.mean(paired[:, aj, k, mj] < -1e-12)),
                } for k in CHECKPOINTS
            }
    return mean, low, high, report


def plot_objective(instance, summaries, output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.45), sharex=True)
    generations = np.arange(201)
    for row, family in enumerate(FAMILIES):
        mean, low, high, _ = summaries[family]
        for col, title in enumerate(("Mean objective gap", "Best-so-far objective gap", "Population variance")):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(title, pad=7)
            ax.axhline(0., color=".25", lw=.7, ls="--", zorder=1)
            for aj, color in enumerate(COLORS):
                ax.fill_between(generations, low[aj, :, col], high[aj, :, col],
                                color=color, alpha=.16, linewidth=0, zorder=2)
                ax.plot(generations, mean[aj, :, col], color=color,
                        lw=1.3, ls="-" if aj == 0 else "-.", zorder=3)
            ax.set_xscale("function", functions=(np.log1p, np.expm1))
            ax.set_xlim(0, 200)
            ax.set_xticks(X_TICKS)
            ax.set_xticklabels([str(k) for k in X_TICKS])
            ax.set_xlabel("Generation (log(1+k) spacing)")
            ax.grid(alpha=.18, lw=.5)
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        axes[row, 0].set_ylabel(("Broad clouds" if family == "broad" else "Separated clusters")
                                + "\nOmitted minus full", fontsize=8)
    handles = [Line2D([0], [0], color=c, lw=1.5, ls=ls, label=label)
               for c, ls, label in zip(COLORS, ("-", "-."), ("Without selection S", "Without recombination R"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .012),
               ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.112, right=.989, bottom=.205, top=.92,
                        hspace=.49, wspace=.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"results/trajectories/finite-population-full")
    parser.add_argument("--check", action="store_true", help="Run an in-memory algebraic self-test; write no output")
    args = parser.parse_args()
    if args.check:
        check()
        return
    import trajectory_study as experiment
    summary_path = args.input/"summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "full"
    assert summary["generations"] == 200 and summary["runs_per_cell"] == 128
    assert summary["sources_sha256"]["runner"] == sha256(experiment.__file__)
    assert summary["sources_sha256"]["protocol"] == sha256(experiment.PROTOCOL)
    audit = {
        "script_sha256": sha256(__file__),
        "source_summary_sha256": sha256(summary_path),
        "population": 32,
        "order": "MRS",
        "displayed_omissions": ["S", "R"],
        "numeric_omissions": list(COMPONENTS),
        "pairing": summary["randomness_pairing"],
        "statistic": "Mean of runwise omitted-minus-full differences",
        "bands": "Pointwise 95% percentile-bootstrap intervals; not simultaneous confidence bands",
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_resampling_unit": "Run index, shared across all assemblies, metrics and generations",
        "x_scale": "All 201 generations; positions proportional to log(1+generation)",
        "y_scale": "Linear, independently scaled panels; zero shown in every panel",
        "interpretation": "Positive gap differences favor the full assembly; positive variance differences "
                          "mean the omission retains more population spread, not necessarily worse optimization.",
        "cases": {},
        "figures": {},
    }
    for ii, instance in enumerate(CASES):
        summaries = {}
        for fi, family in enumerate(FAMILIES):
            key = instance+"__"+family+"__N32"
            path = args.input/(key+".npz")
            assert sha256(path) == summary["cases"][key]["output_sha256"]
            with np.load(path) as saved:
                assemblies = saved["assemblies"].tolist()
                orders = saved["orders"].tolist()
                metrics = saved["metrics"].tolist()
                assert assemblies == [a["id"] for a in summary["assemblies"]]
                assert metrics == summary["metrics"]
                oi = orders.index("MRS")
                traces = saved["traces"][:, :, oi].copy()
                assert traces.shape == (128, 8, 201, 4)
                assert saved["initial_points"].shape == (128, 32, 4)
                assert np.all(traces >= 0)
                record = traces[..., metrics.index("record_gap")]
                assert np.all(np.diff(record, axis=2) <= 0)
                assert np.all(record <= traces[..., metrics.index("mean_gap")]+1e-14)
                final_values = saved["final_values"][:, :, oi]
                assert np.allclose(final_values.mean(axis=-1), traces[:, :, -1, metrics.index("mean_gap")])
                assert np.allclose(traces[:, :, 0], traces[:, :1, 0], rtol=0, atol=0)
                seed = BOOTSTRAP_SEED+10*ii+fi
                summaries[family] = paired_summary(traces, assemblies, metrics, seed)
                case = summaries[family][3]
                epsilon = float(summary["primary_epsilon"])
                case["primary_epsilon"] = epsilon
                case["source_sha256"] = sha256(path)
                case["archive_success_counts_by_assembly"] = {
                    a: int(np.sum(record[:, j, -1] <= epsilon))
                    for j, a in enumerate(assemblies)
                }
                case["initial_archive_success_count"] = int(np.sum(record[:, 0, 0] <= epsilon))
                case["final_mean_gap_above_epsilon_and_variance_below_1e_minus_10"] = {
                    a: int(np.sum((traces[:, j, -1, metrics.index("mean_gap")] > epsilon)
                                  & (traces[:, j, -1, metrics.index("variance")] < 1e-10)))
                    for j, a in enumerate(assemblies)
                }
                audit["cases"][key] = case
        output = ROOT/"figures"/("trajectory_contrasts_"+instance.replace("_d4", "")+".pdf")
        plot_objective(instance, summaries, output)
        audit["figures"][output.name] = sha256(output)
    audit_path = ROOT/"results/trajectories/dynamic_contrast_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2)+"\n")
    print("PASS: complete predeclared dynamic subset, all 128 runs; source hashes, initial pairing, "
          "archive monotonicity, final mean gaps and paired-bootstrap contrasts checked")
    print(audit_path)


def check():
    """Exact constant paired effects must give the same mean and interval.

    This synthetic algebraic fixture is never saved or plotted as evidence.
    """
    names = [FULL, *OMISSIONS]
    fixture = np.ones((128, 4, 201, 3))
    expected = np.empty((3, 201, 3))
    for j in range(3):
        effect = (j+1)*np.arange(201)/200
        fixture[:, j+1] += effect[:, None]
        expected[j] = effect[:, None]
    mean, low, high, _ = paired_summary(fixture, names, list(METRICS), BOOTSTRAP_SEED)
    assert np.allclose(mean, expected, rtol=0, atol=1e-13)
    assert np.allclose(low, expected, rtol=0, atol=1e-13)
    assert np.allclose(high, expected, rtol=0, atol=1e-13)
    assert np.array_equal(mean[:, 0], np.zeros((3, 3)))
    print("PASS: runwise pairing, initial zero contrast, fixed-seed bootstrap and exact-effect self-test")


if __name__ == "__main__":
    main()
