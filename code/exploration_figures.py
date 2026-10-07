"""Complete-matrix analysis of frozen-law exploration diagnostics.

Quantile ranges describe observed distributions, not confidence intervals.
No data or optimizer configurations are selected, refitted, or rerun here.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import exploration_study as experiment

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/exploration/exploration-full"
RESULTS = ROOT / "results/exploration"
FIGURES = ROOT / "figures"
SOURCE_SHA256 = "55522414951385f7e9eb4dfee018b0c0be53eac5a0bf304cc57cc5b0da4a742a"
QUANTILES = np.array([.05, .25, .50, .75, .95])
Q_NAMES = ("q05", "q25", "q50", "q75", "q95")
LABELS = ("PL 4D", "PL 8D", "Rotated PL 4D", "Rotated PL 8D", "Quartic 4D",
          "Quartic 8D", "Rastrigin 4D", "Rosenbrock 4D", "Unequal wells 4D", "Periodic 1D")
COLORS = {"H": "#18668c", "R": "#955a94", "S": "#347c63"}
POSITIVE = "#218675"
NEGATIVE = "#cc7946"
TIE = "#dedede"


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(prefix, values):
    values = np.asarray(values).reshape(-1)
    assert len(values) and np.isfinite(values).all(), prefix
    result = {prefix+"_mean": float(values.mean())}
    result.update({prefix+"_"+name: float(q) for name, q in zip(Q_NAMES, np.quantile(values, QUANTILES))})
    result[prefix+"_exact_zero_fraction"] = float(np.mean(values == 0))
    return result


def collection(arrays, assembly_indices, order_indices, ti, si):
    """Pool equal-size law/configuration cells; batches remain descriptive draws."""
    values = {name: [] for name in ("cost", "phase_cost", "gain", "sample_cost", "exclusive", "lost",
                                   "cost_positive", "novel_delta", "record_novel_delta", "record_delta",
                                   "variance_delta", "variance_on", "variance_off", "V", "best", "input_variance",
                                   "gain_on", "gain_off")}
    component_index = (3, 1, 0)[si]
    for ai in assembly_indices:
        off = experiment.disabled_index(ai, experiment.SWITCHES[si])
        assert off != ai
        for oi in order_indices:
            idx = (slice(None), ai, oi, ti)
            other = (slice(None), off, oi, ti)
            pair = idx+(si,)
            values["cost"].append(arrays["paired_exact_mean_cost"][pair])
            values["phase_cost"].append(arrays["exact_component_mean_changes"][idx+(component_index,)])
            values["gain"].append(arrays["paired_batch_record_gain_difference"][pair])
            values["sample_cost"].append(arrays["paired_batch_mean_gap_difference"][pair])
            values["exclusive"].append(arrays["paired_batch_exclusive_record"][pair])
            values["lost"].append(arrays["paired_batch_lost_record"][pair])
            values["cost_positive"].append(arrays["paired_exact_mean_cost_positive"][pair])
            values["gain_on"].append(arrays["batch_record_gain"][idx])
            values["gain_off"].append(arrays["batch_record_gain"][other])
            for name, source in (("novel_delta", "candidate_novel_count"),
                                 ("record_novel_delta", "candidate_record_and_novel_count"),
                                 ("record_delta", "candidate_record_count")):
                values[name].append((arrays[source][idx]-arrays[source][other])/experiment.CANDIDATES)
            on_variance, off_variance = arrays["exact_total_variance"][idx], arrays["exact_total_variance"][other]
            values["variance_delta"].append(on_variance-off_variance)
            values["variance_on"].append(on_variance)
            values["variance_off"].append(off_variance)
            for name, source in (("V", "V"), ("best", "input_best"), ("input_variance", "input_total_variance")):
                values[name].append(arrays[source])
    return {name: np.concatenate(parts, axis=0) for name, parts in values.items()}


def summarize(values, identity):
    cost, gain = values["cost"], values["gain"]
    mean_scale = 1+values["V"]
    record_scale = 1+values["best"]
    eta = experiment.RECORD_TOLERANCE*record_scale[:, None]
    better, worse, tied = gain > eta, gain < -eta, np.abs(gain) <= eta
    assert np.all(better.astype(int)+worse+tied == 1)
    positive_cost = values["cost_positive"][:, None]
    positive_opportunities = int(positive_cost.sum()*experiment.BATCHES)
    joint = positive_cost & better
    row = dict(identity)
    row.update(law_configuration_order_cells=len(cost), paired_batch_outcomes=gain.size,
               exact_positive_mean_cost_cells=int(positive_cost.sum()),
               exact_positive_mean_cost_fraction=float(positive_cost.mean()),
               positive_cost_batch_opportunities=positive_opportunities,
               paired_gain_positive_count=int(better.sum()), paired_gain_negative_count=int(worse.sum()),
               paired_gain_tied_count=int(tied.sum()),
               paired_gain_positive_probability=float(better.mean()),
               paired_gain_negative_probability=float(worse.mean()),
               paired_gain_tied_probability=float(tied.mean()),
               exclusive_record_probability=float(values["exclusive"].mean()),
               lost_record_probability=float(values["lost"].mean()),
               positive_cost_and_gain_count=int(joint.sum()),
               positive_cost_and_gain_probability=float(joint.mean()),
               positive_cost_and_gain_conditional_probability=(float(joint.sum()/positive_opportunities)
                                                              if positive_opportunities else None),
               positive_cost_and_exclusive_record_count=int((positive_cost & values["exclusive"]).sum()),
               positive_cost_and_lost_record_count=int((positive_cost & values["lost"]).sum()))
    for prefix, data in (("exact_mean_cost", cost), ("exact_mean_cost_normalized", cost/mean_scale),
                         ("phase_mean_cost", values["phase_cost"]),
                         ("phase_mean_cost_normalized", values["phase_cost"]/mean_scale),
                         ("paired_record_advantage", gain),
                         ("paired_record_advantage_normalized", gain/record_scale[:, None]),
                         ("paired_sample_mean_cost", values["sample_cost"]),
                         ("paired_sample_mean_cost_normalized", values["sample_cost"]/mean_scale[:, None]),
                         ("record_gain_enabled", values["gain_on"]),
                         ("record_gain_disabled", values["gain_off"]),
                         ("novelty_probability_difference", values["novel_delta"]),
                         ("record_and_novelty_probability_difference", values["record_novel_delta"]),
                         ("candidate_record_probability_difference", values["record_delta"]),
                         ("total_variance_enabled", values["variance_on"]),
                         ("total_variance_disabled", values["variance_off"]),
                         ("total_variance_difference", values["variance_delta"]),
                         ("total_variance_difference_normalized", values["variance_delta"]/(1+values["input_variance"]))):
        row.update(describe(prefix, data))
    return row


def audit_case(key, arrays, case, saved):
    assert case["laws"] == 128
    assert arrays["exact_phase_means"].shape == (128, 8, 2, 8, 5)
    assert np.array_equal(arrays["input_points"], saved[key+"_input_points"])
    assert np.array_equal(arrays["input_weights"], saved[key+"_input_weights"])
    assert experiment.digest_array(arrays["input_points"]) == case["input_points_sha256"]
    assert experiment.digest_array(arrays["input_weights"]) == case["input_weights_sha256"]
    V = arrays["V"][:, None, None, None]
    output = arrays["exact_output_mean"]
    assert np.allclose(output, saved[key+"_output_V"], rtol=1e-12, atol=1e-12)
    assert np.allclose(arrays["exact_component_mean_changes"].sum(-1), output-V, rtol=1e-12, atol=1e-12)
    assert np.allclose(arrays["paired_exact_mean_cost"][..., 0], arrays["exact_noise_cost"], rtol=0, atol=1e-12)
    signs = sum(arrays["paired_batch_record_gain_"+part+"_count"] for part in ("positive", "negative", "tied"))
    assert np.all(signs == experiment.BATCHES)
    assert np.all(arrays["paired_candidate_objective_difference_sign_counts"].sum(-1) == experiment.CANDIDATES)
    assert np.all(arrays["batch_record_gain"] >= 0)
    assert np.all(arrays["candidate_record_and_novel_count"] <= arrays["candidate_record_count"])
    assert np.all(arrays["candidate_record_and_novel_count"] <= arrays["candidate_novel_count"])
    assert not arrays["candidate_nonfinite_count"].any(), "Do not summarize finite-only outcomes after a failure"
    assert not arrays["input_scale_guard"].any(), "Guarded novelty needs explicit partial-data reporting"
    assert all(np.isfinite(arrays[name]).all() for name in
               ("exact_phase_means", "paired_batch_record_gain_difference", "exact_total_variance"))
    for ai, assembly in enumerate(experiment.model.ASSEMBLIES):
        off = experiment.disabled_index(ai, "H")
        if assembly["sigma"]:
            difference = arrays["exact_total_variance"][:, ai]-arrays["exact_total_variance"][:, off]
            assert np.allclose(difference, experiment.model.TAUS[None, :]*.2**2*arrays["input_points"].shape[-1],
                               atol=1e-12, rtol=1e-12)
    return {"maximum_one_step_mean_relative_error": float(arrays["maximum_one_step_mean_relative_error"].max()),
            "maximum_telescoping_error": float(arrays["maximum_telescoping_error"].max()),
            "nonfinite_candidate_outcomes": 0, "input_scale_guards": 0,
            "material_record_guards": int(arrays["material_record_guard"].sum())}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.8,
                         "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 7.1, "ytick.labelsize": 7.7,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})


def row_background(ax, count):
    for i in range(count):
        if (i//2) % 2 == 0:
            ax.axhspan(i-.48, i+.48, color="#f3f4f4", zorder=0)
    ax.set_ylim(count-.55, -.55)
    ax.set_yticks(np.arange(count))
    ax.tick_params(axis="y", length=0, pad=5)


def quantile_panel(ax, rows, prefix, color):
    row_background(ax, len(rows))
    for i, row in enumerate(rows):
        qs = [row[prefix+"_"+q] for q in Q_NAMES]
        ax.plot([qs[0], qs[-1]], [i, i], color=color, lw=1.1, zorder=2)
        ax.plot([qs[1], qs[-2]], [i, i], color=color, lw=4.2, solid_capstyle="butt", zorder=3)
        ax.plot(qs[2], i, "|", color="#101820", ms=7, mew=1.05, zorder=5)
        ax.plot(row[prefix+"_mean"], i, "D", color=color, mec="white", mew=.3, ms=3.5, zorder=6)
    ax.axvline(0, color="#858585", lw=.7, zorder=1)
    ax.set_xscale("symlog", linthresh=1e-3, linscale=.7)
    ax.set_xticks([-1, -.01, 0, .01, 1], [r"$-1$", r"$-10^{-2}$", "0", r"$10^{-2}$", "1"])
    ax.minorticks_off()
    ax.grid(axis="x", alpha=.2, lw=.4)


def labels_for(rows):
    return [row["short_label"]+" / "+("broad" if row["population_family"] == "broad" else "clustered") for row in rows]


def quantile_legend(fig, y):
    items = [Line2D([0], [0], color="#47545b", lw=1.1, label="5th-95th percentiles"),
             Line2D([0], [0], color="#47545b", lw=4.2, label="25th-75th percentiles"),
             Line2D([0], [0], color="#101820", marker="|", linestyle="None", markersize=7, label="Median"),
             Line2D([0], [0], color="#47545b", marker="D", linestyle="None", markersize=3.5, label="Mean")]
    fig.legend(handles=items, loc="lower center", bbox_to_anchor=(.53, y), ncol=2,
               frameon=False, fontsize=7.2, columnspacing=1.5, handlelength=2.2)


def save_figure(fig, name):
    pdf, png = FIGURES/(name+".pdf"), FIGURES/(name+".png")
    fig.savefig(pdf, metadata={"Title": name, "Creator": "exploration_figures.py"})
    fig.savefig(png, dpi=200)
    width, height = fig.get_size_inches()
    plt.close(fig)
    return {"pdf": str(pdf.relative_to(ROOT)), "png": str(png.relative_to(ROOT)),
            "width_inches": float(width), "height_inches": float(height),
            "pdf_sha256": experiment.model.sha256(pdf), "png_sha256": experiment.model.sha256(png)}


def figures(primary):
    style()
    rows = [row for row in primary if row["switch"] == "H"]
    assert len(rows) == 20
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 7.9), sharey=True,
                             gridspec_kw={"width_ratios": [1.08, 1.18, .93]})
    fig.subplots_adjust(left=.27, right=.985, top=.858, bottom=.24, wspace=.24)
    quantile_panel(axes[0], rows, "exact_mean_cost_normalized", COLORS["H"])
    quantile_panel(axes[1], rows, "paired_record_advantage_normalized", COLORS["H"])
    minimum_cost = min(row["exact_mean_cost_normalized_q05"] for row in rows)
    maximum_cost = max(row["exact_mean_cost_normalized_q95"] for row in rows)
    assert all(row["exact_mean_cost_normalized_q05"] > 0 for row in rows)
    axes[0].set_xscale("log")
    axes[0].set_xlim(.8*minimum_cost, 1.25*maximum_cost)
    axes[0].set_xticks([.001, .01, .1], [r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"])
    axes[0].minorticks_off()
    axes[1].set_xlim(-1.05, 1.05)
    axes[0].set_yticklabels(labels_for(rows))
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    axes[0].set_title("Exact noise mean cost\npositive = worse mean gap", pad=8)
    axes[1].set_title("Paired record advantage\npositive = better record", pad=8)
    axes[0].set_xlabel(r"$C_H/(1+V)$"+"\n(log scale)")
    axes[1].set_xlabel(r"$(J_{\rm on}-J_{\rm off})/(1+f_{\min})$"+"\n(signed log scale)")
    row_background(axes[2], len(rows))
    for i, row in enumerate(rows):
        p = row["paired_gain_positive_probability"]
        t = row["paired_gain_tied_probability"]
        n = row["paired_gain_negative_probability"]
        axes[2].barh(i, p, height=.61, color=POSITIVE)
        axes[2].barh(i, t, left=p, height=.61, color=TIE)
        axes[2].barh(i, n, left=p+t, height=.61, color=NEGATIVE)
        assert abs(p+t+n-1) < 1e-12
    axes[2].set_xlim(0, 1)
    axes[2].set_xticks([0, .5, 1], ["0", "0.5", "1"])
    axes[2].set_xlabel("Paired batch fraction")
    axes[2].set_title("Record comparison\nall paired batches", pad=8)
    axes[2].spines["left"].set_visible(False)
    fig.suptitle("Gaussian noise: mean cost and immediate record opportunities", fontsize=10, y=.975)
    fig.text(.5, .94, r"$\tau=0.1$; 128 input laws per row; four S/R settings and both orders equally weighted",
             ha="center", fontsize=7.6)
    quantile_legend(fig, .125)
    fig.legend(handles=[Patch(color=POSITIVE, label="Better record"), Patch(color=TIE, label="Tie"),
                        Patch(color=NEGATIVE, label="Worse record")],
               loc="lower center", bbox_to_anchor=(.54, .079), ncol=3, frameon=False, fontsize=7.2)
    fig.text(.5, .046, "Ranges describe distributions, not confidence intervals. Exact costs pool 1,024 law-setting-order cells;",
             ha="center", fontsize=7)
    fig.text(.5, .027, "record outcomes pool 8,192 paired batches per row (32 candidates each). Configurations share input laws.",
             ha="center", fontsize=7)
    fig.text(.5, .009, r"The signed log axis is linear for $|x|\leq 10^{-3}$.", ha="center", fontsize=7)
    first = save_figure(fig, "exploration_noise")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 7.6), sharey=True)
    fig.subplots_adjust(left=.27, right=.985, top=.858, bottom=.205, wspace=.25)
    all_ranges = []
    for si, switch in enumerate(experiment.SWITCHES):
        selected = [row for row in primary if row["switch"] == switch]
        assert len(selected) == 20
        quantile_panel(axes[si], selected, "paired_record_advantage_normalized", COLORS[switch])
        axes[si].set_title({"H": "Gaussian\nnoise H", "R": "Midpoint\nrecombination R", "S": "Bounded-score\nselection S"}[switch], pad=8)
        axes[si].set_xlabel("Record advantage\n(normalized; signed log)")
        for row in selected:
            all_ranges.extend([row["paired_record_advantage_normalized_"+q] for q in Q_NAMES])
            all_ranges.append(row["paired_record_advantage_normalized_mean"])
    lo, hi = min(0, min(all_ranges)), max(0, max(all_ranges))
    bound = 1.05*max(abs(lo), abs(hi), 1)
    for ax in axes:
        ax.set_xlim(-bound, bound)
    axes[0].set_yticklabels(labels_for(rows))
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    fig.suptitle("Immediate record effects of enabling each component", fontsize=10.2, y=.975)
    fig.text(.5, .94, r"$\tau=0.1$; $(J_{\rm on}-J_{\rm off})/(1+f_{\min})$; positive means a better record",
             ha="center", fontsize=7.7)
    quantile_legend(fig, .077)
    fig.text(.5, .028, "All 20 problem-family cases; four other-switch settings and both orders equally weighted.", ha="center", fontsize=7)
    fig.text(.5, .01, r"8,192 paired batches per row; descriptive quantiles, not confidence intervals; axes linear for $|x|\leq10^{-3}$.", ha="center", fontsize=7)
    second = save_figure(fig, "exploration_components")
    return [first, second]


def main():
    summary_path = INPUT/"exploration_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Wait for the completed full run before reading its data")
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "full" and summary["total_laws"] == 2560
    assert summary["source_sha256"] == SOURCE_SHA256 == experiment.model.sha256(experiment.__file__)
    assert summary["protocol_sha256"] == experiment.model.sha256(experiment.PROTOCOL)
    assert summary["input_arrays_sha256"] == experiment.model.sha256(experiment.INPUT_FOLDER/"canonical_arrays.npz")
    assert summary["input_summary_sha256"] == experiment.model.sha256(experiment.INPUT_FOLDER/"canonical_summary.json")
    assert summary["total_logical_candidate_outcomes"] == 83886080
    assert summary["total_logical_batch_outcomes"] == 2621440
    assert summary["candidates"] == 256 and summary["batch_size"] == 32
    expected = {spec["id"]+"__"+fam for spec in summary["instances"] for fam in experiment.model.LAW_FAMILIES}
    assert set(summary["cases"]) == expected and len(expected) == 20
    detailed, primary, sensitivity, pair_laws, pair_cases, audits = [], [], [], [], [], {}
    with np.load(experiment.INPUT_FOLDER/"canonical_arrays.npz") as saved:
        for j, spec in enumerate(summary["instances"]):
            for family in experiment.model.LAW_FAMILIES:
                key = spec["id"]+"__"+family
                case = summary["cases"][key]
                path = ROOT/case["output_path"]
                assert experiment.model.sha256(path) == case["output_sha256"]
                with np.load(path) as archive:
                    arrays = {name: archive[name] for name in archive.files}
                    audits[key] = audit_case(key, arrays, case, saved)
                    # Exact symmetric-parent benefits are not new-record gains.
                    prefix = "input_recombination_contribution_"
                    pair_mean = arrays[prefix+"mean"]
                    assert np.allclose(pair_mean, -arrays["input_generator_component_actions"][:, 2, 3],
                                       rtol=1e-12, atol=1e-12)
                    pair_identity = dict(instance=spec["id"], short_label=LABELS[j], population_family=family,
                                         convention="benefit: mean parental objective minus midpoint objective")
                    for li in range(128):
                        row = dict(pair_identity, input_law_index=li)
                        for field in ("mean", "positive_mean", "negative_mean", "positive_mass", "negative_mass", "zero_mass"):
                            row[field] = float(arrays[prefix+field][li])
                        row.update({name: float(v) for name, v in zip(Q_NAMES, arrays[prefix+"quantiles"][li])})
                        pair_laws.append(row)
                    row = dict(pair_identity, input_laws=128, ordered_pairs_per_law=1024)
                    for field in ("mean", "positive_mean", "negative_mean", "positive_mass", "negative_mass", "zero_mass"):
                        row["mean_"+field] = float(arrays[prefix+field].mean())
                    row.update(describe("law_mean_benefit", pair_mean))
                    pair_cases.append(row)
                    for si, switch in enumerate(experiment.SWITCHES):
                        enabled = [ai for ai in range(8) if experiment.disabled_index(ai, switch) != ai]
                        assert len(enabled) == 4
                        for ti, tau in enumerate(experiment.model.TAUS):
                            for ai in enabled:
                                for oi, order in enumerate(experiment.model.ORDERS):
                                    identity = dict(instance=spec["id"], short_label=LABELS[j], population_family=family,
                                                    switch=switch, assembly=experiment.model.ASSEMBLIES[ai]["id"],
                                                    order=order, tau=float(tau), input_laws=128)
                                    detailed.append(summarize(collection(arrays, [ai], [oi], ti, si), identity))
                            if ti == 0 or switch == "H":
                                pooled = collection(arrays, enabled, [0, 1], ti, si)
                                identity = dict(instance=spec["id"], short_label=LABELS[j], population_family=family,
                                                switch=switch, tau=float(tau), input_laws=128,
                                                pooled_other_switch_settings=4, pooled_orders=2)
                                row = summarize(pooled, identity)
                                if ti == 0:
                                    primary.append(row)
                                if switch == "H":
                                    sensitivity.append(row)
    assert len(detailed) == 3840 and len(primary) == 60 and len(sensitivity) == 160
    outputs = [("exploration_complete.csv", detailed),
               ("exploration_primary.csv", primary),
               ("exploration_H_steps.csv", sensitivity),
               ("recombination_pair_laws.csv", pair_laws),
               ("recombination_pairs.csv", pair_cases)]
    for name, rows in outputs:
        write_csv(RESULTS/name, rows)
    figure_paths = figures(primary)
    h = [row for row in primary if row["switch"] == "H"]
    findings = {"H_primary_all_cases": {
        "cases": len(h), "law_configuration_order_cells": sum(row["law_configuration_order_cells"] for row in h),
        "paired_batch_outcomes": sum(row["paired_batch_outcomes"] for row in h),
        "positive_mean_cost_cells": sum(row["exact_positive_mean_cost_cells"] for row in h),
        "positive_cost_batch_opportunities": sum(row["positive_cost_batch_opportunities"] for row in h),
        "positive_cost_and_record_advantage_batches": sum(row["positive_cost_and_gain_count"] for row in h),
        "positive_cost_and_noise_only_record_batches": sum(row["positive_cost_and_exclusive_record_count"] for row in h),
        "positive_cost_and_lost_record_batches": sum(row["positive_cost_and_lost_record_count"] for row in h),
        "cases_with_any_positive_cost_and_record_advantage": sum(row["positive_cost_and_gain_count"] > 0 for row in h),
        "cases_with_positive_raw_average_record_advantage": sum(row["paired_record_advantage_mean"] > 0 for row in h),
        "cases_with_positive_normalized_average_record_advantage": sum(row["paired_record_advantage_normalized_mean"] > 0 for row in h),
        "normalization_warning": "Figures display averages after dividing each paired advantage by its law-specific 1+f_min; their signs can differ from raw averages because normalization changes weights across input laws."},
        "primary_case_component_rows": primary,
        "recombination_pair_cases": pair_cases}
    audit = {"status": "passed", "scope": "Complete frozen-law panel; no optimizer or candidate resimulation",
             "analysis_source_sha256": experiment.model.sha256(__file__),
             "input_summary_sha256": experiment.model.sha256(summary_path), "run_source_sha256": SOURCE_SHA256,
             "protocol_sha256": summary["protocol_sha256"], "cases": audits,
             "complete_rows": len(detailed), "primary_rows": len(primary), "step_sensitivity_rows": len(sensitivity),
             "input_laws": 2560, "mapped_configuration_cells": 327680,
             "mapped_candidate_outcomes": 83886080, "mapped_batch_outcomes": 2621440,
             "quantiles": QUANTILES.tolist(), "interval_interpretation": "Descriptive quantiles, not confidence intervals",
             "pooling": "Equal weights for all laws, four enabled-switch contexts and two orders; eight batches per cell",
             "primary_step": .1, "step_sensitivity": experiment.model.TAUS.tolist(),
             "figure_axis_scale": "Primary exact H cost: log (all positive). Record advantages: symlog, linear threshold 1e-3; displayed quantiles and means are not clipped.",
             "mean_cost_normalization": "1+original mean objective V; positive costs worsen the mean",
             "record_advantage_normalization": "1+original best objective f_min; positive advantages improve the record",
             "spread_scope": "Geometric novelty is distance from original atoms, not basin discovery",
             "logical_samples_are_dependent": "Settings/orders/steps paired; batches conditional on same input law",
             "figure_files": figure_paths,
             "data_tables": [{"path": "results/exploration/"+name, "sha256": experiment.model.sha256(RESULTS/name), "rows": len(rows)}
                             for name, rows in outputs], "findings": findings}
    (RESULTS/"exploration_analysis.json").write_text(json.dumps(audit, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": "passed", "tables": [name for name, _ in outputs],
                      "figures": figure_paths, "findings": findings["H_primary_all_cases"]}, indent=2))


if __name__ == "__main__":
    main()
