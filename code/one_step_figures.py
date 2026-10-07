"""Audit and display the complete canonical assembly calibration/holdout matrix."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import one_step_study as model

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["PL 4D", "PL 8D", "Rotated PL 4D", "Rotated PL 8D", "Quartic 4D", "Quartic 8D",
          "Rastrigin 4D", "Rosenbrock 4D", "Unequal wells 4D", "Periodic 1D"]
ASSEMBLY_LABELS = ["D", "D+H", "D+R", "D+R+H", "D+S", "D+S+H", "D+S+R", "D+S+R+H"]
FAMILY_LABELS = {"broad": "Broad populations", "clustered": "Two-cluster populations"}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def audit(calibration, validation, ca, va):
    assert calibration["status"] == "calibration" and validation["status"] == "validation"
    assert calibration["law_count_per_instance_family"] == 32
    assert validation["law_count_per_instance_family"] == 128
    assert calibration["source_sha256"] == validation["source_sha256"] == model.sha256(model.__file__)
    assert calibration["protocol_sha256"] == validation["protocol_sha256"] == model.PROTOCOL_SHA256
    assert calibration["seed_namespace_base"] == model.CALIBRATION_SEED_BASE
    assert validation["seed_namespace_base"] == model.VALIDATION_SEED_BASE
    expected = {spec["id"]+"__"+family for spec in validation["instances"] for family in model.LAW_FAMILIES}
    assert set(calibration["cases"]) == set(validation["cases"]) == expected and len(expected) == 20
    max_additivity_error = 0.
    for summary, arrays, repetitions in ((calibration, ca, 32), (validation, va, 128)):
        for spec in summary["instances"]:
            for family in model.LAW_FAMILIES:
                key = spec["id"]+"__"+family
                assert np.array_equal(model.laws(spec, repetitions, family), arrays[key+"_input_points"])
                V = arrays[key+"_V"]
                assert np.allclose(V, model.objectives.objective(arrays[key+"_input_points"], spec).mean(axis=1), rtol=1e-14, atol=1e-14)
                components = arrays[key+"_component_actions"]
                g = arrays[key+"_generator_action"]
                output = arrays[key+"_output_V"]
                assert components.shape == (repetitions, 8, 4) and output.shape == (repetitions, 8, 2, 8)
                assert np.array_equal(g, components.sum(axis=2))
                q = (output-V[:, None, None, None])/model.TAUS[None, None, None, :]
                rem = q-g[:, :, None, None]
                assert np.array_equal(q, arrays[key+"_difference_quotient"])
                assert np.array_equal(rem, arrays[key+"_signed_weak_remainder"])
                assert np.array_equal(np.abs(rem)/(1+V[:, None, None, None]), arrays[key+"_normalized_weak_remainder"])
                coeff = arrays[key+"_component_effective_decay_coefficients"]
                total = arrays[key+"_effective_decay_coefficient"]
                max_additivity_error = max(max_additivity_error, float(np.nanmax(np.abs(coeff.sum(axis=2)-total))))
                assert np.allclose(coeff.sum(axis=2), total, rtol=1e-12, atol=1e-12, equal_nan=True)
                for ai, assembly in enumerate(model.ASSEMBLIES):
                    if assembly["omega"] == 0 or assembly["gamma"] == 0:
                        assert np.allclose(output[:, ai, 0], output[:, ai, 1], rtol=1e-13, atol=1e-12)
    calibration_raw_margin_count = calibration_resolved_margin_count = 0
    calibration_maximum_margin = 0.
    validation_resolved_margin_count = 0
    for spec in validation["instances"]:
        instance = spec["id"]
        for ai, assembly in enumerate(model.ASSEMBLIES):
            frozen = calibration["empirical_envelopes"][instance][assembly["id"]]
            rates = np.concatenate([ca[instance+"__"+family+"_effective_decay_coefficient"][:, ai]
                                    for family in model.LAW_FAMILIES])
            remainders = np.concatenate([ca[instance+"__"+family+"_normalized_weak_remainder"][:, ai]
                                         for family in model.LAW_FAMILIES])
            assert frozen["lambda_cal"] == float(np.nanmin(rates))
            assert np.array_equal(frozen["a_cal"], remainders.max(axis=0))
            for family in model.LAW_FAMILIES:
                key = instance+"__"+family
                cal_g, cal_V = ca[key+"_generator_action"][:, ai], ca[key+"_V"]
                cal_margin = cal_g+frozen["lambda_cal"]*cal_V
                cal_tolerance = 1e-12*(1+np.abs(cal_g)+np.abs(frozen["lambda_cal"]*cal_V))
                calibration_raw_margin_count += int((cal_margin > 0).sum())
                calibration_resolved_margin_count += int((cal_margin > cal_tolerance).sum())
                calibration_maximum_margin = max(calibration_maximum_margin, float(cal_margin.max()))
                violation = va[key+"_normalized_weak_remainder"][:, ai] > np.asarray(frozen["a_cal"])
                assert np.array_equal(violation, va[key+"_"+assembly["id"]+"_remainder_envelope_violation"])
                g, V = va[key+"_generator_action"][:, ai], va[key+"_V"]
                violation = (V > model.V_GUARD) & (g+frozen["lambda_cal"]*V > 0)
                validation_resolved_margin_count += int((g+frozen["lambda_cal"]*V > 1e-12*(1+np.abs(g)+np.abs(frozen["lambda_cal"]*V))).sum())
                assert np.array_equal(violation, va[key+"_"+assembly["id"]+"_drift_envelope_violation"])
    return {"complete_instance_family_cases": 20, "assemblies_per_case": 8, "orders": 2, "step_sizes": 8,
            "calibration_laws": 640, "validation_laws": 2560,
            "maximum_component_coefficient_sum_error": max_additivity_error,
            "calibration_raw_positive_drift_margins": calibration_raw_margin_count,
            "calibration_positive_drift_margins_above_roundoff": calibration_resolved_margin_count,
            "calibration_maximum_positive_drift_margin": calibration_maximum_margin,
            "validation_drift_violations_above_roundoff": validation_resolved_margin_count,
            "paired_law_seeds_and_remainders_verified": True,
            "pooled_calibration_envelopes_independently_recomputed": True,
            "all_heldout_violation_flags_recomputed": True}


def tables(summary, arrays):
    assembly_rows, remainder_rows, component_rows = [], [], []
    for spec in summary["instances"]:
        for family in model.LAW_FAMILIES:
            key = spec["id"]+"__"+family
            case = summary["cases"][key]
            for ci, component in enumerate(model.COMPONENTS):
                values = arrays[key+"_component_effective_decay_coefficients"][:, -1, ci]
                component_rows.append({"instance": spec["id"], "population_family": family, "component": component,
                                       "coefficient_definition": "-G_component(f)/V", "laws": len(values),
                                       "mean": float(np.nanmean(values)), "q10": float(np.nanquantile(values, .1)),
                                       "median": float(np.nanmedian(values)), "q90": float(np.nanquantile(values, .9)),
                                       "minimum": float(np.nanmin(values)), "maximum": float(np.nanmax(values)),
                                       "negative_coefficient_count": int((values < 0).sum())})
            for ai, assembly in enumerate(model.ASSEMBLIES):
                row = case["assemblies"][assembly["id"]]
                holdout = row["calibration_envelope_validation"]
                rates = arrays[key+"_effective_decay_coefficient"][:, ai]
                assembly_rows.append({"instance": spec["id"], "population_family": family,
                                      "assembly": assembly["id"], "assembly_label": ASSEMBLY_LABELS[ai],
                                      "omega": assembly["omega"], "gamma": assembly["gamma"], "sigma": assembly["sigma"],
                                      "laws": len(rates), "lambda_cal_both_families": holdout["lambda_cal"],
                                      "lambda_holdout_min": float(np.nanmin(rates)), "lambda_holdout_mean": float(np.nanmean(rates)),
                                      "lambda_holdout_q10": float(np.nanquantile(rates, .1)),
                                      "lambda_holdout_median": float(np.nanmedian(rates)),
                                      "lambda_holdout_q90": float(np.nanquantile(rates, .9)),
                                      "positive_generator_count": row["positive_generator_count"],
                                      "drift_envelope_violations": holdout["drift_violations"],
                                      "guarded_ratios": case["V_guard_count"]})
                for oi, order in enumerate(model.ORDERS):
                    for ti, tau in enumerate(model.TAUS):
                        values = arrays[key+"_normalized_weak_remainder"][:, ai, oi, ti]
                        remainder_rows.append({"instance": spec["id"], "population_family": family,
                                               "assembly": assembly["id"], "order": order, "tau": tau,
                                               "laws": len(values), "a_cal_both_families": holdout["a_cal"][oi][ti],
                                               "holdout_maximum": float(values.max()), "holdout_median": float(np.median(values)),
                                               "heldout_envelope_violations": holdout["remainder_violations"][oi][ti],
                                               "generator_finite_step_sign_disagreements": row["resolved_finite_step_sign_disagreements"][oi][ti]})
    write_csv(ROOT/"results/one-step/one_step_assembly_summary.csv", assembly_rows)
    write_csv(ROOT/"results/one-step/one_step_remainder_holdout.csv", remainder_rows)
    write_csv(ROOT/"results/one-step/one_step_component_rates.csv", component_rows)
    return {"assembly_rows": len(assembly_rows), "remainder_rows": len(remainder_rows), "component_rows": len(component_rows)}


def consistency_figure(summary, arrays):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    colors = plt.get_cmap("tab10").colors
    reductions = []
    for column, family in enumerate(model.LAW_FAMILIES):
        for j, spec in enumerate(summary["instances"]):
            key = spec["id"]+"__"+family
            remainder = arrays[key+"_normalized_weak_remainder"].max(axis=(0, 1, 2))
            ordering = arrays[key+"_normalized_order_quotient"].max(axis=(0, 1))
            axes[0, column].loglog(model.TAUS, remainder, color=colors[j], marker="o", markersize=2.8, label=LABELS[j])
            axes[1, column].loglog(model.TAUS, ordering, color=colors[j], marker="o", markersize=2.8)
            reductions.append({"instance": spec["id"], "family": family,
                               "weak_remainder_reduction": float(remainder[0]/remainder[-1]),
                               "order_quotient_reduction": float(ordering[0]/ordering[-1]),
                               "weak_remainder_decreases_at_every_refinement": bool((np.diff(remainder) < 0).all()),
                               "order_quotient_decreases_at_every_refinement": bool((np.diff(ordering) < 0).all())})
        axes[0, column].set_title(FAMILY_LABELS[family])
        axes[1, column].set_xlabel(r"Macro-step $\tau$")
        for row in range(2):
            axes[row, column].grid(which="both", alpha=.2)
    axes[0, 0].set_ylabel("Normalized weak remainder")
    axes[1, 0].set_ylabel(r"Order difference / $[\tau(1+V)]$")
    axes[0, 0].text(.025, .94, r"$|(V_\tau-V)/\tau-g|/(1+V)$", transform=axes[0, 0].transAxes, fontsize=9, va="top")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="lower center", bbox_to_anchor=(.54, .043), frameon=False, fontsize=8.1)
    fig.text(.54, .012, "Each curve is a maximum over 128 held-out laws and all eight assemblies (top: both orders).\n"
             "Exact finite-law expectations; these sampled-law maxima are not uniform bounds.", ha="center", fontsize=8)
    fig.subplots_adjust(left=.1, right=.99, top=.95, bottom=.2, hspace=.21, wspace=.22)
    for suffix in ("pdf", "png"):
        fig.savefig(ROOT/f"figures/one_step_consistency.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return reductions


def coefficient_figure(summary, arrays):
    component_data, total_data = [], []
    for family in model.LAW_FAMILIES:
        component_data.append(np.stack([np.nanmedian(arrays[spec["id"]+"__"+family+"_component_effective_decay_coefficients"][:, -1], axis=0)
                                        for spec in summary["instances"]]))
        total_data.append(np.stack([np.nanmedian(arrays[spec["id"]+"__"+family+"_effective_decay_coefficient"], axis=0)
                                    for spec in summary["instances"]]))
    maximum = max(np.max(np.abs(a)) for a in component_data+total_data)
    limit = 2**np.ceil(np.log2(maximum))
    norm = SymLogNorm(linthresh=.05, linscale=.7, vmin=-limit, vmax=limit, base=10)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.7), gridspec_kw={"width_ratios": [1, 1.6]})
    for row, family in enumerate(model.LAW_FAMILIES):
        for column, data in enumerate((component_data[row], total_data[row])):
            ax = axes[row, column]
            display = ax.imshow(data, cmap="RdBu", norm=norm, aspect="auto")
            ax.set_yticks(range(10), LABELS if column == 0 else [])
            ax.set_xticks(range(data.shape[1]), ["Drift D", "Diffusion H", "Selection S", "Recomb. R"] if column == 0 else ASSEMBLY_LABELS)
            ax.tick_params(axis="x", rotation=30, labelsize=8.5)
            ax.tick_params(axis="y", labelsize=9)
            for (i, j), value in np.ndenumerate(data):
                # Exact values and quantiles are in the accompanying full CSV.
                ax.text(j, i, "+" if value > 0 else "-" if value < 0 else "0", ha="center", va="center",
                        fontsize=11, color="white" if abs(norm(value)-.5) > .29 else "black")
            ax.set_xticks(np.arange(-.5, data.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=.6)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.set_title(FAMILY_LABELS[family]+(" - components" if column == 0 else " - complete assemblies"), fontsize=11)
    color_axis = fig.add_axes([.30, .135, .61, .022])
    cbar = fig.colorbar(display, cax=color_axis, orientation="horizontal")
    cbar.set_ticks([-16, -4, -1, -.1, 0, .1, 1, 4, 16], labels=["-16", "-4", "-1", "-0.1", "0", "0.1", "1", "4", "16"])
    cbar.set_label(r"Median signed coefficient $-G_j(f)/V$ (components) or $-g/V$ (assemblies)", fontsize=9)
    fig.text(.57, .012, "Positive (blue) means instantaneous gap reduction; negative (red) means gap increase.\n"
             "D is always present; H uses noise scale 0.2. Medians need not add: generator contributions add law by law.\n"
             "Both population geometries and every assembly are retained; component 10th/90th percentiles are saved in the CSV.",
             ha="center", fontsize=8.3)
    fig.subplots_adjust(left=.14, right=.99, top=.955, bottom=.25, wspace=.10, hspace=.39)
    for suffix in ("pdf", "png"):
        fig.savefig(ROOT/f"figures/one_step_components.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    paths = {phase: ROOT/f"results/one-step/canonical-{phase}" for phase in ("calibration", "validation")}
    calibration = json.loads((paths["calibration"]/"canonical_summary.json").read_text())
    validation = json.loads((paths["validation"]/"canonical_summary.json").read_text())
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    with np.load(paths["calibration"]/"canonical_arrays.npz") as ca, np.load(paths["validation"]/"canonical_arrays.npz") as va:
        checks = audit(calibration, validation, ca, va)
        checks["csv_rows"] = tables(validation, va)
        checks["refinement_summary"] = consistency_figure(validation, va)
        coefficient_figure(validation, va)
        checks["positive_generator_actions"] = sum(row["positive_generator_count"] for case in validation["cases"].values() for row in case["assemblies"].values())
        checks["assembly_law_actions"] = 20*128*8
        checks["resolved_finite_step_sign_disagreements"] = sum(int(va[key+"_resolved_finite_step_sign_disagreement"].sum()) for key in validation["cases"])
        checks["drift_envelope_violations"] = sum(row["drift_violations"] for case in validation["pooled_validation_violations"].values() for row in case.values())
        checks["remainder_envelope_violations_by_order_step"] = np.sum([row["remainder_violations"] for case in validation["pooled_validation_violations"].values() for row in case.values()], axis=0).tolist()
        checks["cutoff_affected_support_points"] = sum(case["cutoff_affected_support_points"] for case in validation["cases"].values())
    checks["artifact_hashes"] = {phase: {name: model.sha256(path/name) for name in ("canonical_summary.json", "canonical_arrays.npz")}
                                for phase, path in paths.items()}
    checks["figure_source_sha256"] = model.sha256(__file__)
    checks["interpretation"] = "Frozen-law empirical diagnostics; no uniform certificate or optimizer performance claim"
    (ROOT/"results/one-step/one_step_figure_audit.json").write_text(json.dumps(checks, indent=2)+"\n")
    print(json.dumps({k: v for k, v in checks.items() if k not in ("refinement_summary", "artifact_hashes")}, indent=2))


if __name__ == "__main__":
    main()
