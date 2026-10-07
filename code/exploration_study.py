"""Paired, exploration-aware one-step diagnostics for the frozen one-step laws.

These are conditional candidate distributions, not optimizer trajectories.
The full run must follow a committed protocol and source checkpoint.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import platform
import time
import numpy as np
from scipy.spatial import cKDTree
import scipy
import one_step_study as model


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/exploration-diagnostics.md"
PROTOCOL_SHA256 = "f6fba8a292a3f627aee40c4faadc87d6a79073a4b8ae9fdc0d7536485eeec28a"
MODEL_SHA256 = "ff4404c530be6691d5f2c8402b8f3479efc68169f078d9f96042c45918035b1d"
INPUT_FOLDER = ROOT / "results/one-step/canonical-validation"
INPUT_SUMMARY_SHA256 = "f4a10e06d8fb7eab5a8955be6756bdf24cf700464ced03fe9261172ab8c9bce2"
INPUT_ARRAYS_SHA256 = "2d92da60892b1f1a9de42ae8e4fa359c19523d94dd0abd9b50afde4386bc2c6f"
RUN_SEED_BASE = 2072600000
SMOKE_SEED_BASE = 2172600000
CANDIDATES = 256
BATCH_SIZE = 32
BATCHES = CANDIDATES // BATCH_SIZE
QUANTILES = np.array([.05, .25, .50, .75, .95])
RECORD_TOLERANCE = 1e-12
SCALE_GUARD = 1e-14
SWITCHES = ("H", "R", "S")
COMPONENTS = ("selection", "recombination", "drift", "diffusion")
PHASES = ("input", "first_SR_stage", "second_SR_stage", "drift", "noise")
PRELAWS = ("mu", "S_mu", "R_mu", "RS_mu", "SR_mu")


def digest_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def candidate_seed(instance_index, family, law_index, smoke=False):
    base = SMOKE_SEED_BASE if smoke else RUN_SEED_BASE
    return base + 100000*instance_index + 10000*(family == "clustered") + law_index


def frozen_checks():
    assert model.sha256(model.__file__) == MODEL_SHA256
    model.frozen_input_checks()
    assert model.sha256(PROTOCOL) == PROTOCOL_SHA256, "Protocol changed: freeze the amended protocol before running"
    assert model.sha256(INPUT_FOLDER / "canonical_summary.json") == INPUT_SUMMARY_SHA256
    assert model.sha256(INPUT_FOLDER / "canonical_arrays.npz") == INPUT_ARRAYS_SHA256


def all_preweights(weights, f, tau):
    """Five distinct pre-mutation laws; all assembly/order cells are retained."""
    n = len(weights)
    phi = f/(1+f)
    selected = weights*np.exp(-tau*phi[:n])
    selected /= selected.sum()
    pair = np.outer(weights, weights).ravel()
    selected_pair = np.outer(selected, selected).ravel()
    base = np.r_[weights, np.zeros(n*n)]
    S = np.r_[selected, np.zeros(n*n)]
    R = np.r_[(1-tau)*weights, tau*pair]
    RS = np.r_[(1-tau)*selected, tau*selected_pair]
    SR = R*np.exp(-tau*phi)
    SR /= SR.sum()
    return np.stack([base, S, R, RS, SR])


def prelaw_index(assembly, order_index):
    omega, gamma = assembly["omega"], assembly["gamma"]
    if not gamma:
        return int(omega)
    if not omega:
        return 2
    return 3 + order_index


def inverse_cdf(weights, uniforms):
    active = np.flatnonzero(weights > 0)
    cdf = np.cumsum(weights[active])
    cdf /= cdf[-1]
    cdf[-1] = 1.
    result = active[np.searchsorted(cdf, uniforms, side="right")]
    assert ((result >= 0) & (result < len(weights))).all()
    assert (weights[result] > 0).all()
    return result


def record_data(values, best):
    tolerance = RECORD_TOLERANCE*(1+best)
    finite = np.isfinite(values)
    records = finite & (best-values > tolerance)
    gain = np.where(finite, np.maximum(best-values, 0.), 0.)
    material_eligible = best > tolerance
    material = records & (gain >= .1*best) if material_eligible else np.zeros(values.shape, dtype=bool)
    return records, gain, material


def total_variance(points, weights):
    center = weights @ points
    return float(np.sum(weights*np.sum((points-center)**2, axis=-1)))


def batched_variance(points):
    """Unbiased sample total variance within each independent candidate batch."""
    values = points.reshape(BATCHES, BATCH_SIZE, -1)
    return np.sum((values-values.mean(axis=1, keepdims=True))**2, axis=(1, 2))/(BATCH_SIZE-1)


def finite_quantiles(values):
    finite = values[np.isfinite(values)]
    return np.quantile(finite, QUANTILES) if len(finite) else np.full(len(QUANTILES), np.nan)


def weighted_quantiles(values, weights):
    order = np.argsort(values, kind="stable")
    cdf = np.cumsum(weights[order])
    cdf[-1] = 1.
    return values[order[np.searchsorted(cdf, QUANTILES, side="left")]]


def outcome_statistics(values, points, distance, best, V, input_scale):
    records, gain, material = record_data(values, best)
    finite = np.isfinite(values)
    batches = values.reshape(BATCHES, BATCH_SIZE)
    minima = np.min(batches, axis=1)
    batch_record, batch_gain, batch_material = record_data(minima, best)
    scale_valid = np.isfinite(input_scale) and input_scale > SCALE_GUARD
    ratios = distance/input_scale if scale_valid else np.full(distance.shape, np.nan)
    novel = distance > .5*input_scale if scale_valid else np.zeros(distance.shape, dtype=bool)
    signed_change = values-V
    result = {
        "candidate_record_count": np.count_nonzero(records),
        "candidate_material_record_count": np.count_nonzero(material),
        "candidate_improvement_sum": gain.sum(),
        "candidate_improvement_zero_count": np.count_nonzero(gain == 0),
        "candidate_novel_count": np.count_nonzero(novel),
        "candidate_record_and_novel_count": np.count_nonzero(records & novel),
        "candidate_nonfinite_count": np.count_nonzero(~finite),
        "candidate_objective_quantiles": finite_quantiles(values),
        "candidate_signed_gap_change_quantiles": finite_quantiles(signed_change),
        "candidate_normalized_gap_change_quantiles": finite_quantiles(signed_change/(1+V)),
        "candidate_improvement_quantiles": finite_quantiles(gain),
        "candidate_distance_ratio_quantiles": finite_quantiles(ratios),
        "candidate_distance_ratio_mean": float(np.mean(ratios)),
        "candidate_signed_gap_change_negative_count": np.count_nonzero(signed_change < 0),
        "candidate_signed_gap_change_zero_count": np.count_nonzero(signed_change == 0),
        "candidate_signed_gap_change_positive_count": np.count_nonzero(signed_change > 0),
        "batch_minimum_objective": minima,
        "batch_record": batch_record,
        "batch_record_gain": batch_gain,
        "batch_material_record": batch_material,
        "batch_mean_gap_change": batches.mean(axis=1)-V,
        "batch_normalized_mean_gap_change": (batches.mean(axis=1)-V)/(1+V),
        "batch_candidate_record_count": records.reshape(BATCHES, BATCH_SIZE).sum(axis=1),
        "batch_candidate_novel_count": novel.reshape(BATCHES, BATCH_SIZE).sum(axis=1),
        "batch_candidate_record_and_novel_count": (records & novel).reshape(BATCHES, BATCH_SIZE).sum(axis=1),
        "batch_mean_candidate_improvement": gain.reshape(BATCHES, BATCH_SIZE).mean(axis=1),
        "batch_distance_ratio_mean": ratios.reshape(BATCHES, BATCH_SIZE).mean(axis=1),
        "batch_sample_total_variance": batched_variance(points),
    }
    assert (result["batch_record_gain"] >= 0).all() and (gain >= 0).all()
    return result


def disabled_index(assembly_index, switch):
    assembly = dict(model.ASSEMBLIES[assembly_index])
    field = {"H": "sigma", "R": "gamma", "S": "omega"}[switch]
    if not assembly[field]:
        return assembly_index
    assembly[field] = 0
    return next(i for i, other in enumerate(model.ASSEMBLIES)
                if all(assembly[name] == other[name] for name in ("omega", "gamma", "sigma")))


def one_law(points, weights, spec, seed, reference_output):
    """Compute exact means and paired candidate outcomes at one frozen law."""
    cache = model.cached_support(points[None], spec)
    support, f, drift = (cache[name][0] for name in ("support", "f", "drift"))
    n, d = points.shape
    V = float(weights @ f[:n])
    best = float(f[:n].min())
    tree = cKDTree(points)
    input_scale = float(np.median(tree.query(points, k=2)[0][:, 1]))
    input_variance = total_variance(points, weights)
    rng = np.random.default_rng(seed)
    uniforms = rng.random(CANDIDATES)
    normals = rng.normal(size=(CANDIDATES, d))
    shape = (len(model.ASSEMBLIES), len(model.ORDERS), len(model.TAUS))
    exact_phases = np.empty(shape+(len(PHASES),))
    exact_components = np.empty(shape+(len(COMPONENTS),))
    exact_variance = np.empty(shape)
    candidate_values = np.empty(shape+(CANDIDATES,))
    raw = {}
    max_mass_error = 0.
    parent_digest = hashlib.sha256()
    for ti, tau in enumerate(model.TAUS):
        preweights = all_preweights(weights, f, tau)
        max_mass_error = max(max_mass_error, float(np.max(np.abs(preweights.sum(axis=1)-1))))
        shifted = support+tau*drift
        deterministic_f = model.objectives.objective(shifted, spec)
        gaussian_f = model.gaussian_objective(shifted, tau*.2**2, spec)
        deterministic_distance = tree.query(shifted)[0]
        expected_pre = preweights @ f
        expected_drift = preweights @ deterministic_f
        expected_noise = preweights @ gaussian_f
        sampled = {}
        for pi in range(len(PRELAWS)):
            indices = inverse_cdf(preweights[pi], uniforms)
            parent_digest.update(indices.astype(np.uint16).tobytes())
            deterministic_points = shifted[indices]
            noisy_points = deterministic_points+.2*np.sqrt(tau)*normals
            noisy_f = model.objectives.objective(noisy_points, spec)
            noisy_distance = tree.query(noisy_points)[0]
            sampled[pi] = ((deterministic_f[indices], deterministic_points, deterministic_distance[indices]),
                           (noisy_f, noisy_points, noisy_distance))
        for ai, assembly in enumerate(model.ASSEMBLIES):
            hi = int(assembly["sigma"] > 0)
            for oi in range(len(model.ORDERS)):
                pi = prelaw_index(assembly, oi)
                # Both noise switches reuse the identical inverse-CDF support
                # index and deterministic drift point. S/R are not resampled.
                values, candidate_points, distance = sampled[pi][hi]
                candidate_values[ai, oi, ti] = values
                final = expected_noise[pi] if hi else expected_drift[pi]
                if oi == 0:
                    first = expected_pre[1] if assembly["omega"] else V
                    second = expected_pre[pi] if assembly["gamma"] else first
                else:
                    first = expected_pre[2] if assembly["gamma"] else V
                    second = expected_pre[pi] if assembly["omega"] else first
                phases = np.array([V, first, second, expected_drift[pi], final])
                changes = np.diff(phases)
                exact_phases[ai, oi, ti] = phases
                exact_components[ai, oi, ti] = changes if oi == 0 else changes[[1, 0, 2, 3]]
                exact_variance[ai, oi, ti] = total_variance(shifted, preweights[pi])+d*tau*assembly["sigma"]**2
                stats = outcome_statistics(values, candidate_points, distance, best, V, input_scale)
                for name, value in stats.items():
                    value = np.asarray(value)
                    if name not in raw:
                        raw[name] = np.empty(shape+value.shape, dtype=value.dtype)
                    raw[name][ai, oi, ti] = value
    mean_error = float(np.max(np.abs(exact_phases[..., -1]-reference_output)/(1+np.abs(reference_output))))
    telescope_error = float(np.max(np.abs(exact_components.sum(axis=-1)-(exact_phases[..., -1]-V))/(1+V)))
    assert max_mass_error < 1e-12
    assert mean_error < 1e-12, (spec["id"], seed, mean_error)
    assert telescope_error < 1e-12
    exact_mean = exact_phases[..., -1]
    paired_cost = np.zeros(shape+(len(SWITCHES),))
    paired_gain = np.zeros(shape+(len(SWITCHES), BATCHES))
    paired_mean = np.zeros_like(paired_gain)
    paired_exclusive = np.zeros_like(paired_gain, dtype=bool)
    paired_lost = np.zeros_like(paired_gain, dtype=bool)
    paired_candidate_exclusive = np.zeros(shape+(len(SWITCHES),), dtype=np.int64)
    paired_candidate_lost = np.zeros_like(paired_candidate_exclusive)
    paired_candidate_quantiles = np.zeros(shape+(len(SWITCHES), len(QUANTILES)))
    paired_candidate_sign_counts = np.zeros(shape+(len(SWITCHES), 3), dtype=np.int64)
    paired_candidate_positive_sum = np.zeros(shape+(len(SWITCHES),))
    paired_candidate_negative_sum = np.zeros_like(paired_candidate_positive_sum)
    paired_batch_gain_quantiles = np.zeros_like(paired_candidate_quantiles)
    paired_enabled = np.zeros((len(model.ASSEMBLIES), len(SWITCHES)), dtype=bool)
    exact_cost_positive = np.zeros(shape+(len(SWITCHES),), dtype=bool)
    eta = RECORD_TOLERANCE*(1+best)
    for ai in range(len(model.ASSEMBLIES)):
        for si, switch in enumerate(SWITCHES):
            other = disabled_index(ai, switch)
            paired_enabled[ai, si] = other != ai
            paired_cost[ai, ..., si] = exact_mean[ai]-exact_mean[other]
            paired_gain[ai, ..., si, :] = raw["batch_record_gain"][ai]-raw["batch_record_gain"][other]
            paired_mean[ai, ..., si, :] = raw["batch_mean_gap_change"][ai]-raw["batch_mean_gap_change"][other]
            paired_exclusive[ai, ..., si, :] = raw["batch_record"][ai] & ~raw["batch_record"][other]
            paired_lost[ai, ..., si, :] = ~raw["batch_record"][ai] & raw["batch_record"][other]
            yes, _, _ = record_data(candidate_values[ai], best)
            no, _, _ = record_data(candidate_values[other], best)
            paired_candidate_exclusive[ai, ..., si] = (yes & ~no).sum(axis=-1)
            paired_candidate_lost[ai, ..., si] = (~yes & no).sum(axis=-1)
            paired_candidate_quantiles[ai, ..., si, :] = np.quantile(candidate_values[ai]-candidate_values[other], QUANTILES, axis=-1).transpose(1, 2, 0)
            difference = candidate_values[ai]-candidate_values[other]
            paired_candidate_positive_sum[ai, ..., si] = np.maximum(difference, 0).sum(axis=-1)
            paired_candidate_negative_sum[ai, ..., si] = np.minimum(difference, 0).sum(axis=-1)
            paired_candidate_sign_counts[ai, ..., si, 0] = (difference < -eta).sum(axis=-1)
            paired_candidate_sign_counts[ai, ..., si, 1] = (np.abs(difference) <= eta).sum(axis=-1)
            paired_candidate_sign_counts[ai, ..., si, 2] = (difference > eta).sum(axis=-1)
            paired_batch_gain_quantiles[ai, ..., si, :] = np.quantile(paired_gain[ai, ..., si, :], QUANTILES, axis=-1).transpose(1, 2, 0)
            exact_cost_positive[ai, ..., si] = paired_cost[ai, ..., si] > eta
            if other == ai:
                assert not np.any(paired_cost[ai, ..., si]) and not np.any(paired_gain[ai, ..., si, :])
                assert not np.any(paired_exclusive[ai, ..., si, :]) and not np.any(paired_lost[ai, ..., si, :])
                assert not np.any(paired_candidate_positive_sum[ai, ..., si]) and not np.any(paired_candidate_negative_sum[ai, ..., si])
    assert np.allclose((paired_candidate_positive_sum+paired_candidate_negative_sum)/CANDIDATES,
                       paired_mean.mean(axis=-1), rtol=1e-11, atol=1e-12*(1+V))
    pair_weights = np.outer(weights, weights).ravel()
    recombination_contribution = (.5*(f[:n, None]+f[None, :n])).ravel()-f[n:]
    recombination_mean = float(pair_weights @ recombination_contribution)
    assert np.isclose(recombination_mean, V-pair_weights@f[n:], rtol=1e-12, atol=1e-12)
    raw.update(input_points=points, input_weights=weights, V=np.asarray(V), input_best=np.asarray(best),
               candidate_seed=np.asarray(seed, dtype=np.int64), uniform_digest=np.asarray(digest_array(uniforms)),
               normal_digest=np.asarray(digest_array(normals)), parent_index_digest=np.asarray(parent_digest.hexdigest()),
               input_median_nearest_neighbor_distance=np.asarray(input_scale),
               input_scale_guard=np.asarray(not np.isfinite(input_scale) or input_scale <= SCALE_GUARD),
               material_record_guard=np.asarray(best <= RECORD_TOLERANCE*(1+best)),
               input_total_variance=np.asarray(input_variance), exact_phase_means=exact_phases,
               exact_component_mean_changes=exact_components, exact_total_variance=exact_variance,
               exact_total_variance_change=exact_variance-input_variance,
               exact_output_mean=exact_mean, exact_noise_cost=exact_components[..., 3],
               paired_switch_enabled=paired_enabled, paired_exact_mean_cost=paired_cost,
               paired_exact_mean_cost_positive=exact_cost_positive,
               paired_batch_record_gain_difference=paired_gain, paired_batch_mean_gap_difference=paired_mean,
               paired_batch_exclusive_record=paired_exclusive, paired_batch_lost_record=paired_lost,
               paired_candidate_exclusive_record_count=paired_candidate_exclusive,
               paired_candidate_lost_record_count=paired_candidate_lost,
               paired_candidate_objective_difference_quantiles=paired_candidate_quantiles,
               paired_candidate_objective_difference_sign_counts=paired_candidate_sign_counts,
               paired_candidate_objective_difference_positive_sum=paired_candidate_positive_sum,
               paired_candidate_objective_difference_negative_signed_sum=paired_candidate_negative_sum,
               paired_batch_record_gain_difference_quantiles=paired_batch_gain_quantiles,
               paired_batch_record_gain_positive_count=(paired_gain > eta).sum(axis=-1),
               paired_batch_record_gain_negative_count=(paired_gain < -eta).sum(axis=-1),
               paired_batch_record_gain_tied_count=(np.abs(paired_gain) <= eta).sum(axis=-1),
               paired_batch_record_gain_exact_zero_count=(paired_gain == 0).sum(axis=-1),
               paired_positive_cost_and_batch_gain=(exact_cost_positive[..., None] & (paired_gain > eta)),
               paired_positive_cost_and_exclusive_record=(exact_cost_positive[..., None] & paired_exclusive),
               input_recombination_contribution_quantiles=weighted_quantiles(recombination_contribution, pair_weights),
               input_recombination_contribution_mean=np.asarray(recombination_mean),
               input_recombination_contribution_positive_mean=np.asarray(pair_weights@np.maximum(recombination_contribution, 0)),
               input_recombination_contribution_negative_mean=np.asarray(pair_weights@np.minimum(recombination_contribution, 0)),
               input_recombination_contribution_positive_mass=np.asarray(pair_weights@(recombination_contribution > 0)),
               input_recombination_contribution_negative_mass=np.asarray(pair_weights@(recombination_contribution < 0)),
               input_recombination_contribution_zero_mass=np.asarray(pair_weights@(recombination_contribution == 0)),
               cutoff_affected_support_count=np.asarray(np.count_nonzero(cache["cutoff"][0] < 1)),
               nonfinite_exact_value_count=np.asarray(np.count_nonzero(~np.isfinite(exact_phases))+np.count_nonzero(~np.isfinite(exact_variance))),
               maximum_one_step_mean_relative_error=np.asarray(mean_error), maximum_telescoping_error=np.asarray(telescope_error),
               maximum_mass_error=np.asarray(max_mass_error))
    for ai, assembly in enumerate(model.ASSEMBLIES):
        for ci, enabled in ((0, assembly["omega"]), (1, assembly["gamma"]), (3, assembly["sigma"])):
            if not enabled:
                assert np.all(exact_components[ai, ..., ci] == 0)
    return raw


def seed_checks():
    namespaces = {}
    for name, smoke in (("run", False), ("smoke", True)):
        values = {candidate_seed(j, family, r, smoke) for j in range(10)
                  for family in model.LAW_FAMILIES for r in range(128)}
        assert len(values) == 2560
        namespaces[name] = values
    assert namespaces["run"].isdisjoint(namespaces["smoke"])
    old = {base+1000*j+offset+r for base in (model.CALIBRATION_SEED_BASE, model.VALIDATION_SEED_BASE, model.SMOKE_SEED_BASE)
           for j in range(10) for offset in model.FAMILY_SEED_OFFSETS.values() for r in range(128)}
    assert all(values.isdisjoint(old) for values in namespaces.values())
    assert min(namespaces["run"]) > 1503919000  # All earlier restart namespaces.
    return {name: [min(values), max(values)] for name, values in namespaces.items()}


def implementation_checks(summary, saved):
    """Small deterministic checks; never select parameters using their outcomes."""
    ranges = seed_checks()
    spec = summary["instances"][0]
    key = spec["id"]+"__broad"
    points, weights = saved[key+"_input_points"][0], saved[key+"_input_weights"][0]
    cache = model.cached_support(points[None], spec)
    _, reference = model.operator_diagnostics(cache, spec, weights[None])
    assert np.allclose(reference["output_V"][0], saved[key+"_output_V"][0], rtol=1e-13, atol=1e-12)
    first = one_law(points, weights, spec, candidate_seed(0, "broad", 0, True), reference["output_V"][0])
    second = one_law(points, weights, spec, candidate_seed(0, "broad", 0, True), reference["output_V"][0])
    assert all(np.array_equal(first[name], second[name], equal_nan=True) if first[name].dtype.kind in "fc"
               else np.array_equal(first[name], second[name]) for name in first)
    f = cache["f"][0]
    for tau in model.TAUS:
        pre = all_preweights(weights, f, tau)
        assert (pre >= 0).all() and np.allclose(pre.sum(axis=1), 1., rtol=0, atol=1e-14)
        u = np.array([0., .2, np.nextafter(1., 0.)])
        for w in pre:
            idx = inverse_cdf(w, u)
            assert np.all((0 <= idx) & (idx < len(w)))
        for ai, assembly in enumerate(model.ASSEMBLIES):
            if not assembly["omega"] or not assembly["gamma"]:
                assert prelaw_index(assembly, 0) == prelaw_index(assembly, 1)
            off = disabled_index(ai, "H")
            for oi in range(2):
                assert prelaw_index(assembly, oi) == prelaw_index(model.ASSEMBLIES[off], oi)
    testvalues = np.array([1., 1.-.5e-12, 1.-3e-12, .8, np.inf, np.nan])
    records, gains, material = record_data(testvalues, 1.)
    assert np.array_equal(records, [False, False, True, True, False, False])
    assert gains[1] > 0 and not records[1] and gains[0] == 0 and material.sum() == 1
    assert not record_data(np.array([0., -1e-15]), 0.)[0].any()
    assert not record_data(np.array([0., -1e-15]), 0.)[2].any()
    guarded = outcome_statistics(np.zeros(CANDIDATES), np.zeros((CANDIDATES, 2)),
                                 np.zeros(CANDIDATES), 0., 0., 0.)
    assert np.isnan(guarded["candidate_distance_ratio_quantiles"]).all()
    assert guarded["candidate_novel_count"] == guarded["candidate_record_count"] == 0
    synthetic = np.arange(CANDIDATES*2).reshape(CANDIDATES, 2)/100.
    variance = batched_variance(synthetic)
    assert np.allclose(variance, synthetic.reshape(BATCHES, BATCH_SIZE, 2).var(axis=1, ddof=1).sum(axis=1))
    assert np.allclose(total_variance(points, weights), np.trace(np.cov(points.T, aweights=weights, ddof=0)))
    return {"seed_ranges": ranges, "seed_namespaces_disjoint": True, "repeatability": "bitwise passed",
            "support_and_both_order_weights": "passed", "same_parent_H_coupling": "passed",
            "record_tolerance_and_zero_reference_guards": "passed", "variance_identities": "passed",
            "maximum_one_step_mean_relative_error": float(first["maximum_one_step_mean_relative_error"]),
            "maximum_telescoping_error": float(first["maximum_telescoping_error"])}


def run_panel(summary, saved, repetitions, smoke):
    folder = ROOT / "results/exploration" / ("exploration-smoke" if smoke else "exploration-full")
    if smoke and folder.exists():
        folder = ROOT / "results/exploration" / ("exploration-smoke-"+model.sha256(__file__)[:12])
    if (folder / "exploration_summary.json").exists() or (not smoke and folder.exists()):
        raise FileExistsError("Preserving prior outcomes: "+str(folder))
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cases = {}
    for j, spec in enumerate(summary["instances"]):
        for family in model.LAW_FAMILIES:
            key = spec["id"]+"__"+family
            points, weights = saved[key+"_input_points"][:repetitions], saved[key+"_input_weights"][:repetitions]
            reference = saved[key+"_output_V"][:repetitions]
            rows = [one_law(points[r], weights[r], spec, candidate_seed(j, family, r, smoke), reference[r])
                    for r in range(repetitions)]
            arrays = {name: np.stack([row[name] for row in rows]) for name in rows[0]}
            arrays["input_generator_component_actions"] = saved[key+"_component_actions"][:repetitions]
            arrays["input_generator_action"] = saved[key+"_generator_action"][:repetitions]
            path = folder / (key+".npz")
            np.savez_compressed(path, **arrays)
            n, d = points.shape[1:]
            cases[key] = {"instance": spec["id"], "family": family, "laws": repetitions,
                          "logical_candidate_outcomes": repetitions*8*2*8*CANDIDATES,
                          "logical_batch_outcomes": repetitions*8*2*8*BATCHES,
                          "support_and_fd_objective_queries": repetitions*(n+n*n)*(2*d+1),
                          "deterministic_reference_objective_queries": repetitions*8*(n+n*n),
                          "noisy_candidate_objective_queries": repetitions*8*5*CANDIDATES,
                          "analytic_gaussian_expectation_values": repetitions*8*(n+n*n),
                          "analytic_gradient_and_laplacian_points": repetitions*n,
                          "input_scale_guards": int(arrays["input_scale_guard"].sum()),
                          "material_record_guards": int(arrays["material_record_guard"].sum()),
                          "nonfinite_candidate_outcomes": int(arrays["candidate_nonfinite_count"].sum()),
                          "nonfinite_exact_values": int(arrays["nonfinite_exact_value_count"].sum()),
                          "cutoff_affected_support_points": int(arrays["cutoff_affected_support_count"].sum()),
                          "maximum_one_step_mean_relative_error": float(arrays["maximum_one_step_mean_relative_error"].max()),
                          "maximum_telescoping_error": float(arrays["maximum_telescoping_error"].max()),
                          "output_path": str(path.relative_to(ROOT)), "output_sha256": model.sha256(path),
                          "output_bytes": path.stat().st_size, "input_points_sha256": digest_array(points),
                          "input_weights_sha256": digest_array(weights)}
            print(key, "laws", repetitions, "bytes", path.stat().st_size, flush=True)
    assert len(cases) == 20 and all(case["laws"] == repetitions for case in cases.values())
    return folder, {"status": "smoke" if smoke else "full", "instances": summary["instances"], "cases": cases,
                    "elapsed_seconds": time.monotonic()-started, "laws_per_case": repetitions,
                    "total_laws": repetitions*20,
                    "total_logical_candidate_outcomes": sum(case["logical_candidate_outcomes"] for case in cases.values()),
                    "total_logical_batch_outcomes": sum(case["logical_batch_outcomes"] for case in cases.values()),
                    "source_sha256": model.sha256(__file__), "protocol_sha256": PROTOCOL_SHA256,
                    "one_step_source_sha256": MODEL_SHA256, "input_summary_sha256": INPUT_SUMMARY_SHA256,
                    "input_arrays_sha256": INPUT_ARRAYS_SHA256, "assemblies": model.ASSEMBLIES,
                    "orders": list(model.ORDERS), "taus": model.TAUS.tolist(), "prelaw_order": list(PRELAWS),
                    "component_order": list(COMPONENTS), "phase_order": list(PHASES), "switch_order": list(SWITCHES),
                    "input_generator_component_order": list(model.COMPONENTS),
                    "candidates": CANDIDATES, "batch_size": BATCH_SIZE, "batches": BATCHES,
                    "candidate_quantiles": QUANTILES.tolist(), "record_tolerance": "1e-12*(1+input_best)",
                    "sign_conventions": "Mean costs and paired objective differences: enabled minus disabled, positive is adverse. Record gains and symmetric recombination contributions: positive is beneficial. Continuous positive-part record gains are not thresholded; only event classifications use eta.",
                    "candidate_signed_gap_change_baseline": "f(Y)-V, where V is the ORIGINAL POPULATION MEAN objective, not the sampled parent objective",
                    "input_recombination_contribution_convention": "BENEFIT: (f(X)+f(Xprime))/2-f(midpoint); its mean is minus the unit-rate input recombination generator action. The R exact_component_mean_changes instead record COST: output mean minus stage-input mean.",
                    "paired_contrast_conventions": "paired_*objective_difference and paired_*mean_gap_difference are COST enabled-minus-disabled; paired_*record_gain_difference is BENEFIT enabled-minus-disabled. Negative signed objective sums are stored with negative sign, not as magnitudes.",
                    "paired_candidate_sign_order": ["negative", "tied", "positive"],
                    "recombination_quantile_rule": "left inverse of exact weighted empirical CDF",
                    "novelty_threshold": "0.5*input_median_nearest_neighbor_distance",
                    "array_axes": "Per-law scalars: law; outcomes: law,assembly,order,step; append batch or quantile where named; paired arrays append switch then batch/quantile; exact_phase_means and exact_component_mean_changes append named phase/component",
                    "seed_formula": "base+100000*instance_index+10000*(family==clustered)+law_index; same U and Z across all steps and switches",
                    "seed_base": SMOKE_SEED_BASE if smoke else RUN_SEED_BASE,
                    "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                    "scope": "Conditional one-step diagnostics on reused one-step-study finite laws; no trajectory, basin-entry, global convergence or independent law-validation claim",
                    "cost_scope": "Actual objective queries, counting cached evaluations once; analytic Gaussian and derivative calculations separate; implementation checks excluded"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("check", "smoke", "run"):
        modes.add_argument("--"+name, action="store_true")
    args = parser.parse_args()
    frozen_checks()
    summary = json.loads((INPUT_FOLDER / "canonical_summary.json").read_text())
    with np.load(INPUT_FOLDER / "canonical_arrays.npz") as saved:
        checks = implementation_checks(summary, saved)
        if args.check:
            print(json.dumps(checks, indent=2, allow_nan=False))
            return
        folder, result = run_panel(summary, saved, 2 if args.smoke else 128, args.smoke)
    result["implementation_checks"] = checks
    result["nonfinite_audit_passed"] = not any(case["nonfinite_candidate_outcomes"] or case["nonfinite_exact_values"]
                                              for case in result["cases"].values())
    (folder / "exploration_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    assert result["nonfinite_audit_passed"], "Nonfinite outcomes recorded in the complete outputs: diagnostic run failed"
    print(json.dumps({"folder": str(folder), "laws": result["total_laws"],
                      "logical_candidate_outcomes": result["total_logical_candidate_outcomes"],
                      "elapsed_seconds": result["elapsed_seconds"],
                      "stored_bytes": sum(case["output_bytes"] for case in result["cases"].values())}, indent=2))


if __name__ == "__main__":
    main()
