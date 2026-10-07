"""Exact frozen-law checks of eight assemblies of canonical operators.

No named optimizer is run. Calibration envelopes are empirical, not uniform
certificates. Run --check/--smoke and commit before --calibrate/--validate.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import platform
import time
import numpy as np
import objectives

ROOT = Path(__file__).resolve().parents[1]
OBJECTIVE_SOURCE_SHA256 = "7cedd6ce6389eb05aced752f5139f2c62d6820ab9b4aae7a7f4a33816cb0cedd"
PROTOCOL_SHA256 = "8c02210d7418b257805b974838037d81b058cf64214fc4e7d515a89c34b81b55"
CALIBRATION_SEED_BASE = 1572610000
VALIDATION_SEED_BASE = 1672610000
SMOKE_SEED_BASE = 1772610000
POPULATION = 32
FD_RADIUS = .001
V_GUARD = 1e-14
TAUS = .1/2.**np.arange(8)
ORDERS = ("MRS", "MSR")
COMPONENTS = ("drift", "diffusion", "selection", "recombination")
LAW_FAMILIES = ("broad", "clustered")
FAMILY_SEED_OFFSETS = {"broad": 0, "clustered": 100000}
ASSEMBLIES = [dict(id=f"S{omega}_R{gamma}_H{int(sigma > 0)}", omega=omega, gamma=gamma, sigma=sigma)
              for omega, gamma, sigma in itertools.product((0, 1), (0, 1), (0., .2))]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_input_checks():
    assert sha256(objectives.__file__) == OBJECTIVE_SOURCE_SHA256
    assert sha256(ROOT/"protocols/one-step-assembly.md") == PROTOCOL_SHA256


def gradient_laplacian(x, spec):
    """Analytic derivatives of the unchanged objective; diagnostics only."""
    family = spec["family"]
    if family in ("separable_pl", "rotated_anisotropic_pl"):
        Q = np.asarray(spec["rotation"])
        z = x@Q
        a = np.asarray(spec["weights"])
        gradient = (a*(2*z+1.05*np.sin(2*z)))@Q.T
        laplacian = np.sum(a*(2+2.1*np.cos(2*z)), axis=-1)
    elif family == "quartic":
        gradient = 4*x*(x*x-1)
        laplacian = np.sum(12*x*x-4, axis=-1)
    elif family == "rastrigin":
        gradient = 2*x+20*np.pi*np.sin(2*np.pi*x)
        laplacian = np.sum(2+40*np.pi*np.pi*np.cos(2*np.pi*x), axis=-1)
    elif family == "rosenbrock":
        Q = np.asarray(spec["rotation"])
        z = x@Q
        difference = z[..., 1:]-z[..., :-1]**2
        dz = np.zeros_like(z)
        dz[..., :-1] += -4*z[..., :-1]*difference+.02*(z[..., :-1]-1)
        dz[..., 1:] += 2*difference
        gradient = dz@Q.T
        laplacian = np.sum(12*z[..., :-1]**2-4*z[..., 1:]+2.02, axis=-1)
    elif family == "wells":
        gradient = x.copy()
        gradient[..., 0] = 4*x[..., 0]*(x[..., 0]**2-1)+.4*(x[..., 0]-1)
        laplacian = 12*x[..., 0]**2-3.6+(spec["dimension"]-1)
    elif family == "periodic":
        gradient = np.sin(x)
        laplacian = np.cos(x[..., 0])
    else:
        raise ValueError(family)
    return gradient, laplacian


def gaussian_objective(mean, variance, spec):
    """Exactly E[f(mean+sqrt(variance) Z)] for Z standard Gaussian.

    The variance is a scalar per coordinate. Orthogonal rotations preserve it;
    hence distinct transformed coordinates remain independent in Rosenbrock.
    """
    assert variance >= 0
    family, v = spec["family"], variance
    if family in ("separable_pl", "rotated_anisotropic_pl"):
        z = mean@np.asarray(spec["rotation"])
        return np.sum(np.asarray(spec["weights"])*(z*z+v+.525*(1-np.exp(-2*v)*np.cos(2*z))), axis=-1)
    if family == "quartic":
        return np.sum((mean*mean-1)**2+(6*mean*mean-2)*v+3*v*v, axis=-1)
    if family == "rastrigin":
        return np.sum(mean*mean+v+10*(1-np.exp(-2*np.pi*np.pi*v)*np.cos(2*np.pi*mean)), axis=-1)
    if family == "rosenbrock":
        z = mean@np.asarray(spec["rotation"])
        previous, following = z[..., :-1], z[..., 1:]
        return np.sum((following-previous*previous)**2
                      +(1+6*previous*previous-2*following)*v+3*v*v
                      +.01*((1-previous)**2+v), axis=-1)
    if family == "wells":
        x = mean[..., 0]
        return ((x*x-1)**2+(6*x*x-2)*v+3*v*v
                +.2*((x-1)**2+v)+.5*np.sum(mean[..., 1:]**2+v, axis=-1))
    if family == "periodic":
        return 1-np.exp(-v/2)*np.cos(mean[..., 0])
    raise ValueError(family)


def smooth_cutoff(x):
    """C-infinity radial cutoff, constant near the origin and its endpoints."""
    radius = 10*np.sqrt(x.shape[-1])
    relative = np.linalg.norm(x, axis=-1)/radius
    chi = np.ones(relative.shape)
    chi[relative >= 2] = 0.
    inside = (relative > 1) & (relative < 2)
    u = relative[inside]-1
    left, right = np.exp(-1/u), np.exp(-1/(1-u))
    chi[inside] = right/(left+right)
    return chi


def fd_drift(x, spec):
    """The finite-difference vector field is part of the declared mutation."""
    d = x.shape[-1]
    field = np.zeros_like(x)
    for j in range(d):
        offset = np.zeros(d); offset[j] = FD_RADIUS
        field[..., j] = (objectives.objective(x+offset, spec)-objectives.objective(x-offset, spec))/(2*FD_RADIUS)
    chi = smooth_cutoff(x)
    norm = np.sqrt(1+np.sum(field*field, axis=-1))
    return -chi[..., None]*field/norm[..., None], chi


def laws(spec, repetitions, family="broad"):
    points = []
    for r in range(repetitions):
        rng = np.random.default_rng(spec["initial_seed_base"]+FAMILY_SEED_OFFSETS[family]+r)
        if family == "broad":
            cloud = rng.normal(.6, 1.2, (POPULATION, spec["dimension"]))
        elif family == "clustered":
            centers = 2*rng.integers(0, 2, size=(POPULATION, 1))-1
            cloud = centers+.1*rng.normal(size=(POPULATION, spec["dimension"]))
        else:
            raise ValueError(family)
        points.append(cloud)
    return np.stack(points)


def cached_support(points, spec):
    repetitions, n, d = points.shape
    midpoints = .5*(points[:, :, None, :]+points[:, None, :, :])
    support = np.concatenate([points, midpoints.reshape(repetitions, n*n, d)], axis=1)
    f = objectives.objective(support, spec)
    assert np.isfinite(f).all() and (f >= -1e-12).all()
    drift, cutoff = fd_drift(support, spec)
    grad, lap = gradient_laplacian(points, spec)
    return {"points": points, "support": support, "f": f, "drift": drift,
            "cutoff": cutoff, "gradient_base": grad, "laplacian_base": lap,
            "phi": f/(1+f),
            "charged_diagnostic_objective_queries": repetitions*(n+n*n)*(2*d+1),
            "analytic_derivative_points": repetitions*n}


def operator_diagnostics(cache, spec, weights=None):
    """Exact finite-support S/R and exact final Gaussian M expectation."""
    points, support, f = cache["points"], cache["support"], cache["f"]
    r, n, _ = points.shape
    if weights is None:
        weights = np.full((r, n), 1/n)
    assert weights.shape == (r, n) and (weights >= 0).all()
    assert np.allclose(weights.sum(axis=1), 1., atol=1e-14, rtol=0)
    f_base = f[:, :n]
    pair_weights = weights[:, :, None]*weights[:, None, :]
    V = np.sum(weights*f_base, axis=1)
    base_drift = np.sum(weights*np.sum(cache["drift"][:, :n]*cache["gradient_base"], axis=-1), axis=1)
    base_diffusion = .5*np.sum(weights*cache["laplacian_base"], axis=1)
    phi_base = cache["phi"][:, :n]
    base_selection = -(np.sum(weights*phi_base*f_base, axis=1)
                       -np.sum(weights*phi_base, axis=1)*V)
    base_recombination = np.sum(pair_weights*f[:, n:].reshape(r, n, n), axis=(1, 2))-V
    actions = np.empty((r, len(ASSEMBLIES), len(COMPONENTS)))
    output = np.empty((r, len(ASSEMBLIES), len(ORDERS), len(TAUS)))
    maximum_mass_error = 0.
    for ai, assembly in enumerate(ASSEMBLIES):
        omega, gamma, sigma = assembly["omega"], assembly["gamma"], assembly["sigma"]
        actions[:, ai, :] = np.stack([base_drift, sigma*sigma*base_diffusion,
                                     omega*base_selection, gamma*base_recombination], axis=1)
        for ti, tau in enumerate(TAUS):
            selected = weights*np.exp(-tau*omega*phi_base)
            selected /= selected.sum(axis=1, keepdims=True)
            selected_pairs = selected[:, :, None]*selected[:, None, :]
            mrs_weights = np.concatenate([(1-gamma*tau)*selected,
                                           gamma*tau*selected_pairs.reshape(r, n*n)], axis=1)
            msr_weights = np.concatenate([(1-gamma*tau)*weights,
                                           gamma*tau*pair_weights.reshape(r, n*n)], axis=1)
            msr_weights *= np.exp(-tau*omega*cache["phi"])
            msr_weights /= msr_weights.sum(axis=1, keepdims=True)
            after_mutation = gaussian_objective(support+tau*cache["drift"], tau*sigma*sigma, spec)
            for oi, final_weights in enumerate((mrs_weights, msr_weights)):
                maximum_mass_error = max(maximum_mass_error, float(np.max(np.abs(final_weights.sum(axis=1)-1))))
                output[:, ai, oi, ti] = np.sum(final_weights*after_mutation, axis=1)
    assert maximum_mass_error < 1e-12
    generator = actions.sum(axis=2)
    quotient = (output-V[:, None, None, None])/TAUS[None, None, None, :]
    signed_remainder = quotient-generator[:, :, None, None]
    normalized = np.abs(signed_remainder)/(1+V[:, None, None, None])
    valid_V = V > V_GUARD
    coefficients = np.full_like(actions, np.nan)
    coefficients[valid_V] = -actions[valid_V]/V[valid_V, None, None]
    total_coefficients = np.full_like(generator, np.nan)
    total_coefficients[valid_V] = -generator[valid_V]/V[valid_V, None]
    sign_disagreements = quotient*generator[:, :, None, None] < 0
    sign_tolerance = 1e-14*(1+V[:, None, None, None])
    resolved_sign_disagreements = sign_disagreements & (np.abs(quotient) > sign_tolerance) & (np.abs(generator[:, :, None, None]) > sign_tolerance)
    raw = {"input_points": points, "input_weights": weights, "V": V,
           "component_actions": actions, "generator_action": generator,
           "component_effective_decay_coefficients": coefficients,
           "effective_decay_coefficient": total_coefficients, "V_ratio_valid": valid_V,
           "output_V": output, "difference_quotient": quotient,
           "signed_weak_remainder": signed_remainder, "normalized_weak_remainder": normalized,
           "signed_order_difference": output[:, :, 0]-output[:, :, 1],
           "normalized_order_quotient": np.abs(output[:, :, 0]-output[:, :, 1])/(TAUS[None, None, :]*(1+V[:, None, None])),
           "finite_step_sign_disagreement": sign_disagreements,
           "resolved_finite_step_sign_disagreement": resolved_sign_disagreements,
           "cutoff_values": cache["cutoff"],
           "cutoff_base_weight": np.sum(weights*(cache["cutoff"][:, :n] < 1), axis=1),
           "cutoff_midpoint_weight": np.sum(pair_weights*(cache["cutoff"][:, n:].reshape(r, n, n) < 1), axis=(1, 2))}
    summary = {"instance": spec["id"], "law_count": r, "population": n,
               "support_points_per_law": n+n*n, "maximum_mass_error": maximum_mass_error,
               "diagnostic_objective_queries": int(cache["charged_diagnostic_objective_queries"]),
               "analytic_gradient_and_laplacian_points": int(cache["analytic_derivative_points"]),
               "analytic_gaussian_expectation_values": int(r*(n+n*n)*len(ASSEMBLIES)*len(TAUS)),
               "V_guard_count": int((~valid_V).sum()),
               "cutoff_affected_support_points": int((cache["cutoff"] < 1).sum()),
               "cutoff_zero_support_points": int((cache["cutoff"] == 0).sum()),
               "nonfinite_outputs": int((~np.isfinite(output)).sum()), "assemblies": {}}
    assert summary["nonfinite_outputs"] == 0
    for ai, assembly in enumerate(ASSEMBLIES):
        rates = total_coefficients[valid_V, ai]
        summary["assemblies"][assembly["id"]] = {
            "component_action_min": actions[:, ai].min(axis=0).tolist(),
            "component_action_max": actions[:, ai].max(axis=0).tolist(),
            "component_action_mean": actions[:, ai].mean(axis=0).tolist(),
            "positive_generator_count": int((generator[:, ai] > 0).sum()),
            "lambda_empirical_min": float(rates.min()) if len(rates) else None,
            "lambda_empirical_median": float(np.median(rates)) if len(rates) else None,
            "lambda_empirical_max": float(rates.max()) if len(rates) else None,
            "normalized_remainder_max": normalized[:, ai].max(axis=0).tolist(),
            "normalized_remainder_median": np.median(normalized[:, ai], axis=0).tolist(),
            "normalized_order_quotient_max": raw["normalized_order_quotient"][:, ai].max(axis=0).tolist(),
            "finite_step_sign_disagreements": sign_disagreements[:, ai].sum(axis=0).tolist(),
            "resolved_finite_step_sign_disagreements": resolved_sign_disagreements[:, ai].sum(axis=0).tolist()}
    return summary, raw


def calibration_envelopes(summary):
    envelopes = {}
    for spec in summary["instances"]:
        instance = spec["id"]
        envelopes[instance] = {}
        for assembly in ASSEMBLIES:
            rows = [summary["cases"][instance+"__"+family]["assemblies"][assembly["id"]]
                    for family in LAW_FAMILIES]
            minima = [row["lambda_empirical_min"] for row in rows if row["lambda_empirical_min"] is not None]
            envelopes[instance][assembly["id"]] = {
                "lambda_cal": min(minima) if minima else None,
                "a_cal": np.max([row["normalized_remainder_max"] for row in rows], axis=0).tolist(),
                "pooled_law_families": list(LAW_FAMILIES)}
    return envelopes


def apply_validation_envelopes(summary, arrays, envelopes):
    for case_key, case in summary["cases"].items():
        instance = case["instance"]
        V = arrays[case_key+"_V"]
        generator = arrays[case_key+"_generator_action"]
        normalized = arrays[case_key+"_normalized_weak_remainder"]
        valid_V = arrays[case_key+"_V_ratio_valid"]
        for ai, assembly in enumerate(ASSEMBLIES):
            envelope = envelopes[instance][assembly["id"]]
            lam = envelope["lambda_cal"]
            remainder_violations = normalized[:, ai] > np.asarray(envelope["a_cal"])[None, :, :]
            margin = np.full(len(V), np.nan) if lam is None else generator[:, ai]+lam*V
            drift_violations = valid_V & (margin > 0)
            prefix = case_key+"_"+assembly["id"]
            arrays[prefix+"_remainder_envelope_violation"] = remainder_violations
            arrays[prefix+"_drift_envelope_violation"] = drift_violations
            arrays[prefix+"_drift_envelope_signed_margin"] = margin
            case["assemblies"][assembly["id"]]["calibration_envelope_validation"] = {
                "lambda_cal": lam, "a_cal": envelope["a_cal"],
                "eligible_drift_laws": int(valid_V.sum()) if lam is not None else 0,
                "drift_violations": int(drift_violations.sum()),
                "remainder_violations": remainder_violations.sum(axis=0).tolist(),
                "maximum_signed_drift_margin": float(np.nanmax(margin)) if lam is not None else None}
    pooled = {}
    for spec in summary["instances"]:
        instance = spec["id"]
        pooled[instance] = {}
        for assembly in ASSEMBLIES:
            rows = [summary["cases"][instance+"__"+family]["assemblies"][assembly["id"]]["calibration_envelope_validation"]
                    for family in LAW_FAMILIES]
            pooled[instance][assembly["id"]] = {
                "eligible_drift_laws": sum(row["eligible_drift_laws"] for row in rows),
                "drift_violations": sum(row["drift_violations"] for row in rows),
                "remainder_violations": np.sum([row["remainder_violations"] for row in rows], axis=0).tolist()}
    summary["pooled_validation_violations"] = pooled


def gaussian_quadrature(mean, variance, spec):
    """Independent tensor Gaussian quadrature on anchored one/two-way terms.

    All seven objectives are sums of at most two-coordinate interactions in
    their declared rotated coordinates. Summing all anchored pairs therefore
    gives their full expectation, without using the closed-form formulas.
    """
    nodes, weights = np.polynomial.hermite.hermgauss(24)
    nodes = np.sqrt(2*variance)*nodes; weights = weights/np.sqrt(np.pi)
    Q = np.asarray(spec["rotation"])
    d = len(mean)
    baseline = float(objectives.objective(mean, spec))
    singles = np.array([np.dot(weights, objectives.objective(mean+nodes[:, None]*Q[:, j], spec)) for j in range(d)])
    result = baseline+np.sum(singles-baseline)
    for i in range(d):
        for j in range(i+1, d):
            points = mean+nodes[:, None, None]*Q[:, i]+nodes[None, :, None]*Q[:, j]
            pair = np.einsum("i,j,ij->", weights, weights, objectives.objective(points, spec))
            result += pair-singles[i]-singles[j]+baseline
    return result


def implementation_checks():
    frozen_input_checks()
    seed_sets = {}
    for name, base, count in (("calibration", CALIBRATION_SEED_BASE, 32),
                              ("validation", VALIDATION_SEED_BASE, 128),
                              ("smoke", SMOKE_SEED_BASE, 4)):
        values = {base+1000*j+offset+r for j in range(10) for r in range(count)
                  for offset in FAMILY_SEED_OFFSETS.values()}
        assert len(values) == 20*count
        seed_sets[name] = values
    for left, right in itertools.combinations(seed_sets.values(), 2):
        assert left.isdisjoint(right)
    assert min(seed_sets["calibration"]) > 1503919000  # Largest prior restart smoke seed.
    maximum_gradient_error = maximum_laplacian_error = maximum_gaussian_error = 0.
    for spec in objectives.instances(SMOKE_SEED_BASE):
        rng = np.random.default_rng(spec["initial_seed_base"])
        x = rng.normal(.6, 1.2, (3, spec["dimension"]))
        g, lap = gradient_laplacian(x, spec)
        numerical_g = np.zeros_like(x); numerical_lap = np.zeros(3)
        center = objectives.objective(x, spec)
        for j in range(spec["dimension"]):
            unit = np.eye(spec["dimension"])[j]
            numerical_g[:, j] = (objectives.objective(x+1e-5*unit, spec)-objectives.objective(x-1e-5*unit, spec))/(2e-5)
            numerical_lap += (objectives.objective(x+1e-4*unit, spec)-2*center+objectives.objective(x-1e-4*unit, spec))/1e-8
        gradient_error = float(np.max(np.abs(g-numerical_g)/(1+np.abs(g))))
        laplacian_error = float(np.max(np.abs(lap-numerical_lap)/(1+np.abs(lap))))
        assert gradient_error < 1e-6, (spec["id"], gradient_error)
        assert laplacian_error < 1e-4, (spec["id"], laplacian_error)
        maximum_gradient_error = max(maximum_gradient_error, gradient_error)
        maximum_laplacian_error = max(maximum_laplacian_error, laplacian_error)
        assert np.allclose(gaussian_objective(x, 0., spec), center, rtol=1e-12, atol=1e-12)
        for variance in (.005, .07, .2):
            numerical = np.array([gaussian_quadrature(mean, variance, spec) for mean in x])
            exact = gaussian_objective(x, variance, spec)
            error = float(np.max(np.abs(exact-numerical)/(1+np.abs(exact))))
            assert error < 1e-10, (spec["id"], variance, error)
            maximum_gaussian_error = max(maximum_gaussian_error, error)
        for family in LAW_FAMILIES:
            cache = cached_support(laws(spec, 2, family), spec)
            summary, raw = operator_diagnostics(cache, spec)
            assert np.allclose(raw["component_actions"].sum(axis=2), raw["generator_action"], rtol=0, atol=0)
            assert (raw["component_actions"][:, :, 2] <= 1e-12).all()
            for ai, assembly in enumerate(ASSEMBLIES):
                if assembly["omega"] == 0 or assembly["gamma"] == 0:
                    assert np.allclose(raw["output_V"][:, ai, 0], raw["output_V"][:, ai, 1], rtol=1e-13, atol=1e-12)
            assert np.allclose(raw["component_effective_decay_coefficients"].sum(axis=2), raw["effective_decay_coefficient"], rtol=1e-13, atol=1e-13)
    # The cutoff is inactive inside R and exactly zero outside 2R, and the
    # field remains bounded on both the transition and the outer region.
    spec = objectives.instances(SMOKE_SEED_BASE)[0]
    d = spec["dimension"]; radius = 10*np.sqrt(d)
    x = np.zeros((5, d)); x[:, 0] = radius*np.array([0, .5, 1, 1.5, 2.1])
    b, chi = fd_drift(x, spec)
    assert np.allclose(chi, [1, 1, 1, .5, 0])
    assert (np.linalg.norm(b, axis=1) <= 1).all() and (b[-1] == 0).all()
    return {"maximum_relative_gradient_error": maximum_gradient_error,
            "maximum_relative_laplacian_error": maximum_laplacian_error,
            "maximum_relative_gaussian_quadrature_error": maximum_gaussian_error,
            "gaussian_quadrature_variances": [.005, .07, .2], "all_ten_instances_checked": True,
            "mass_component_sum_and_inactive_order_checks": "passed", "cutoff_checks": "passed",
            "seed_ranges": {name: [min(values), max(values)] for name, values in seed_sets.items()},
            "fresh_seed_namespaces_disjoint": True}


def run_suite(repetitions, seed_base):
    started = time.monotonic()
    cases, arrays = {}, {"taus": TAUS}
    specs = objectives.instances(seed_base)
    for spec in specs:
        for family in LAW_FAMILIES:
            points = laws(spec, repetitions, family)
            case, raw = operator_diagnostics(cached_support(points, spec), spec)
            case.update(law_family=family,
                        law_seed_base=spec["initial_seed_base"]+FAMILY_SEED_OFFSETS[family])
            key = spec["id"]+"__"+family
            cases[key] = case
            arrays.update({key+"_"+field: values for field, values in raw.items()})
            print(key, "laws", repetitions, "assemblies", len(ASSEMBLIES),
                  "diagnostic queries", case["diagnostic_objective_queries"],
                  "cutoff points", case["cutoff_affected_support_points"], flush=True)
    assert len(cases) == 20
    return {"instances": specs, "cases": cases, "law_count_per_instance": 2*repetitions,
            "law_count_per_instance_family": repetitions, "law_families": list(LAW_FAMILIES),
            "family_seed_offsets": FAMILY_SEED_OFFSETS,
            "assemblies": ASSEMBLIES, "orders": list(ORDERS), "component_order": list(COMPONENTS),
            "taus": TAUS.tolist(), "population": POPULATION, "seed_namespace_base": seed_base,
            "fd_radius": FD_RADIUS, "V_ratio_guard": V_GUARD,
            "diagnostic_objective_queries": sum(case["diagnostic_objective_queries"] for case in cases.values()),
            "elapsed_seconds": time.monotonic()-started}, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("check", "smoke", "calibrate", "validate"):
        modes.add_argument("--"+name, action="store_true")
    args = parser.parse_args()
    checks = implementation_checks()
    if args.check:
        print(json.dumps(checks, indent=2)); return
    mode = "smoke" if args.smoke else "calibration" if args.calibrate else "validation"
    folder = ROOT/"results/one-step"/("canonical-smoke-two-families" if args.smoke else "canonical-"+mode)
    if mode != "smoke" and (folder/"canonical_summary.json").exists():
        raise FileExistsError("Preserving completed results: "+str(folder))
    calibration = None
    if args.validate:
        path = ROOT/"results/one-step/canonical-calibration/canonical_summary.json"
        calibration = json.loads(path.read_text())
        assert calibration["source_sha256"] == sha256(__file__), "Source changed after calibration"
        assert calibration["protocol_sha256"] == PROTOCOL_SHA256
        assert calibration["seed_namespace_base"] == CALIBRATION_SEED_BASE
        assert calibration["law_count_per_instance_family"] == 32
        assert calibration["law_families"] == list(LAW_FAMILIES)
    repetitions = 4 if args.smoke else 32 if args.calibrate else 128
    seed_base = SMOKE_SEED_BASE if args.smoke else CALIBRATION_SEED_BASE if args.calibrate else VALIDATION_SEED_BASE
    summary, arrays = run_suite(repetitions, seed_base)
    summary.update(status=mode, implementation_checks=checks, source_sha256=sha256(__file__),
                   objective_source_sha256=OBJECTIVE_SOURCE_SHA256, protocol_sha256=PROTOCOL_SHA256,
                   python=platform.python_version(), numpy=np.__version__,
                   initial_law="32 iid atoms with equal weights: broad N(.6*1,1.2^2 I), or half/half N(+1,.1^2 I)/N(-1,.1^2 I) per point; assemblies/orders/steps paired on each law",
                   scope="Exact objective expectations on sampled finite laws; no optimizer trajectory, particle limit, uniform drift certificate or global performance claim",
                   array_axes="V and cutoff weights: law; component actions: law,assembly,component; outputs/remainders: law,assembly,order,tau",
                   oracle_cost_scope="Actual objective calls used to cache all support values and central differences; analytic derivatives/Gaussian expectations counted separately; implementation tests excluded from experiment totals")
    if args.calibrate:
        summary["empirical_envelopes"] = calibration_envelopes(summary)
    if args.validate:
        apply_validation_envelopes(summary, arrays, calibration["empirical_envelopes"])
        summary["calibration_summary_sha256"] = sha256(ROOT/"results/one-step/canonical-calibration/canonical_summary.json")
        summary["calibration_seed_namespace_base"] = CALIBRATION_SEED_BASE
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"canonical_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    np.savez_compressed(folder/"canonical_arrays.npz", **arrays)
    print(json.dumps({"folder": str(folder), "laws_per_instance_family": repetitions,
                      "diagnostic_objective_queries": summary["diagnostic_objective_queries"],
                      "elapsed_seconds": summary["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
