"""Full-panel finite-population discovery experiment, without fitted certificates.

Run --check, then --smoke, and freeze/commit the protocol before --run.
Output directories and files are created exclusively: no run is overwritten.
All recorded probabilities are observations; statistical analysis is separate.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import platform
import time

import numpy as np

import one_step_study as model


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/finite-population.md"
TAU = .1
GENERATIONS = 200
REPETITIONS = 128
POPULATIONS = (16, 32, 64, 128)
EPSILONS = np.array([.1, .01, 1.])  # Primary first; do not sort implicitly.
PROBE_SIZE = 128
AUDIT_BATCHES = np.array([1, 4, 16, 32])
AUDIT_SIZE = 32
FULL_SEED = 2972600000
SMOKE_SEED = 3072600000
CHECK_SEED = 3172600000
METRICS = ("record_gap", "mean_gap", "variance", "minimizer_distance2")
QUERY_CATEGORIES = ("initial", "selection_trial", "finite_difference", "output")
STREAMS = {"initial": 0, "evolution": 1, "probe": 2, "audit": 3}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json_exclusive(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def save_exclusive(path, **arrays):
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def seed_words(base, stream, instance, family, run, population=0, assembly=0, order=0):
    """Initialization deliberately omits N/assembly/order for nested prefixes."""
    if stream == "initial":
        return [base, STREAMS[stream], instance, family, run]
    return [base, STREAMS[stream], instance, family, run, population, assembly, order]


def generators(words):
    return [np.random.default_rng(np.random.SeedSequence(row)) for row in words]


def initial_panel(spec, family, repetitions, base, instance_index, family_index):
    words = [seed_words(base, "initial", instance_index, family_index, r)
             for r in range(repetitions)]
    rows = []
    for rng in generators(words):
        shape = (max(POPULATIONS), spec["dimension"])
        if family == "broad":
            x = rng.normal(.6, 1.2, shape)
        elif family == "clustered":
            signs = 2*rng.integers(0, 2, (shape[0], 1))-1
            x = signs+.1*rng.normal(size=shape)
        else:
            raise ValueError(family)
        rows.append(x)
    return np.stack(rows), np.asarray(words, dtype=np.uint64)


class Ledger:
    """Actual objective calls, classified and archived per independent run.

    Different streams own different ledgers; diagnostic values cannot enter
    the operational archive. Cached current values are not charged again.
    Each vector entry is one objective call, including repeated locations.
    """
    def __init__(self, repetitions):
        self.counts = np.zeros((repetitions, len(QUERY_CATEGORIES)), dtype=np.int64)
        self.best = np.full(repetitions, np.inf)

    def evaluate(self, points, spec, category, run_ids=None):
        if not np.isfinite(points).all():
            raise FloatingPointError("Nonfinite query coordinates; preserve this run")
        values = model.objectives.objective(points, spec)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise FloatingPointError("Nonfinite/negative objective; preserve this run")
        index = QUERY_CATEGORIES.index(category)
        if run_ids is None:
            if values.ndim != 2 or values.shape[0] != len(self.best):
                raise ValueError("Rectangular query batches need run, candidate axes")
            self.counts[:, index] += values.shape[1]
            self.best = np.minimum(self.best, values.min(axis=1))
        else:
            if values.ndim != 1 or values.shape != run_ids.shape:
                raise ValueError("Ragged query batches need matching run indices")
            self.counts[:, index] += np.bincount(run_ids, minlength=len(self.best))
            np.minimum.at(self.best, run_ids, values)
        return values


def finite_difference_drift(points, spec, ledger):
    field = np.empty_like(points)
    dimension = points.shape[-1]
    for j in range(dimension):
        offset = np.zeros(dimension)
        offset[j] = model.FD_RADIUS
        plus = ledger.evaluate(points+offset, spec, "finite_difference")
        minus = ledger.evaluate(points-offset, spec, "finite_difference")
        field[..., j] = (plus-minus)/(2*model.FD_RADIUS)
    cutoff = model.smooth_cutoff(points)
    field *= -cutoff[..., None]/np.sqrt(1+np.sum(field*field, axis=-1))[..., None]
    if not np.isfinite(field).all():
        raise FloatingPointError("Nonfinite finite-difference drift")
    return field, np.count_nonzero(cutoff != 1, axis=1)


def premutation_sample(points, values, rngs, size, spec, assembly, order, ledger):
    """Exact samples of R S mu (MRS) or S R mu (MSR), without N^2 support.

    In MSR with both switches enabled, rejection proposals have law R mu;
    acceptance exp(-tau Phi) gives exactly S R mu. Every trial score is
    evaluated, charged, and archived, including rejected/copy proposals.
    With an identity component, both orders use the same direct sampler.
    """
    repetitions, population, dimension = points.shape
    if order not in model.ORDERS:
        raise ValueError(order)
    omega, gamma = assembly["omega"], assembly["gamma"]
    if order == "MSR" and omega and gamma:
        sampled = np.empty((repetitions, size, dimension))
        pending = np.ones((repetitions, size), dtype=bool)
        while pending.any():
            run_ids, slots = np.nonzero(pending)
            needed = np.bincount(run_ids, minlength=repetitions)
            uniforms = np.concatenate([rng.random((int(n), 4))
                                       for rng, n in zip(rngs, needed) if n])
            first = (population*uniforms[:, 0]).astype(np.int64)
            second = (population*uniforms[:, 1]).astype(np.int64)
            proposed = points[run_ids, first].copy()
            mix = uniforms[:, 2] < gamma*TAU
            proposed[mix] = .5*(proposed[mix]+points[run_ids[mix], second[mix]])
            scores = ledger.evaluate(proposed, spec, "selection_trial", run_ids)
            accepted = uniforms[:, 3] < np.exp(-omega*TAU*scores/(1+scores))
            sampled[run_ids[accepted], slots[accepted]] = proposed[accepted]
            pending[run_ids[accepted], slots[accepted]] = False
        return sampled

    uniforms = np.stack([rng.random((size, 3)) for rng in rngs])
    if omega:
        weights = np.exp(-omega*TAU*values/(1+values))
        cdf = np.cumsum(weights/weights.sum(axis=1, keepdims=True), axis=1)
        cdf[:, -1] = 1.
        # Per-row search is O(size log N); no run x size x N comparison array.
        indices = np.stack([np.searchsorted(row, u[:, :2], side="right")
                            for row, u in zip(cdf, uniforms)])
    else:
        indices = (population*uniforms[:, :, :2]).astype(np.int64)
    run_ids = np.arange(repetitions)[:, None]
    first = points[run_ids, indices[..., 0]]
    second = points[run_ids, indices[..., 1]]
    mix = uniforms[..., 2] < gamma*TAU
    return np.where(mix[..., None], .5*(first+second), first)


def transition_sample(points, values, rngs, size, spec, assembly, order, ledger):
    before = premutation_sample(points, values, rngs, size, spec, assembly, order, ledger)
    drift, affected = finite_difference_drift(before, spec, ledger)
    after = before+TAU*drift
    if assembly["sigma"]:
        normals = np.stack([rng.normal(size=(size, spec["dimension"])) for rng in rngs])
        after += assembly["sigma"]*np.sqrt(TAU)*normals
    output_values = ledger.evaluate(after, spec, "output")
    return after, output_values, affected


def population_metrics(points, values, record, spec):
    center = points.mean(axis=1, keepdims=True)
    variance = np.mean(np.sum((points-center)**2, axis=-1), axis=1)
    if spec["family"] == "quartic":
        distance2 = np.sum((np.abs(points)-1)**2, axis=-1)
    elif spec["family"] == "periodic":
        nearest = 2*np.pi*np.rint(points/(2*np.pi))
        distance2 = np.sum((points-nearest)**2, axis=-1)
    else:
        distance2 = np.sum((points-np.asarray(spec["target"]))**2, axis=-1)
    return np.stack([record, values.mean(axis=1), variance, distance2.mean(axis=1)], axis=-1)


def run_cell(initial, spec, generations, words, assembly, order, failure_path):
    repetitions, population, _ = initial.shape
    rng = {name: generators(rows) for name, rows in words.items()}
    ledgers = {name: Ledger(repetitions) for name in words}
    operational = ledgers["evolution"]
    points = initial.copy()
    values = operational.evaluate(points, spec, "initial")
    traces = np.full((repetitions, generations+1, len(METRICS)), np.nan)
    good_fraction = np.full((repetitions, generations+1, len(EPSILONS)), np.nan)
    offspring_hit = np.zeros((repetitions, generations+1, len(EPSILONS)), dtype=bool)
    probe_counts = np.zeros((repetitions, generations, len(EPSILONS)), dtype=np.uint16)
    audit_counts = np.zeros((repetitions, generations, len(AUDIT_BATCHES), len(EPSILONS)),
                            dtype=np.uint16)
    audit_miss = np.ones((repetitions, generations+1, len(AUDIT_BATCHES), len(EPSILONS)), dtype=bool)
    counts = {name: np.zeros((repetitions, generations+1, len(QUERY_CATEGORIES)), dtype=np.int64)
              for name in words}
    cutoff = {name: np.zeros((repetitions, generations), dtype=np.uint16) for name in words}
    traces[:, 0] = population_metrics(points, values, operational.best, spec)
    good_fraction[:, 0] = np.mean(values[..., None] <= EPSILONS, axis=1)
    counts["evolution"][:, 0] = operational.counts
    generation = 0
    try:
        for generation in range(1, generations+1):
            # All three streams use the same pre-update law. Neither observer
            # changes current points, current cached scores, or the archive.
            _, probe_values, changed = transition_sample(
                points, values, rng["probe"], PROBE_SIZE, spec, assembly, order, ledgers["probe"])
            cutoff["probe"][:, generation-1] = changed
            probe_counts[:, generation-1] = np.sum(probe_values[..., None] <= EPSILONS, axis=1)
            _, audit_values, changed = transition_sample(
                points, values, rng["audit"], AUDIT_SIZE, spec, assembly, order, ledgers["audit"])
            cutoff["audit"][:, generation-1] = changed
            successes = np.cumsum(audit_values[..., None] <= EPSILONS, axis=1)
            audit_counts[:, generation-1] = successes[:, AUDIT_BATCHES-1]
            audit_miss[:, generation] = (audit_miss[:, generation-1]
                                          & (audit_counts[:, generation-1] == 0))
            points, values, changed = transition_sample(
                points, values, rng["evolution"], population, spec, assembly, order, operational)
            cutoff["evolution"][:, generation-1] = changed
            good = values[..., None] <= EPSILONS
            offspring_hit[:, generation] = offspring_hit[:, generation-1] | np.any(good, axis=1)
            good_fraction[:, generation] = np.mean(good, axis=1)
            traces[:, generation] = population_metrics(points, values, operational.best, spec)
            for name in words:
                counts[name][:, generation] = ledgers[name].counts
        validate_cell(traces, good_fraction, offspring_hit, probe_counts, audit_counts,
                      audit_miss, counts, points, values, population, spec)
    except Exception:
        save_exclusive(failure_path, generation=generation, points=points, values=values,
                       traces=traces, operational_counts=operational.counts,
                       operational_record=operational.best, probe_counts=probe_counts,
                       audit_counts=audit_counts, offspring_hit=offspring_hit)
        raise
    arrays = dict(traces=traces, current_good_fraction=good_fraction,
                  offspring_cumulative_hit=offspring_hit, probe_success_counts=probe_counts,
                  audit_batch_success_counts=audit_counts, audit_cumulative_miss=audit_miss,
                  final_points=points, final_values=values)
    for name in words:
        arrays[name+"_cumulative_calls_by_category"] = counts[name]
        arrays[name+"_cutoff_affected"] = cutoff[name]
    return arrays


def validate_cell(traces, good_fraction, hit, probe_counts, audit_counts, audit_miss,
                  counts, points, values, population, spec):
    assert np.isfinite(traces).all() and np.isfinite(points).all() and np.isfinite(values).all()
    assert np.all(np.diff(traces[..., 0], axis=1) <= 0)
    assert np.all(traces[:, -1, 0] <= values.min(axis=1))
    assert np.all((good_fraction >= 0) & (good_fraction <= 1))
    assert not hit[:, 0].any() and audit_miss[:, 0].all()
    assert not np.any(hit[:, :-1] & ~hit[:, 1:])
    assert not np.any(~audit_miss[:, :-1] & audit_miss[:, 1:])
    assert np.all(probe_counts <= PROBE_SIZE)
    assert np.all(audit_counts <= AUDIT_BATCHES[None, None, :, None])
    assert np.all(np.diff(audit_counts.astype(np.int64), axis=2) >= 0)
    epsilon_order = np.argsort(EPSILONS)
    assert np.all(np.diff(probe_counts[..., epsilon_order].astype(np.int64), axis=-1) >= 0)
    assert np.all(np.diff(audit_counts[..., epsilon_order].astype(np.int64), axis=-1) >= 0)
    assert np.all(counts["evolution"][:, 0, 0] == population)
    for name, size in (("evolution", population), ("probe", PROBE_SIZE), ("audit", AUDIT_SIZE)):
        differences = np.diff(counts[name], axis=1)
        assert np.all(differences >= 0)
        assert np.all(differences[..., 2] == 2*spec["dimension"]*size)
        assert np.all(differences[..., 3] == size)
        assert np.all(differences[..., 0] == 0)
    assert not counts["probe"][:, 0].any() and not counts["audit"][:, 0].any()


def run_case(job):
    spec, instance_index, family, family_index, population, repetitions, generations, base, folder = job
    started = time.monotonic()
    folder = Path(folder)
    key = spec["id"]+"__"+family+"__N"+str(population)
    complete_initial, initial_words = initial_panel(spec, family, repetitions, base,
                                                   instance_index, family_index)
    initial = complete_initial[:, :population].copy()
    all_arrays = {}
    stream_words = {name: np.empty((repetitions, len(model.ASSEMBLIES), len(model.ORDERS), 8),
                                  dtype=np.uint64)
                    for name in ("evolution", "probe", "audit")}
    for ai, assembly in enumerate(model.ASSEMBLIES):
        for oi, order in enumerate(model.ORDERS):
            words = {name: [seed_words(base, name, instance_index, family_index, r,
                                      population, ai, oi) for r in range(repetitions)]
                     for name in stream_words}
            for name in words:
                stream_words[name][:, ai, oi] = words[name]
            result = run_cell(initial, spec, generations, words, assembly, order,
                              folder/(key+"__"+assembly["id"]+"__"+order+"__failure.npz"))
            if not all_arrays:
                all_arrays = {name: np.empty((repetitions, len(model.ASSEMBLIES), len(model.ORDERS))
                                            + array.shape[1:], dtype=array.dtype)
                              for name, array in result.items()}
            for name, array in result.items():
                all_arrays[name][:, ai, oi] = array
    path = folder/(key+".npz")
    save_exclusive(path, **all_arrays, initial_points=initial, initial_seed_words=initial_words,
                   **{name+"_seed_words": array for name, array in stream_words.items()},
                   epsilons=EPSILONS, audit_batch_sizes=AUDIT_BATCHES,
                   metrics=np.asarray(METRICS), query_categories=np.asarray(QUERY_CATEGORIES),
                   assemblies=np.asarray([a["id"] for a in model.ASSEMBLIES]),
                   orders=np.asarray(model.ORDERS))
    summary = {"output_sha256": sha256(path), "elapsed_seconds": time.monotonic()-started,
               "population": population, "runs": repetitions, "generations": generations,
               "final_record_success_counts":
                   np.sum(all_arrays["traces"][:, :, :, -1, 0, None] <= EPSILONS, axis=0).tolist(),
               "final_offspring_success_counts":
                   all_arrays["offspring_cumulative_hit"][:, :, :, -1].sum(axis=0).tolist(),
               "final_audit_failure_counts":
                   all_arrays["audit_cumulative_miss"][:, :, :, -1].sum(axis=0).tolist()}
    for name in stream_words:
        calls = all_arrays[name+"_cumulative_calls_by_category"][:, :, :, -1]
        summary[name+"_total_objective_calls"] = int(calls.sum())
        summary[name+"_calls_by_category"] = calls.sum(axis=(0, 1, 2)).tolist()
        summary[name+"_cutoff_affected"] = int(all_arrays[name+"_cutoff_affected"].sum())
    write_json_exclusive(folder/(key+".json"), summary)
    return key, summary


def check():
    model.frozen_input_checks()
    specs = model.objectives.instances()
    assert len(specs) == 10 and len(model.ASSEMBLIES) == 8 and len(model.ORDERS) == 2
    assert PROBE_SIZE <= np.iinfo(np.uint16).max
    # Every objective uses the existing field, including its smooth cutoff.
    for ii, spec in enumerate(specs):
        panel, words = initial_panel(spec, "broad", 2, CHECK_SEED, ii, 0)
        repeated, repeated_words = initial_panel(spec, "broad", 2, CHECK_SEED, ii, 0)
        assert np.array_equal(panel, repeated) and np.array_equal(words, repeated_words)
        for population in POPULATIONS:
            assert np.array_equal(panel[:, :population], repeated[:, :population])
        ledger = Ledger(2)
        drift, _ = finite_difference_drift(panel[:, :3], spec, ledger)
        assert np.allclose(drift, model.fd_drift(panel[:, :3], spec)[0], atol=1e-15, rtol=1e-14)
        assert np.all(ledger.counts[:, 2] == 6*spec["dimension"])
    # Check exact finite-support weights of both orders against sampled laws.
    spec = next(s for s in specs if s["family"] == "periodic")
    points = np.array([[[0.], [2.], [7.]]])
    values = model.objectives.objective(points, spec)
    n = points.shape[1]
    mids = .5*(points[0, :, None]+points[0, None, :]).reshape(-1, 1)
    support = np.concatenate([points[0], mids])[:, 0]
    locations, inverse = np.unique(support, return_inverse=True)
    score = model.objectives.objective(support[:, None], spec)
    draws = 30000
    for ai, assembly in enumerate(model.ASSEMBLIES):
        selected = np.exp(-TAU*assembly["omega"]*values[0]/(1+values[0]))
        selected /= selected.sum()
        for oi, order in enumerate(model.ORDERS):
            if order == "MRS":
                weight = np.r_[(1-TAU*assembly["gamma"])*selected,
                               TAU*assembly["gamma"]*np.outer(selected, selected).ravel()]
            else:
                weight = np.r_[np.full(n, (1-TAU*assembly["gamma"])/n),
                               np.full(n*n, TAU*assembly["gamma"]/(n*n))]
                weight *= np.exp(-TAU*assembly["omega"]*score/(1+score))
                weight /= weight.sum()
            exact = np.bincount(inverse, weights=weight, minlength=len(locations))
            ledger = Ledger(1)
            sampled = premutation_sample(points, values, [np.random.default_rng(CHECK_SEED+ai*2+oi)],
                                          draws, spec, assembly, order, ledger)[0, :, 0]
            assert np.isin(sampled, locations).all()
            observed = np.bincount(np.searchsorted(locations, sampled), minlength=len(locations))/draws
            tolerance = 7*np.sqrt(exact*(1-exact)/draws)+7/draws
            assert np.all(np.abs(observed-exact) <= tolerance), (assembly, order, exact, observed)
            if order == "MSR" and assembly["omega"] and assembly["gamma"]:
                assert ledger.counts[0, 1] >= draws
            else:
                assert not ledger.counts.any()
    # Force a better rejected proposal: its score must still enter the archive.
    class ScriptedRandom:
        def __init__(self):
            self.rows = [np.array([[0., 0., .5, .99999]]), np.array([[.75, .75, .5, 0.]])]
        def random(self, shape):
            row = self.rows.pop(0)
            assert row.shape == shape
            return row
    x = np.array([[[1.], [2.]]])
    f = model.objectives.objective(x, spec)
    ledger = Ledger(1)
    assembly = dict(omega=1, gamma=1, sigma=0.)
    sampled = premutation_sample(x, f, [ScriptedRandom()], 1, spec, assembly, "MSR", ledger)
    assert sampled[0, 0, 0] == 2. and ledger.counts[0, 1] == 2
    assert ledger.best[0] == f[0, 0] < f[0, 1]
    # Observer randomness and ledgers do not alter operational transitions.
    evolution_words = [seed_words(CHECK_SEED, "evolution", 0, 0, 0, 16, 7, 1)]
    left, right, observer = Ledger(1), Ledger(1), Ledger(1)
    assembly = dict(omega=1, gamma=1, sigma=.2)
    first = transition_sample(x, f, generators(evolution_words), 16, spec, assembly, "MSR", left)
    transition_sample(x, f, [np.random.default_rng(CHECK_SEED+999)], 128, spec, assembly, "MSR", observer)
    second = transition_sample(x, f, generators(evolution_words), 16, spec, assembly, "MSR", right)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert np.array_equal(left.counts, right.counts) and np.array_equal(left.best, right.best)
    print("PASS: ten objective/field checks; exact-order finite-support sampling; "
          "rejected-score archive; independent observers; nested initialization", flush=True)


def run(smoke=False, output=None, workers=1):
    started = time.monotonic()
    model.frozen_input_checks()
    if not PROTOCOL.is_file():
        raise FileNotFoundError("Freeze the declared protocol before running: "+str(PROTOCOL))
    specs = model.objectives.instances()
    repetitions, generations = (3, 3) if smoke else (REPETITIONS, GENERATIONS)
    populations = (16, 32) if smoke else POPULATIONS
    base = SMOKE_SEED if smoke else FULL_SEED
    folder = Path(output).resolve() if output else ROOT/"results/trajectories"/("finite-population-smoke" if smoke
                                                                   else "finite-population-full")
    # Even an empty existing directory is not reused. Partial runs are retained.
    folder.mkdir(parents=True, exist_ok=False)
    sources = {"runner": sha256(__file__), "one_step_study": sha256(model.__file__),
               "objectives": sha256(model.objectives.__file__), "protocol": sha256(PROTOCOL)}
    manifest = {"status": "smoke" if smoke else "full", "sources_sha256": sources,
                "instances": specs, "families": model.LAW_FAMILIES,
                "assemblies": model.ASSEMBLIES, "orders": model.ORDERS,
                "populations": populations, "runs_per_cell": repetitions,
                "generations": generations, "tau": TAU, "epsilons": EPSILONS.tolist(),
                "primary_epsilon": float(EPSILONS[0]), "probe_size": PROBE_SIZE,
                "audit_batch_sizes": AUDIT_BATCHES.tolist(), "metrics": METRICS,
                "query_categories": QUERY_CATEGORIES, "seed_base": base,
                "seed_construction": "SeedSequence entropy words saved in each NPZ; initial: "
                    "[base,0,instance,family,run]; other: [base,stream,instance,family,run,N,assembly,order]",
                "stream_ids": STREAMS, "randomness_pairing": "Identical initial panels across assemblies/orders; "
                    "nested N prefixes of 128 atoms. Evolution/probe/audit streams independent, including "
                    "across assemblies/orders/N. Audit B values share nested batch prefixes.",
                "array_axes": "Common leading axes: run, assembly, order. Traces/current_good_fraction/"
                    "offspring_cumulative_hit/cumulative_calls/audit_cumulative_miss include generation zero. "
                    "Probe counts/audit batch counts/cutoffs index pre-update laws for generations 1..K. "
                    "Audit arrays end in batch-size, epsilon; other success arrays end in epsilon. "
                    "Call arrays end in query category; traces end in metric.",
                "accounting": "Operational archive includes initialization, all evolution finite-difference "
                    "queries, all evolution rejection-trial scores, and output values. Cached current selection "
                    "scores are reused without a call. Probe/audit evaluations are separate diagnostic costs "
                    "and never enter the operational archive. Offspring-only discovery excludes initialization, "
                    "probes, and selection trials. No objective values from observers are reused by evolution.",
                "interpretation": "Full-panel finite-population observations and conditional-discovery "
                    "recursion diagnostics; not a proved mean-field contraction or Wasserstein approximation.",
                "python": platform.python_version(), "numpy": np.__version__, "workers": workers}
    write_json_exclusive(folder/"manifest.json", manifest)
    jobs = [(spec, ii, family, fi, population, repetitions, generations, base, str(folder))
            for ii, spec in enumerate(specs) for fi, family in enumerate(model.LAW_FAMILIES)
            for population in populations]
    cases = {}
    try:
        if workers == 1:
            results = map(run_case, jobs)
            for key, result in results:
                cases[key] = result
                print(key, "complete in", round(result["elapsed_seconds"], 2), "s", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run_case, job) for job in jobs]
                try:
                    for future in as_completed(futures):
                        key, result = future.result()
                        cases[key] = result
                        print(key, "complete in", round(result["elapsed_seconds"], 2), "s", flush=True)
                except Exception:
                    # Preserve failed/finished outputs, but do not launch the
                    # remaining matrix after a numerical or accounting failure.
                    for future in futures:
                        future.cancel()
                    raise
        assert len(cases) == len(specs)*len(model.LAW_FAMILIES)*len(populations)
        assert sha256(__file__) == sources["runner"] and sha256(PROTOCOL) == sources["protocol"]
        assert sha256(model.__file__) == sources["one_step_study"]
        assert sha256(model.objectives.__file__) == sources["objectives"]
    except Exception as error:
        write_json_exclusive(folder/"failure.json", {"error_type": type(error).__name__,
                             "message": str(error), "completed_cases": sorted(cases),
                             "elapsed_seconds": time.monotonic()-started,
                             "policy": "Retain every partial output; no replacement or silent restart."})
        raise
    summary = dict(manifest, cases=dict(sorted(cases.items())), elapsed_seconds=time.monotonic()-started)
    for stream in ("evolution", "probe", "audit"):
        summary[stream+"_total_objective_calls"] = sum(c[stream+"_total_objective_calls"] for c in cases.values())
    summary["diagnostic_total_objective_calls"] = (summary["probe_total_objective_calls"]
                                                   + summary["audit_total_objective_calls"])
    write_json_exclusive(folder/"summary.json", summary)
    print("PASS: full declared matrix; finite values; monotone archives; call accounting; "
          "all seeds/initial states and observer outcomes saved", flush=True)
    print("Results:", folder, "elapsed seconds:", round(summary["elapsed_seconds"], 2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Read-only mathematical/sampler regression checks")
    mode.add_argument("--smoke", action="store_true", help="All instances/families/assemblies/orders; 3 runs, 3 generations, N=16,32")
    mode.add_argument("--run", action="store_true", help="The frozen full experiment; never selected implicitly")
    parser.add_argument("--output", type=Path, help="New output directory; existing directories are rejected")
    parser.add_argument("--workers", type=int, default=1, help="Independent case/N processes; default 1")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.check:
        if args.output is not None:
            parser.error("--check does not write experiment outputs")
        check()
    else:
        run(smoke=args.smoke, output=args.output, workers=args.workers)
