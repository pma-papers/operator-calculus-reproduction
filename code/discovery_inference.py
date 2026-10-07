"""Finite-sample inference for the conditional discovery recursion.

No mean-field approximation is inferred. Confidence is pointwise unless an
explicit multiplicity allocation is supplied. All functions are pure.
"""
import math
import numpy as np
from scipy.stats import beta, binom

PROBES = 128
ETA = .001
Q0 = .25
ALPHA = .05
TAIL_CUTOFF = .001
ENDPOINTS = (25, 50, 100, 200)


def cp_lower(successes, trials, error):
    counts = np.asarray(successes)
    assert np.all((0 <= counts) & (counts <= trials))
    result = np.zeros(counts.shape, dtype=float)
    positive = counts > 0
    result[positive] = beta.ppf(error, counts[positive], trials-counts[positive]+1)
    return result


def cp_upper(successes, trials, error):
    counts = np.asarray(successes)
    assert np.all((0 <= counts) & (counts <= trials))
    result = np.ones(counts.shape, dtype=float)
    below = counts < trials
    result[below] = beta.isf(error, counts[below]+1, trials-counts[below])
    return result


def low_mass_flags(probe_counts, probes=PROBES, eta=ETA, q0=Q0):
    lookup = cp_lower(np.arange(probes+1), probes, eta)
    return lookup[np.asarray(probe_counts)] < q0


def discovery_bound(flag_counts, repetitions, batch, endpoint,
                    alpha=ALPHA, eta=ETA, q0=Q0):
    """Time is the last axis; flags refer to proposal generations 1,...,m.

    Returns the corrected bound, flagged-mass diagnostics and fixed-tail
    bookkeeping. Input may have any preceding configuration axes.
    """
    flag_counts = np.asarray(flag_counts)
    assert 1 <= endpoint <= flag_counts.shape[-1]
    assert np.all((0 <= flag_counts) & (flag_counts <= repetitions))
    rho = (1-q0)**batch
    retained = min(endpoint, max(1, math.ceil(math.log(TAIL_CUTOFF)/math.log(rho))))
    weights = rho**np.arange(retained)
    last = flag_counts[..., endpoint-retained:endpoint][..., ::-1]
    lookup = cp_upper(np.arange(repetitions+1), repetitions, alpha/retained)
    recent = np.sum(weights*np.minimum(1., lookup[last]+eta), axis=-1)
    omitted = rho**retained*(-np.expm1((endpoint-retained)*np.log(rho)))/(1-rho)
    sampling = rho**endpoint
    frequency = flag_counts[..., :endpoint]/repetitions
    all_weights = rho**np.arange(endpoint-1, -1, -1)
    discounted = np.sum(frequency*all_weights, axis=-1)
    undiscounted = frequency.sum(axis=-1)
    return {"upper": np.minimum(1., sampling+recent+omitted),
            "flag_discounted": discounted, "flag_undiscounted": undiscounted,
            "sampling": sampling, "omitted_tail": omitted,
            "retained_generations": retained, "rho": rho,
            "recent_confidence_term": recent}


def checks():
    lower = cp_lower(np.arange(PROBES+1), PROBES, ETA)
    assert lower[0] == 0 and np.all(np.diff(lower) > 0)
    # Exact enumeration verifies the first diagnostic coverage, not a simulation.
    for q in np.r_[np.linspace(.001, .999, 199), Q0]:
        probability = binom.pmf(np.arange(PROBES+1), PROBES, q)
        assert probability[lower > q].sum() <= ETA+1e-12
        flag_probability = probability[lower < Q0].sum()
        assert float(q < Q0) <= flag_probability+ETA+1e-12
    for error in (.05, .05/25, .05/6400):
        upper = cp_upper(np.arange(129), 128, error)
        assert upper[-1] == 1 and np.all(np.diff(upper) > 0)
        for p in np.linspace(.001, .999, 199):
            assert binom.pmf(np.arange(129), 128, p)[upper < p].sum() <= error+1e-12
    flag_counts = np.zeros((2, 200), dtype=int)
    for B in (1, 4, 16, 32, 64, 128):
        result = discovery_bound(flag_counts, 128, B, 200)
        rho, J = result["rho"], result["retained_generations"]
        expected_tail = np.sum(rho**np.arange(J, 200))
        assert np.isclose(result["omitted_tail"], expected_tail)
        assert np.all(result["upper"] > 0)
        all_bad = discovery_bound(np.full((200,), 128), 128, B, 200)
        assert np.isclose(all_bad["upper"], 1.)
        first = discovery_bound(np.zeros((200,), dtype=int), 128, B, 1)
        assert first["retained_generations"] == 1 and first["omitted_tail"] == 0
        print(f"B={B}: best possible pointwise95% upper={result['upper'][0]:.6f}, J={J}")
    print("PASS: exact binomial coverage grids, low-mass inclusion, CP monotonicity, discounted tails")


if __name__ == "__main__":
    checks()
