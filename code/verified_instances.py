"""Algorithm-level trajectories of the three verified instances (verified-instances protocol).

Panel A: isotropic CBO at fixed temperature; target decomposition V <= 2W + 2b.
Panel B: the manuscript's CMA-ES-type example on 1 - cos(y); expected flow
         versus the certified rate 1/4, plus finite-step stochastic updates.
Panel C: derivative-free recombinative ES on panel objectives versus the
         certified, unfitted mean-gap envelopes.
Run --check, then --smoke, and freeze/commit the protocol before --run.
Output directories are created exclusively: no run is overwritten.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.special import ndtr, erf

import objectives as panel


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/"protocols/verified-instances.md"
FULL_SEED, SMOKE_SEED, CHECK_SEED = 4072600000, 4172600000, 4272600000
SPECS = {row["id"]: row for row in panel.instances()}

# Panel A: CBO.
CBO = dict(instance="separable_pl_d4", lam=1., sigma=.5, alpha=50., dt=.01, steps=2000, N=128)
# Panel B: CMA-ES-type example (manuscript: d=1, two offspring, best one selected).
CMA_RATES = dict(c_m=1., c_sigma=1., c_c=1., eta_sigma=1/8, c_1=1/8, c_mu=1/8)
CMA_T, CMA_TAUS = 40., (.05, .2)
CMA_INITIAL = [  # (m, sigma, C, p_sigma, p_c); all in the invariant region
    (.25, .5, 1., 0., 0.),
    (.25, .5, 1., .5, -.5), (.25, .5, 1., -.5, .5),
    (-.25, .5, 1., .5, -.5), (-.25, .5, 1., -.5, .5)]
# Panel C: recombinative ES.
RES_INSTANCES = ("separable_pl_d4", "rotated_anisotropic_pl_d4", "quartic_d4")
RES = dict(t=.5, r=1e-3, selection_rate=.2, N=32, K=60, epsilon=.1)
QUARTIC_BOX = (.4, 1.2)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def f(x, instance):
    return panel.objective(np.asarray(x, dtype=float), SPECS[instance])


# ---------------------------------------------------------------- Panel A
def cbo_run(rng, runs, steps):
    spec = SPECS[CBO["instance"]]
    d, lam, sigma, alpha, dt, N = spec["dimension"], CBO["lam"], CBO["sigma"], CBO["alpha"], CBO["dt"], CBO["N"]
    target = np.asarray(spec["target"])
    X = rng.normal(.6, 1.2, size=(runs, N, d))
    out = np.empty((runs, steps+1, 4))  # V_*, W_alpha, b_alpha, mean gap
    for k in range(steps+1):
        values = f(X, CBO["instance"])
        w = np.exp(-alpha*(values-values.min(axis=1, keepdims=True)))
        v = np.einsum("rn,rnd->rd", w, X)/w.sum(axis=1)[:, None]
        out[:, k, 0] = np.mean(np.sum((X-target)**2, axis=-1), axis=1)
        out[:, k, 1] = np.mean(np.sum((X-v[:, None])**2, axis=-1), axis=1)
        out[:, k, 2] = np.sum((v-target)**2, axis=-1)
        out[:, k, 3] = values.mean(axis=1)-spec["f_star"]
        if k == steps:
            break
        diff = X-v[:, None]
        X = X-lam*diff*dt+sigma*np.linalg.norm(diff, axis=-1, keepdims=True)*math.sqrt(dt)*rng.normal(size=X.shape)
    assert np.all(out[..., 0] <= 2*out[..., 1]+2*out[..., 2]+1e-12), "V <= 2W + 2b violated"
    return out


# ---------------------------------------------------------------- Panel B
NODES, WEIGHTS = np.polynomial.legendre.leggauss(80)


def z_quadrature(m, s, periodic):
    """Gauss-Legendre on [-12, 12], split where dist(y, 2 pi Z) (or |y|) has kinks."""
    kinks = np.arange(-8, 9)*np.pi if periodic else np.array([0.])
    cuts = np.unique(np.clip(np.r_[-12., 12., (kinks-m)/s], -12, 12))
    lo, hi = cuts[:-1, None], cuts[1:, None]
    return ((hi+lo)/2+(hi-lo)/2*NODES).ravel(), ((hi-lo)/2*WEIGHTS).ravel()


def selected_moments(m, s, periodic=True):
    """Exact E[Z_sel], E[Z_sel^2] for the better of two candidates m + s Z_i.

    E phi(Z_sel) = 2 int phi(z) n(z) P(f(m + s Z') > f(m + s z)) dz.  For
    f = 1 - cos the event is dist(y', 2 pi Z) > dist(y, 2 pi Z); for the
    quadratic reference it is |y'| > |y|.
    """
    Z_GRID, Z_WEIGHTS = z_quadrature(m, s, periodic)
    y = m+s*Z_GRID
    delta = np.abs((y+np.pi) % (2*np.pi)-np.pi) if periodic else np.abs(y)
    periods = np.arange(-int(12*s/(2*np.pi))-2, int(12*s/(2*np.pi))+3) if periodic else np.array([0])
    centers = 2*np.pi*periods[:, None]
    inside = ndtr((centers+delta-m)/s)-ndtr((centers-delta-m)/s)
    better = np.clip(1-inside.sum(axis=0), 0, 1)
    density = np.exp(-Z_GRID**2/2)/math.sqrt(2*math.pi)
    base = 2*Z_WEIGHTS*density*better
    return float(base@Z_GRID), float(base@Z_GRID**2)


def cma_rhs(_, state):
    m, sigma, C, p_sigma, p_c = state
    s = sigma*math.sqrt(C)
    g, h = selected_moments(m, s)
    r = CMA_RATES
    return [r["c_m"]*s*g,
            sigma*r["eta_sigma"]*(p_sigma**2-1),
            r["c_1"]*(p_c**2-C)+r["c_mu"]*(C*h-C),
            -r["c_sigma"]*p_sigma+math.sqrt(r["c_sigma"]*(2-r["c_sigma"]))*g,
            -r["c_c"]*p_c+math.sqrt(r["c_c"]*(2-r["c_c"]))*math.sqrt(C)*g]


def in_region(m, sigma, C, p_sigma, p_c, slack=1e-9):
    s = sigma*np.sqrt(C)
    return ((s > 0) & (s <= .5+slack) & (np.abs(m) <= s/2+slack) & (np.abs(p_sigma) <= .5+slack)
            & (np.abs(p_c) <= np.sqrt(C)/2+slack))


def cma_flow(times):
    L, region = [], []
    for state in CMA_INITIAL:
        sol = solve_ivp(cma_rhs, (0, CMA_T), state, t_eval=times, method="DOP853", rtol=1e-10, atol=1e-12)
        assert sol.success, sol.message
        m, sigma, C, p_sigma, p_c = sol.y
        L.append(m**2+sigma**2*C)
        region.append(in_region(m, sigma, C, p_sigma, p_c))
    return np.array(L), np.array(region)


def cma_stochastic(rng, tau, runs):
    steps = int(round(CMA_T/tau))
    r = CMA_RATES
    L = np.empty((len(CMA_INITIAL), runs, steps+1))
    stayed = np.empty((len(CMA_INITIAL), runs), dtype=bool)
    positive = np.empty((len(CMA_INITIAL), runs), dtype=bool)
    for i, state in enumerate(CMA_INITIAL):
        m, sigma, C, p_sigma, p_c = (np.full(runs, v) for v in state)
        ok, pos = np.ones(runs, bool), np.ones(runs, bool)
        L[i, :, 0] = m**2+sigma**2*C
        for k in range(steps):
            Z = rng.normal(size=(runs, 2))
            Y = m[:, None]+(sigma*np.sqrt(C))[:, None]*Z
            z = np.where(1-np.cos(Y[:, 0]) <= 1-np.cos(Y[:, 1]), Z[:, 0], Z[:, 1])
            y_w = np.sqrt(C)*z
            m, p_sigma, p_c, sigma, C = (
                m+tau*r["c_m"]*sigma*y_w,
                p_sigma+tau*(-r["c_sigma"]*p_sigma+math.sqrt(r["c_sigma"]*(2-r["c_sigma"]))*z),
                p_c+tau*(-r["c_c"]*p_c+math.sqrt(r["c_c"]*(2-r["c_c"]))*y_w),
                sigma+tau*r["eta_sigma"]*sigma*(p_sigma**2-1),
                C+tau*(r["c_1"]*(p_c**2-C)+r["c_mu"]*(C*z*z-C)))
            pos &= (sigma > 0) & (C > 0)
            sigma, C = np.maximum(sigma, 1e-300), np.maximum(C, 1e-300)
            ok &= in_region(m, sigma, C, p_sigma, p_c)
            L[i, :, k+1] = m**2+sigma**2*C
        stayed[i], positive[i] = ok, pos
    return L, stayed, positive


# ---------------------------------------------------------------- Panel C
def res_constants(instance):
    spec = SPECS[instance]
    d, r, b = spec["dimension"], RES["r"], 1.05
    if spec["family"] == "quartic":
        a, bU = QUARTIC_BOX
        mu, L, kappa, q0 = 8*a*a, 12*bU*bU-4, 4-12*a*a, (1+a)**2
        M3 = 24*(bU+r)
    else:
        weights = np.asarray(spec["weights"])
        amin, amax = weights.min(), weights.max()
        mu, L = amin*(2-2*b/math.pi)**2/(2*(1+b)), 2*amax*(1+b)
        kappa, q0, M3 = 2*amax*(b-1), amin, 4*amax*b
    e = math.sqrt(d)*M3*r*r/6
    h = RES["t"]/L
    rho_mut = 1-2*mu*h*(.9-.55*L*h)
    c_mut = (2.5*h+5.5*L*h*h)*e*e
    p = min(1., 3*q0/kappa*(1/rho_mut-1)/2)
    rho_rec = 1+kappa*p/(3*q0)
    rho = rho_rec*rho_mut
    assert 0 < rho_mut < 1 and rho < 1
    return dict(mu_PL=mu, L=L, kappa=kappa, q0=q0, M3=M3, surrogate_error=e, h=h, rho_mut=rho_mut,
                c_mut=c_mut, blend_probability=p, rho_rec=rho_rec, rho_RES=rho)


def central_difference(x, instance, r):
    d = x.shape[-1]
    g = np.empty_like(x)
    for j in range(d):
        step = np.zeros(d)
        step[j] = r
        g[..., j] = (f(x+step, instance)-f(x-step, instance))/(2*r)
    return g


def res_run(rng, instance, runs, K):
    spec, const = SPECS[instance], res_constants(instance)
    d, N = spec["dimension"], RES["N"]
    quartic = spec["family"] == "quartic"
    if quartic:
        X = rng.uniform(*QUARTIC_BOX, size=(runs, N, d))
    else:
        X = rng.normal(.6, 1.2, size=(runs, N, d))
    gaps = np.empty((runs, K+1))
    best = np.empty((runs, K+1))
    inside = True
    for k in range(K+1):
        values = f(X, instance)-spec["f_star"]
        gaps[:, k], best[:, k] = values.mean(axis=1), values.min(axis=1)
        if quartic:
            inside &= bool(np.all((X >= QUARTIC_BOX[0]) & (X <= QUARTIC_BOX[1])))
        if k == K:
            break
        w = np.exp(-RES["selection_rate"]*values)
        cdf = np.cumsum(w/w.sum(axis=1, keepdims=True), axis=1)
        cdf[:, -1] = 1.
        parents = [np.minimum((cdf[:, None, :] < rng.random((runs, N, 1))).sum(-1), N-1) for _ in range(2)]
        XI = np.take_along_axis(X, parents[0][..., None], axis=1)
        XJ = np.take_along_axis(X, parents[1][..., None], axis=1)
        blend = rng.random((runs, N, 1)) < const["blend_probability"]
        xi = np.where(blend, rng.random((runs, N, 1)), (rng.random((runs, N, 1)) < .5).astype(float))
        Z = xi*XI+(1-xi)*XJ
        X = Z-const["h"]*central_difference(Z, instance, RES["r"])
        if quartic:
            inside &= bool(np.all((Z >= QUARTIC_BOX[0]) & (Z <= QUARTIC_BOX[1])))
    rho, c = const["rho_RES"], const["c_mut"]
    k = np.arange(K+1)
    envelope = rho**k*gaps[:, :1]+c*(1-rho**k)/(1-rho)
    return gaps, best, envelope, inside


# ---------------------------------------------------------------- checks and runs
def checks():
    results = {}
    # Ranked moments: quadratic reference against the closed forms.
    worst = 0.
    for a in np.linspace(-.5, .5, 11):
        g, h = selected_moments(a, 1., periodic=False)
        worst = max(worst, abs(g+erf(a)/math.sqrt(math.pi)), abs(h-(1-2/math.pi*math.exp(-a*a))))
    assert worst < 1e-10, worst
    results["ranked_moment_closed_form_error"] = worst
    # Periodic moments against Monte Carlo.
    rng = np.random.default_rng(CHECK_SEED)
    Z = rng.normal(size=(2_000_000, 2))
    for m, s in ((.25, .5), (-.2, .45), (.1, .3)):
        y = m+s*Z
        zs = np.where(1-np.cos(y[:, 0]) <= 1-np.cos(y[:, 1]), Z[:, 0], Z[:, 1])
        g, h = selected_moments(m, s)
        assert abs(g-zs.mean()) < 5e-3 and abs(h-(zs**2).mean()) < 5e-3, (m, s, g, zs.mean())
    results["periodic_moments_match_monte_carlo"] = True
    # Quartic central difference identity from the manuscript.
    x = rng.uniform(.4, 1.2, size=(5, 4))
    r = RES["r"]
    assert np.allclose(central_difference(x, "quartic_d4", r), 4*x*(x*x-1+r*r), rtol=1e-6, atol=1e-8)
    results["quartic_central_difference_identity"] = True
    results["res_constants"] = {name: res_constants(name) for name in RES_INSTANCES}
    # Short runs of every component.
    cbo_run(rng, 2, 20)
    L, stayed, _ = cma_stochastic(rng, .05, 4)
    res_run(rng, "quartic_d4", 2, 5)
    results["short_runs"] = "passed"
    return results


def run(seed, cbo_runs, cbo_steps, cma_runs, res_runs, K):
    rng = {name: np.random.default_rng([seed, i]) for i, name in enumerate(("cbo", "cma", "res"))}
    started = time.monotonic()
    arrays, summary = {}, {"cbo": {}, "cma": {}, "res": {}}
    arrays["cbo"] = cbo_run(rng["cbo"], cbo_runs, cbo_steps)
    summary["cbo"] = {**CBO, "runs": cbo_runs, "steps": cbo_steps,
                      "centered_coefficient": 2*CBO["lam"]-SPECS[CBO["instance"]]["dimension"]*CBO["sigma"]**2,
                      "quantities": ["V_star", "W_alpha", "b_alpha", "mean_gap"],
                      "decomposition_V_le_2W_plus_2b_holds": True}
    times = np.linspace(0, CMA_T, 801)
    L_flow, region = cma_flow(times)
    ratio = L_flow*np.exp(times/4)/L_flow[:, :1]
    arrays.update(cma_times=times, cma_flow_L=L_flow, cma_flow_region=region)
    summary["cma"] = {"rates": CMA_RATES, "initial_states": CMA_INITIAL, "horizon": CMA_T,
                      "certified_rate": .25, "max_L_exp_t_over_4_ratio": float(ratio.max()),
                      "flow_region_invariant": bool(region.all()),
                      "flow_certificate_holds": bool(ratio.max() <= 1+1e-8 and region.all()),
                      "empirical_flow_rate_fit_free_final": (-np.log(L_flow[:, -1]/L_flow[:, 0])/CMA_T).tolist(),
                      "finite_step": {}}
    for tau in CMA_TAUS:
        L, stayed, positive = cma_stochastic(rng["cma"], tau, cma_runs)
        arrays[f"cma_tau{tau}_L"] = L
        arrays[f"cma_tau{tau}_stayed"] = stayed
        summary["cma"]["finite_step"][str(tau)] = {
            "runs_per_initial_state": cma_runs,
            "fraction_remaining_in_region": stayed.mean(axis=1).tolist(),
            "fraction_positive_sigma_C": positive.mean(axis=1).tolist(),
            "median_final_L_over_L0": np.median(L[:, :, -1]/L[:, :, 0], axis=1).tolist(),
            "mean_final_L_over_L0": np.mean(L[:, :, -1]/L[:, :, 0], axis=1).tolist()}
    for name in RES_INSTANCES:
        gaps, best, envelope, inside = res_run(rng["res"], name, res_runs, K)
        mean, se = gaps.mean(axis=0), gaps.std(axis=0, ddof=1)/math.sqrt(res_runs)
        bound = envelope.mean(axis=0)
        margin = mean-bound-2.33*se
        eps = RES["epsilon"]
        failure = np.mean(np.minimum.accumulate(best, axis=1) > eps, axis=0)
        arrays[f"res_{name}_gaps"], arrays[f"res_{name}_envelope"], arrays[f"res_{name}_best"] = gaps, envelope, best
        summary["res"][name] = {**res_constants(name), "runs": res_runs, "generations": K, "N": RES["N"],
                                "max_excess_mean_minus_envelope_minus_2p33se": float(margin.max()),
                                "certificate_consistent": bool(margin.max() <= 0),
                                "max_mean_over_envelope": float((mean/bound).max()),
                                "box_persistence": bool(inside) if name.startswith("quartic") else None,
                                "no_eps_good_fraction_at": {str(k): float(failure[k]) for k in sorted({0, 10, 25, 50, K}) if k <= K},
                                "markov_bound_at": {str(k): float(min(1, bound[k]/eps)) for k in sorted({0, 10, 25, 50, K}) if k <= K}}
    summary["elapsed_seconds"] = time.monotonic()-started
    return summary, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Formula and implementation checks; write nothing")
    mode.add_argument("--smoke", action="store_true", help="Reduced runs in a separate directory")
    mode.add_argument("--run", action="store_true", help="The frozen full protocol")
    parser.add_argument("--output", type=Path, help="New output directory; existing directories are rejected")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(checks(), indent=2))
        print("PASS: ranked moments, periodic moments, quartic differences, constants, short runs")
        return
    if not PROTOCOL.is_file():
        raise FileNotFoundError("Freeze the declared protocol before running: "+str(PROTOCOL))
    folder = args.output.resolve() if args.output else ROOT/"results/verified-instances"/(
        "verified-instances-smoke" if args.smoke else "verified-instances-full")
    folder.mkdir(parents=True, exist_ok=False)
    check_results = checks()
    if args.smoke:
        summary, arrays = run(SMOKE_SEED, 4, 200, 8, 8, 20)
    else:
        summary, arrays = run(FULL_SEED, 64, CBO["steps"], 256, 256, RES["K"])
    summary.update(status="smoke" if args.smoke else "full", checks=check_results,
                   sources_sha256={"runner": sha256(__file__), "objectives": sha256(panel.__file__),
                                   "protocol": sha256(PROTOCOL)},
                   python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    with (folder/"summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (folder/"arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    print(json.dumps({key: summary[key] for key in ("cma", "res")}, indent=1)[:4000])
    print("Output:", folder)


if __name__ == "__main__":
    main()
