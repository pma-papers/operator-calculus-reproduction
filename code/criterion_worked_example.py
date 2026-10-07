"""Worked evaluation of criterion (7)-(8) on separable PL, d=4 (Appendix H).

Takes the reference-run modulus estimate of reference_run_modulus.py and
computes every other ingredient of the criterion for the identity-kernel
assemblies of the frozen trajectory study: L_K, the level-band constants
L_{T,eps}, r_0, kappa, H, the two density bounds M (worst-case Gaussian-mixture
bound and empirical at generation 200), r_*, the burn-in index from measured
success mass, the discount sum, and the left-hand side of (7).  Reports the
resulting failure bound next to the observed failure and the measured
(Appendix G.6) bound for the same cell.  Writes results/analysis/criterion_worked_example.json.
"""
from pathlib import Path
import json
import math

import numpy as np

import objectives as panel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"results/trajectories/finite-population-full"
OUT = ROOT/"results/analysis/criterion_worked_example.json"
CASE, ASSEMBLY, ORDER = "separable_pl_d4__broad", "S1_R1_H1", "MRS"
EPS, ETA_LEV, DELTA = .1, .05, .05
BINS = 20                    # sub-bands for the empirical density bound
SIGMA, TAU = .2, .1          # Table 2: H adds sigma*sqrt(tau)*Z
SPEC = next(r for r in panel.instances() if r["id"] == "separable_pl_d4")


def f(x):
    return panel.objective(np.asarray(x, dtype=float), SPEC)


def grad(x):
    return 2*x+1.05*np.sin(2*x)


def main():
    d = SPEC["dimension"]
    rng = np.random.default_rng(4272600002)
    # ---- level band D = {eps-eta <= f <= eps+eta}: sample it by rejection from a box
    pts = rng.uniform(-.6, .6, size=(4_000_000, d))
    vals = f(pts)
    band = pts[(vals >= EPS-ETA_LEV) & (vals <= EPS+ETA_LEV)]
    gnorm = np.linalg.norm(grad(band), axis=1)
    kappa, L_band = float(gnorm.min()), float(gnorm.max())
    # Lipschitz constant on the r0-neighborhood of the level set: the band contains it if L*r0 <= eta
    r0 = ETA_LEV/L_band
    # level-set area H(u) = sup over u in band of the (d-1)-measure of {f=u}: coarea estimate on the band
    # H(u) ~ (volume of {u-h<=f<=u+h})/(2h) * (average |grad f|) ; take the sup over a grid of u
    box_vol = 1.2**d
    Hs = []
    for u in np.linspace(EPS-ETA_LEV, EPS+ETA_LEV, 11):
        h = .005
        sel = (vals >= u-h) & (vals <= u+h)
        vol = sel.mean()*box_vol
        Hs.append(vol/(2*h)*np.linalg.norm(grad(pts[sel]), axis=1).mean())
    H = float(max(Hs))
    # ---- density bounds of the search law on the band
    rho_worst = (2*math.pi*SIGMA**2*TAU)**(-d/2)      # Gaussian mixture with covariance sigma^2 tau I
    M_worst = rho_worst*H/kappa
    # empirical objective-value density at generation 200 (final populations, all runs)
    summary = json.loads((DATA/"summary.json").read_text())
    ai = [a["id"] for a in summary["assemblies"]].index(ASSEMBLY)
    oi = summary["orders"].index(ORDER)
    results = {}
    for N in (16, 32, 64, 128):
        with np.load(DATA/f"{CASE}__N{N}.npz") as dd:
            fv = dd["final_values"][:, ai, oi].ravel()
            probes = dd["probe_success_counts"][:, ai, oi, :, 0]/128.    # runs, generations
            offspring_fail = 1-dd["offspring_cumulative_hit"][:, ai, oi, -1, 0].mean()
        # empirical density bound: the largest objective-value density over BINS equal
        # sub-bands of the level band (a histogram sup, not the band average, which is
        # only a lower bound on the L-infinity bound that the assumption requires)
        counts, _ = np.histogram(fv, bins=BINS, range=(EPS-ETA_LEV, EPS+ETA_LEV))
        M_emp = float(counts.max()/len(fv)/(2*ETA_LEV/BINS))
        mean_success = probes.mean(axis=0)
        k_eps = int(np.argmax(mean_success >= .5))+1 if (mean_success >= .5).any() else None
        # the corollary needs p >= 1/2 at every generation from k_eps on, not only at the crossing
        stays_above = bool(k_eps is not None and (mean_success[k_eps-1:] >= .5).all())
        results[str(N)] = {"M_empirical_gen200_histogram_sup": M_emp, "histogram_bins": BINS,
                           "k_eps_from_measured_success_mass": k_eps,
                           "success_mass_stays_above_half_after_k_eps": stays_above,
                           "observed_offspring_failure_gen200": float(offspring_fail)}
    ref = json.loads((ROOT/"results/analysis/reference_run_modulus.json").read_text())["cases"][CASE]
    with np.load(ROOT/"results/trajectories/finite_population_analysis.npz") as a:
        cases, pops = a["cases"].tolist(), a["populations"].tolist()
        upper = a["upper"]
    ci = cases.index(CASE)
    m = 200
    L_K = 1.
    for M_name, M in (("worst_case", M_worst), ("empirical_gen200", None)):
        for N in (16, 32, 64):
            r = results[str(N)]
            Mv = M if M is not None else max(r["M_empirical_gen200_histogram_sup"], 1e-12)
            r_star = min(r0, 1/(16*L_band*Mv))
            Ahat = ref[str(N)]["mean_W2sq_vs_reference"]
            Ahat_plus = ref[str(N)]["bootstrap_95pct"][1]
            bprime = ref["fitted_decay_exponent_16_to_64"]
            A_plug = 2*Ahat_plus*(1+(N/128)**bprime)
            discount = 1/(1-math.exp(-N/4))
            lhs = (8*L_K/r_star)**2*A_plug*discount
            needed_A = DELTA/2/((8*L_K/r_star)**2*discount)
            # with Ahat ~ c N^{-1/2}: N_needed = N*(Ahat_plug/needed_A)^2
            N_needed = N*(A_plug/needed_A)**2
            r.setdefault("criterion", {})[M_name] = {
                "M": Mv, "r_star": r_star, "A_plug_in": A_plug, "discount_sum_bound": discount,
                "lhs_of_(7)": lhs, "delta_over_2": DELTA/2, "criterion_satisfied": lhs <= DELTA/2,
                "A_needed_for_delta": needed_A, "N_needed_if_A_decays_like_N_minus_half": N_needed,
                "measured_route_pointwise_95pct_upper_bound_gen200": float(upper[ci, pops.index(N), ai, oi, 0, 0, -1])}
    out = {"case": CASE, "assembly": ASSEMBLY, "order": ORDER, "epsilon": EPS, "delta": DELTA, "eta_lev": ETA_LEV,
           "L_K": L_K, "kappa_band": kappa, "L_band": L_band, "r0": r0, "H_levelset_area_sup": H,
           "rho_worst_gaussian_mixture": rho_worst, "M_worst_case": M_worst, "by_N": results}
    OUT.write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps({k: v for k, v in out.items() if k != "by_N"}, indent=1))
    for N, r in results.items():
        print(N, "k_eps", r["k_eps_from_measured_success_mass"], "obs fail", r["observed_offspring_failure_gen200"], "M_emp", round(r["M_empirical_gen200_histogram_sup"], 4))
        for name, c in r.get("criterion", {}).items():
            print("   ", name, "r*=%.2e" % c["r_star"], "lhs=%.3g" % c["lhs_of_(7)"], "N_needed=%.2e" % c["N_needed_if_A_decays_like_N_minus_half"], "G.6 bound=%.3f" % c["measured_route_pointwise_95pct_upper_bound_gen200"])


if __name__ == "__main__":
    main()
