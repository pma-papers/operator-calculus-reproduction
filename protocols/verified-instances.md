# Algorithm-level trajectories of the three verified instances

Purpose: update the three-panel Lyapunov figure of an earlier version of the manuscript (CBO, CMA-ES, recombinative ES on 4D Ackley) so that each panel runs the algorithm exactly as specified in the current statements and, wherever the manuscript supplies explicit unfitted constants, compares trajectories with the proved envelope. This protocol is fixed before the full run; smoke and check runs use separate seeds and directories.

## Why the earlier figure is not reused unchanged

- Ackley has a non-differentiable cusp at its minimizer; the recombinative-ES and CMA landscape results use smoothness, and the current panel uses smooth objectives.
- All three earlier runs clip to a box; no current statement includes a projection.
- The earlier CBO anneals the inverse temperature to 10^6; the current CBO certificate uses a fixed temperature.
- The earlier recombinative ES adds annealed Gaussian noise and uses a fitted local-regression gradient; the current proposition is noise-free with an explicit surrogate-error bound.
- The earlier recombinative ES plots the mean squared distance; the current Lyapunov quantity is the mean objective gap.
- The earlier CMA-ES uses a unit interpolation step; the current CMA results concern the expected small-step flow.
- The earlier curves are compared with no predicted rate. The current manuscript proves explicit rates for the CMA example and for the recombinative ES on panel objectives; these are tested directly.

## Panel A: CBO with fixed temperature (target decomposition, no envelope)

Isotropic CBO, h = 1, Euler-Maruyama with step dt = 0.01 to T = 20 (2000 steps), no projection. Objective: separable non-convex PL, d = 4 (panel definition). lambda = 1, sigma = 0.5 (2 lambda - d sigma^2 = 1), fixed alpha = 50, weights exp(-alpha (f - min f)) per step. N = 128 particles, R = 64 independent runs, initial particles iid from the panel's broad distribution N(0.6*1, 1.2^2 I).

Recorded per step: V_* = mean ||x - y_*||^2, W_alpha = mean ||x - v_alpha||^2, b_alpha = ||v_alpha - y_*||^2, mean objective gap. The identity V_* <= 2 W_alpha + 2 b_alpha is checked at every step. The centered coefficient 2 lambda - d sigma^2 is shown only as a reference slope.

No certified envelope is drawn: for quadratic-growth objectives, the residual level implied by the current CBO certificate is at least 4 r^2, whereas every law in its entry class has V_* < r^2. This is recorded as a finding, not repaired by choosing constants after the run.

## Panel B: CMA-ES-type flow on the periodic landscape (certified rate 1/4)

Exactly the manuscript example: d = 1, f(y) = 1 - cos(y), y_* = 0, lambda_pop = 2, mu = w_1 = 1, c_m = c_sigma = c_c = 1, eta_sigma = c_1 = c_mu = 1/8. Lyapunov quantity L = (m - y_*)^2 + s^2 with s = sigma sqrt(C). Certified: L(t) <= exp(-t/4) L(0) for the expected flow started in the invariant region.

Initial states (all in the region): the manuscript example (m = 1/4, sigma = 1/2, C = 1, p_sigma = p_c = 0) and four corners (m = +-s/2, sigma = 1/2, C = 1, p_sigma = +-1/2, p_c = -+1/2).

1. Expected flow: the ODE of the manuscript with the exact selected-step moments g_s(a), h_s(a) computed by one-dimensional quadrature of the exact two-offspring ranking probability (sum over periods of normal CDFs). Solved on t in [0, 40] with rtol = 1e-10, atol = 1e-12.
2. Finite-step stochastic update of the manuscript with tau = 0.05 and tau = 0.2 to t = 40, one draw of two offspring per step, R = 256 runs per initial state. Reported: median and 10/90% quantiles of L, the fraction of runs that remain in the invariant region, and the sample mean of L. The certificate does not cover these runs; they show how far a finite stochastic implementation departs from the expected flow.

Pass criterion (expected flow only): max_t L(t) exp(t/4) / L(0) <= 1 + 1e-8 and region invariance at all solver output times.

## Panel C: derivative-free recombinative ES on panel objectives (certified envelopes)

Objectives (panel definitions): separable non-convex PL d = 4, rotated anisotropic PL d = 4, multimodal quartic d = 4 on U = [0.4, 1.2]^4 with y_* = 1. Constants exactly as in the manuscript's nonconvex certificates with dimensionless step t = L h = 0.5, central-difference radius r = 1e-3, selection weight phi(u) = exp(-0.2 u), blending probability p = min{1, p_crit / 2} with uniform interpolation coefficient, parents resampled independently. N = 32, R = 256 runs, K = 60 generations. Initialization: PL objectives from the broad distribution N(0.6*1, 1.2^2 I); quartic iid uniform on U.

The proposition is conditional on the initial population: E[E_k | F_0] <= rho^k E_0 + c_mut (1 - rho^k)/(1 - rho). The reported envelope uses each run's own E_0 and is averaged over runs; the reported trajectory is the run average of E_k. Pass criterion: at every k, the run average does not exceed the averaged envelope by more than the one-sided 99% normal upper margin of the run average (2.33 standard errors), and for the quartic every individual, recombined point and mutation segment remains in U. Also recorded: the fraction of runs with an epsilon-good individual (epsilon = 0.1) and the Markov-bound prediction for that failure probability.

## Reporting

All runs, generations and initial states are retained. No constant is fitted to the observed trajectories. Seeds: full run 4072600000 + offsets, smoke 4172600000, check 4272600000. Outputs: results/verified-instances/verified-instances-full/ (summary.json, arrays.npz), figure figures/verified_instances.pdf.
