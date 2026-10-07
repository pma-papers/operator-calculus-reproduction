# Finite populations, conditional discovery, and component contrasts

Frozen before the full run. The study extends the common canonical-assembly experiment, not a named optimizer. It combines interacting-population diagnostics with independent controlled discovery batches. It does not presume the state-law approximation or mean-field concentration hypotheses of Corollary 4.2.

## Fixed experimental matrix

Use all ten objective/dimension instances from the one-step common panel, both original initialization distributions, all eight S/R/H switch settings (D always present), and both S/R orders. The objectives, finite-difference drift, cutoff, bounded selection score, midpoint recombination, and Gaussian diffusion are unchanged. Step size is 0.1, finite-difference radius 0.001, active recombination/selection strengths 1, active diffusion standard deviation 0.2. Use N = 16, 32, 64, 128, 128 independent repetitions, and 200 generations in every cell. The primary objective tolerance is 0.1; 0.01 and 1 are predeclared sensitivities. No failed case is replaced or omitted.

Initial populations are newly drawn from the same broad Gaussian and two-cluster distributions; they are not claimed to be the saved one-step laws. For each case and repetition, use common initial points across assemblies/orders and nested prefixes across N. Fix distinct seed namespaces for initialization, evolution, target-mass probes, and audit batches. Independence is required across repetitions and between the three sampling streams. Sharing randomness across configurations is allowed and must be described in the run metadata.

At each generation the assembled proposal law is obtained from the current empirical population. For S then R, sample selected parents from the exact exponential weights and apply midpoint mixing with probability 0.1 when R is active. For R then S with both active, sample the recombination proposal and accept with probability exp(-0.1 Phi), where Phi=f/(1+f); repeat until accepted. This exactly normalizes selection without enumerating N squared parent pairs. Identity cases use the direct categorical sampler. Apply the unchanged finite-difference drift and optional Gaussian noise last. The N operational offspring are conditionally iid from this proposal law and become the next population.

## Three independent streams and accounting

1. Evolution: N offspring per generation. Record the population mean gap, good fraction, spread, passive best-so-far gap, and cumulative offspring-only non-discovery. The passive archive includes initialization, rejected/accepted selection-score evaluations, finite-difference probes, and offspring outputs, but never influences evolution. The offspring-only event deliberately excludes these additional opportunities, making it the event directly addressed by the batch recursion.
2. Probability probes: 128 independent candidates from the same pre-update law. Save counts below all three tolerances. They estimate the actual finite-population proposal success probability, not the limiting mean-field probability or a Wasserstein error.
3. Controlled audit: 32 further independent candidates, with nested batch prefixes B=1,4,16,32. Save cumulative non-discovery for each B and tolerance. This separates the number of fresh tested candidates B from the state-population size N while keeping the evolution unchanged.

Count actual objective calls, including finite differences and rejection scores. Reuse cached current objective values when possible. Report evolution cost and both diagnostic costs separately. Diagnostic candidates do not enter the optimization archive; no diagnostic cost is presented as operational search cost. The audit batches are an auxiliary experiment, not a cost-free improvement of the optimizer.

## Predeclared inference

Let q_j be the actual pre-batch proposal probability of f <= epsilon, q0=1/4, a_j=P(q_j<q0), and rho=(1-q0)^B. The manuscript's conditional discovery recursion gives

S_m <= rho^m + sum_{j=1}^m rho^(m-j) a_j.

Use B=N for operational offspring and B=1,4,16,32 for the independent audit. The inequality permits adaptive evolution and does not assume independent operational batches. Do not substitute a product of random success probabilities for operational failure.

For each run and generation compute a one-sided exact binomial lower confidence limit L for q_j with error eta=0.001, and flag Z=1{L<q0}. Then a_j <= E[Z]+eta. Across the 128 independent repetitions, use one-sided exact binomial upper confidence limits for E[Z]. At a specified endpoint m, let J=min(m,max(1,ceil(log(0.001)/log(rho)))). Allocate error alpha/J to each of the last J limits, where alpha=0.05, and bound earlier a_j by 1. The resulting upper bound is

min(1, rho^m + sum_{l=0}^{J-1} rho^l min(1,U_{m-l}+eta) + sum_{l=J}^{m-1} rho^l).

It has at least 95% coverage for a specified configuration/tolerance/batch/endpoint, not simultaneously across all figures or cells. Use generation endpoints 25,50,100,200 for summary tables. For the final primary-tolerance matrix, additionally compute a simultaneous version allocating alpha across all 1,280 population/configuration cells and five batch-stream choices. Retain vacuous bounds as results; do not retune parameters or target levels to make them nonvacuous. Empirical discounted and undiscounted flagged-mass sums and the sampling term are diagnostics, not probability bounds without their confidence corrections.

The experiment tests the discovery recursion and quantitatively diagnoses finite-population target mass. It does not certify the limiting reference trajectory, the theorem-derived burn-in, the boundary-strip constants, or the uniform state-law approximation A_N(T). Fixed positive diffusion can cause residual mean-gap/spread floors.

## Presentation and dynamic contrasts

Keep the complete-panel evidence primary. Present all problem/dimension and assembly/order cells in population-size/discovery summaries. Distinguish offspring discovery from the archive and substantial current proposal mass. Small N can change exploration or cause collapse, so monotonic improvement with N is not assumed. In addition to the fixed-generation comparisons, report archive discovery at actual objective-call budgets 1,024, 2,048, 4,096, and 8,192. Use only the last fully completed generation within a budget (including the initialized population), with no interpolation or information from an incompletely affordable generation. These conservative generation-end measurements count selection/finite-difference/output costs but exclude independent diagnostics.

For continuity with the earlier dynamic illustration, retain the unequal-wells and quartic 4D illustrations, both initialization families, now using the N=32 cells and all 128 new matched starts. Record selection-, recombination-, and diffusion-off contrasts against the full assembly. Plot the selection/recombination contrasts separately from diffusion so its larger magnitude does not hide them. For mean gap, archived gap, and population variance, plot the mean within-run difference (component omitted minus full), with pointwise percentile-bootstrap 95% bands from 2,000 resamples of run indices. Use a fixed bootstrap seed recorded in the plotting source. Linear vertical scales and log(1+generation) horizontal spacing retain all 200 generations while expanding the early transient. Include diffusion contrasts and absolute full-assembly values in the numeric audit. Audit times are 0,1,5,10,25,50,100,200. These bootstrap bands are approximate and pointwise, not simultaneous significance tests. Do not change operator strengths, select runs by outcome, or select a favorable new objective for these illustrations. Report small effects honestly. These plots illustrate mechanisms, not universal operator rankings or asymptotic convergence.

## Reproducibility

Smoke/check runs use separate seeds and directories. Commit this protocol and the tested simulation source before executing the full matrix. Save source/protocol hashes, seed metadata, complete metric/count arrays, final populations, and objective-call accounting. Preserve the earlier illustration, its data, and all earlier manuscripts. New presentation/analysis scripts may be improved after the full run, but any substantive deviation from this protocol must be documented rather than hidden.
