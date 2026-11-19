"""Explanatory figures of the article (Figs. 1-5), drawn with the EvoScope toolbox.

    explanatory_operators.pdf            Fig. 1: one step of each operator of the full variant (selection S,
                                         recombination R, descent step D, Gaussian mutation H) on a population,
                                         with the density of one coordinate before and after; the change per unit
                                         time step for decreasing time steps and its limit, the effect; the
                                         effects on broad and on clustered populations
    explanatory_conditions.pdf           Fig. 2: the four conditions (A1)-(A4); for each, one operator that
                                         satisfies it and one that violates it
    explanatory_population_distance.pdf  Fig. 3: the two parts of the distance between populations (motion and
                                         reweighting)
    explanatory_evolution_equation.pdf   Fig. 4: a population at three times; three error measures that are zero
                                         at convergence (mean gap, mean squared distance to an optimum, share not
                                         yet near an optimum), observed and as start value plus the accumulated
                                         effects of the operators (256 runs, N = 256, time step 0.025,
                                         240 generations; about 13 minutes)
    explanatory_convergence_targets.pdf  Fig. 5: four populations that separate the convergence targets, and a
                                         Gaussian search distribution with sampled candidates

The figures are illustrations, not results of the numerical study: they use the operators, test problem
(quartic, d = 4) and initial-population families of the study, but the populations are drawn here with the
fixed seeds below.  The numbers printed in the figures and quoted in their captions are written to
results/analysis/explanatory_figures_audit.json.

Needs EvoScope (https://github.com/pma-papers/evoscope), installed or with its src/ folder given by
--evoscope or the environment variable EVOSCOPE_SRC:

    python code/explanatory_figures.py [--evoscope PATH] [--tau 0.025] [--quick] [FIGURE ...]

FIGURE is one or more of operators, conditions, distance, evolution, targets (default: all).  --quick runs
the evolution figure with N = 64 and 6 runs (a trial, not the figure of the article); a time step other
than 0.025 is saved under its own name.  The simulated arrays of the evolution figure are kept in
results/explanatory-cache/ (not shipped) and reused when the test functions and settings match, so that a
second run takes seconds; delete the folder to recompute.
"""
from pathlib import Path
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT/"figures"
AUDIT_PATH = ROOT/"results/analysis/explanatory_figures_audit.json"
FIGS = ("operators", "conditions", "distance", "evolution", "targets")
parser = argparse.ArgumentParser(description="Explanatory figures of the article (Figs. 1-5).")
parser.add_argument("figures", nargs="*", choices=FIGS, metavar="FIGURE", help=f"one or more of {', '.join(FIGS)}")
parser.add_argument("--evoscope", type=Path, default=os.environ.get("EVOSCOPE_SRC"), help="src/ folder of EvoScope")
parser.add_argument("--tau", type=float, default=.025, help="time step of the evolution figure (default 0.025)")
parser.add_argument("--quick", action="store_true", help="trial run of the evolution figure with N = 64 and 6 runs")
ARGS = parser.parse_args()
if ARGS.evoscope:
    sys.path.insert(0, str(Path(ARGS.evoscope).resolve()))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import evoscope
    from evoscope import attribution, problems
    from evoscope.operators import Assembly, Drift, GaussianNoise, Recombination, Selection, normalized_fd_drift, objective_rate
    from evoscope.population import broad_gaussian, two_cluster
except ImportError:
    sys.exit("EvoScope not found: install it (pip install git+https://github.com/pma-papers/evoscope) "
             "or pass its src/ folder with --evoscope")

WIDTH = 7.16
BLUE, ORANGE, GREEN, RED, DARK = "#246f96", "#c16c23", "#347c63", "#8a2b2b", "#1d2b3a"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
AUDIT = {}


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/name, metadata={"Title": Path(name).stem, "CreationDate": None, "ModDate": None})
    plt.close(fig)
    print("Wrote", OUT/name)


# --------------------------------------------------------------------------- operators and effects
OPCOLOR = {"S": GREEN, "R": ORANGE, "D": BLUE, "H": "#6a4c93", "sum": DARK}


def operators():
    """Top row: one small step of each operator on one population.  Bottom row: the decrease of the mean gap
    per unit time step approaches the effect as the time step shrinks, and the effects on two kinds of
    population."""
    rng = np.random.default_rng(0)
    f = problems.get("quartic_d4")
    ops = {"S": Selection(objective_rate(f.value), name="S"), "R": Recombination(name="R"),
           "D": Drift(normalized_fd_drift(f.value), name="D"), "H": GaussianNoise(.2, name="H")}
    variant = Assembly(list(ops.values()))
    broad = [broad_gaussian(32, 4, rng) for _ in range(128)]
    clustered = [two_cluster(32, 4, rng) for _ in range(128)]
    pop = broad[0]
    V = pop.expect(f.value(pop.points))

    fig = plt.figure(figsize=(WIDTH, 4.45))
    top = [fig.add_axes([.03+.245*k, .625, .2, .322]) for k in range(4)]
    strips = [fig.add_axes([.03+.245*k, .515, .2, .085]) for k in range(4)]
    limit, bars_b, sketch, bars_c = (fig.add_axes([.085, .1, .25, .255]), fig.add_axes([.415, .1, .17, .255]),
                                      fig.add_axes([.612, .085, .15, .255]), fig.add_axes([.805, .1, .185, .255]))
    u = np.linspace(-3.3, 3.3, 260)
    X1, X2 = np.meshgrid(u, u)
    level = np.log1p((X1**2-1)**2+(X2**2-1)**2)
    VIEW = ((-.1, 2.9), (-.78, 2.22))                           # window of the panels (a)-(d)

    def background(ax, title, lim=2.45):
        ax.contour(X1, X2, level, levels=14, colors="0.8", linewidths=.5)
        ax.scatter(*np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]).T, marker="*", s=30, color="0.5", zorder=2)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xticks=[], yticks=[])
        ax.set_title(title, fontsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ---- (a)-(d): one enlarged step of each operator on a two-dimensional population, with its ellipse
    f2 = lambda y: np.sum((y**2-1)**2, axis=-1)
    grad2 = lambda y: 4*y*(y**2-1)
    rng2 = np.random.default_rng(4)
    center, spread = np.array([1.62, .72]), np.array([.42, .36])
    small = center+spread*rng2.normal(size=(60, 2))             # the individuals that are drawn
    large = center+spread*rng2.normal(size=(6000, 2))           # a large population of the same kind, for the densities
    KAPPA, SHARE, STEP, SIGMA = 4., .7, .55, .42                 # enlarged selection strength, replaced share, step, spread
    angle = np.linspace(0, 2*np.pi, 200)

    def ellipse(ax, pts, weights=None, **style):
        weights = np.full(len(pts), 1/len(pts)) if weights is None else weights/weights.sum()
        mean = weights@pts
        cov = (weights[:, None]*(pts-mean)).T@(pts-mean)
        vals, vecs = np.linalg.eigh(cov)
        curve = mean[:, None]+2*vecs@(np.sqrt(vals)[:, None]*np.vstack([np.cos(angle), np.sin(angle)]))
        ax.plot(curve[0], curve[1], **style)
        return mean

    def means(ax, m0, m1, color):
        ax.scatter(*m0, marker="+", s=42, color="0.3", linewidths=1.1, zorder=6)
        ax.scatter(*m1, marker="+", s=42, color=color, linewidths=1.3, zorder=7)
        if np.linalg.norm(m1-m0) > .05:
            ax.annotate("", xy=m1, xytext=m0, zorder=6, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, mutation_scale=7,
                                                                       shrinkA=0, shrinkB=0))

    def descent(y):
        g = grad2(y)
        return y-STEP*g/np.sqrt(1+np.sum(g**2, axis=1, keepdims=True))

    def offspring(y, n, rng):
        return (y[rng.integers(0, len(y), n)]+y[rng.integers(0, len(y), n)])/2

    xs = np.linspace(-.1, 2.9, 300)
    before_density = kde(xs, large[:, 0], h=.08)
    after_density, record_top = {}, {}
    # (a) selection: nobody moves, the weights change
    background(top[0], "(a) selection S")
    w_small, w_large = np.exp(-KAPPA*f2(small)/(1+f2(small))), np.exp(-KAPPA*f2(large)/(1+f2(large)))
    m0 = ellipse(top[0], large, color="0.45", lw=1., ls="--", zorder=4)
    m1 = ellipse(top[0], large, w_large, color=OPCOLOR["S"], lw=1.3, zorder=5)
    top[0].scatter(small[:, 0], small[:, 1], s=5+48*w_small/w_small.max(), color=OPCOLOR["S"], alpha=.85, zorder=3, linewidths=0)
    means(top[0], m0, m1, OPCOLOR["S"])
    after_density["S"] = kde(xs, large[:, 0], w_large, h=.08)
    # (b) recombination: a share of the population is replaced by midpoints of two parents
    background(top[1], "(b) recombination R")
    kept = rng2.random(len(small)) > SHARE
    children = offspring(small, int((~kept).sum()), rng2)
    large_after = np.vstack([large[rng2.random(len(large)) > SHARE], offspring(large, int(SHARE*len(large)), rng2)])
    m0 = ellipse(top[1], large, color="0.45", lw=1., ls="--", zorder=4)
    m1 = ellipse(top[1], large_after, color=OPCOLOR["R"], lw=1.3, zorder=5)
    top[1].scatter(small[:, 0], small[:, 1], s=7, color="0.6", zorder=3, linewidths=0)
    top[1].scatter(children[:, 0], children[:, 1], s=13, marker="X", color=OPCOLOR["R"], zorder=4, linewidths=0)
    means(top[1], m0, m1, OPCOLOR["R"])
    after_density["R"] = kde(xs, large_after[:, 0], h=.08)
    # (c) descent step: everybody moves downhill
    background(top[2], "(c) descent step D")
    moved = descent(small)
    m0 = ellipse(top[2], large, color="0.45", lw=1., ls="--", zorder=4)
    m1 = ellipse(top[2], descent(large), color=OPCOLOR["D"], lw=1.3, zorder=5)
    top[2].scatter(small[:, 0], small[:, 1], s=7, color="0.6", zorder=3, linewidths=0)
    for a_, b_ in zip(small, moved):
        top[2].annotate("", xy=b_, xytext=a_, zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=OPCOLOR["D"], lw=.6, mutation_scale=4, shrinkA=0, shrinkB=0))
    means(top[2], m0, m1, OPCOLOR["D"])
    after_density["D"] = kde(xs, descent(large)[:, 0], h=.08)
    # (d) Gaussian mutation: everybody is perturbed at random
    background(top[3], "(d) Gaussian mutation H")
    kicked = small+SIGMA*rng2.normal(size=small.shape)
    m0 = ellipse(top[3], large, color="0.45", lw=1., ls="--", zorder=4)
    m1 = ellipse(top[3], large+SIGMA*rng2.normal(size=large.shape), color=OPCOLOR["H"], lw=1.3, zorder=5)
    top[3].scatter(small[:, 0], small[:, 1], s=7, color="0.6", zorder=3, linewidths=0)
    top[3].scatter(kicked[:, 0], kicked[:, 1], s=7, color=OPCOLOR["H"], zorder=4, linewidths=0)
    means(top[3], m0, m1, OPCOLOR["H"])
    after_density["H"] = kde(xs, large[:, 0], h=np.sqrt(.08**2+SIGMA**2))
    for ax in top:
        ax.set(xlim=VIEW[0], ylim=VIEW[1])
    for strip, name in zip(strips, "SRDH"):
        strip.fill_between(xs, before_density, color="0.6", alpha=.45, lw=0)
        strip.plot(xs, after_density[name], color=OPCOLOR[name], lw=1.1)
        strip.set(xlim=VIEW[0], ylim=(0, None), yticks=[], xticks=[0, 1, 2])
        strip.tick_params(axis="x", labelsize=6.5, pad=1)
        strip.text(.985, .98, "density of $x_1$", transform=strip.transAxes, ha="right", va="top", fontsize=6.5)
        for spine in ("top", "right", "left"):
            strip.spines[spine].set_visible(False)

    # (e) the decrease per unit time step approaches the effect
    taus = np.array([.4, .2, .1, .05, .025, .0125])
    record = {"taus": taus.tolist(), "V": float(V)}
    for name, op in list(ops.items())+[("sum", variant)]:
        single = op if name == "sum" else Assembly([op])
        quotient = np.array([-single.finite_step_quotient(pop, t, f)/V for t in taus])
        target = -single.generator(pop, f)/V
        limit.plot(taus, quotient, marker="o", ms=2.6, lw=1.1, color=OPCOLOR[name],
                   label="all" if name == "sum" else name)
        limit.plot([taus[-1]*.55], [target], marker="D", ms=3.6, color=OPCOLOR[name], clip_on=False)
        record[name] = {"finite_step": quotient.tolist(), "effect": float(target)}
    assert abs(record["sum"]["effect"]-sum(record[n]["effect"] for n in "SRDH")) < 1e-9
    limit.set_xscale("log")
    limit.set_xlim(taus[0]*1.25, taus[-1]*.5)
    limit.axhline(0, color="0.4", lw=.6)
    limit.plot([], [], marker="D", ms=3.6, lw=0, color="0.35", label="limit")
    limit.set_xlabel(r"time step $\tau$ (decreasing to the right)")
    limit.set_ylabel("relative decrease of the\n"+r"mean gap $\mathbb{E}_\mu[f(X)]-f_*$"+"\nper unit time step", fontsize=7, linespacing=1.15, labelpad=2)
    limit.set_title("(e) smaller and smaller steps", fontsize=8)
    limit.legend(frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(.5, 1.0), handlelength=1., columnspacing=.6,
                 handletextpad=.25, fontsize=6.4, borderaxespad=.1)
    lo, hi = limit.get_ylim()
    limit.set_ylim(lo, hi+.22*(hi-lo))

    # (f), (h) effects over many populations; (g) a two-cluster population
    def bars(ax, pops, title, ylabel):
        result = attribution.coefficients(variant, pops, f)
        columns = [result["ratios"][:, j] for j in range(4)]+[result["total"]]
        for k, (name, values) in enumerate(zip(["S", "R", "D", "H", "sum"], columns)):
            median, low, high = np.median(values), values.min(), values.max()
            ax.bar(k, median, width=.62, color=OPCOLOR[name], alpha=.9)
            ax.plot([k, k], [low, high], color="0.15", lw=.9)
            for end in (low, high):
                ax.plot([k-.12, k+.12], [end, end], color="0.15", lw=.9)
        ax.axhline(0, color="0.3", lw=.7)
        ax.axvline(3.5, color="0.75", lw=.7, ls=":")
        ax.set_xticks(range(5), ["S", "R", "D", "H", "sum"])
        if ylabel:                                             # the two bar panels show the same quantity
            ax.set_ylabel("relative effect", labelpad=1, fontsize=7.5)
        ax.set_title(title, fontsize=8)
        assert np.allclose(result["total"], result["ratios"].sum(axis=1))
        return {"median": [float(np.median(c)) for c in columns], "min": [float(np.min(c)) for c in columns],
                "max": [float(np.max(c)) for c in columns]}

    record["broad"] = bars(bars_b, broad, "(f) broad populations", True)
    background(sketch, "(g) two clusters", lim=2.2)
    cp = clustered[0].points
    sketch.scatter(cp[:, 0], cp[:, 1], s=7, color="0.45", zorder=3)
    a_, b_ = cp[np.argmax(cp[:, 0]+cp[:, 1])], cp[np.argmin(cp[:, 0]+cp[:, 1])]
    sketch.plot([a_[0], b_[0]], [a_[1], b_[1]], ls="--", color=ORANGE, lw=.8)
    sketch.scatter([(a_[0]+b_[0])/2], [(a_[1]+b_[1])/2], marker="X", s=40, color=ORANGE, zorder=4)
    record["clustered"] = bars(bars_c, clustered, "(h) clustered populations", False)
    AUDIT["operators"] = {"problem": "quartic_d4", "populations_per_family": 128, "N": 32, "seed": 0,
                              "schematic_top_row": {"dimension": 2, "individuals": 60, "selection_strength": KAPPA,
                                                    "replaced_share": SHARE, "descent_step": STEP, "mutation_sd": SIGMA},
                              **record}
    save(fig, "explanatory_operators.pdf")


# --------------------------------------------------------------------------- conditions
def kde(x, points, weights=None, h=.15):
    """Density of a weighted sample of one coordinate on the grid x (Gaussian kernel of width h)."""
    points = np.asarray(points, dtype=float)
    weights = np.full(points.size, 1/points.size) if weights is None else np.asarray(weights)/np.sum(weights)
    out = np.zeros_like(x)
    for start in range(0, points.size, 4000):
        block = slice(start, start+4000)
        out += np.sum(weights[block, None]*np.exp(-.5*((x[None, :]-points[block, None])/h)**2), axis=0)
    return out/(h*np.sqrt(2*np.pi))


def gauss(x, m, s):
    return np.exp(-.5*((x-m)/s)**2)/(s*np.sqrt(2*np.pi))


def mixture(x, weights, means, sds):
    return sum(w*gauss(x, m, s) for w, m, s in zip(weights, means, sds))


def w2(x, p, q):
    """2-Wasserstein distance of two densities on the grid x (quantile coupling)."""
    dx = x[1]-x[0]
    P, Q = np.cumsum(p)*dx, np.cumsum(q)*dx
    u = np.linspace(.0005, .9995, 4000)
    return float(np.sqrt(np.mean((np.interp(u, P/P[-1], x)-np.interp(u, Q/Q[-1], x))**2)))


def conditions():
    """For each of (A1)-(A4): a population in two dimensions before and after one step of an operator that
    satisfies the condition (upper half) and of one that violates it (lower half), each with the density of
    the first coordinate below it."""
    fig = plt.figure(figsize=(WIDTH, 5.05))
    X0, DX, W = .062, .2365, .2                       # columns
    ROWS = ((.642, .53), (.176, .064))                # bottoms of swarm and marginal axes: satisfied, violated
    HS, HM = .275, .092
    u = np.linspace(-3.2, 3.2, 160)
    U1, U2 = np.meshgrid(u, u)
    level = np.log1p((U1**2-1)**2+(U2**2-1)**2)
    x = np.linspace(-4, 4, 2001)
    record = {}
    f2 = lambda y: np.sum((y**2-1)**2, axis=-1)
    grad = lambda y: 4*y*(y**2-1)

    def cell(col, row, lim=2.6):
        swarm = fig.add_axes([X0+DX*col, ROWS[row][0], W, HS])
        marg = fig.add_axes([X0+DX*col, ROWS[row][1], W, HM])
        swarm.contour(U1, U2, level, levels=8, colors="0.82", linewidths=.5)
        swarm.scatter(*np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]).T, marker="*", s=22, color="0.55", zorder=2)
        swarm.set(xlim=(-lim, lim), ylim=(-lim, lim), xticks=[], yticks=[])
        color = GREEN if row == 0 else RED
        for ax in (swarm, marg):
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_color(color); spine.set_linewidth(.9)
        marg.set(xlim=(-lim, lim), yticks=[])
        marg.tick_params(axis="x", labelsize=6.5, pad=1)
        return swarm, marg

    def before_after(ax, before, after, lim=2.6):
        ax.scatter(before[:, 0], before[:, 1], s=5, color="0.55", zorder=3, linewidths=0)
        inside = np.all(np.abs(after) < lim, axis=1)
        ax.scatter(after[inside, 0], after[inside, 1], s=5, color=BLUE, zorder=4, linewidths=0)

    for row, (word, color) in enumerate((("condition satisfied", GREEN), ("condition violated", RED))):
        fig.text(.012, ROWS[row][1]+(ROWS[row][0]+HS-ROWS[row][1])/2, word, rotation=90, ha="center", va="center",
                 fontsize=8.5, fontweight="bold", color=color)
    titles = ("(A1) definite effect\nper unit time step", "(A2) the population\nstays bounded",
              "(A3) a small step\nchanges little", "(A4) similar populations,\nsimilar effect")
    for col, title in enumerate(titles):
        fig.text(X0+DX*col+W/2, .995, title, ha="center", va="top", fontsize=8.5, linespacing=1.05)

    # ---------------- (A1) mutation scaled by sqrt(tau) against a perturbation of fixed size
    rng = np.random.default_rng(0)
    weights, means, sds, sigma, n = (.6, .4), (-1., .8), (.35, .5), .8, 140
    which = rng.random(n) < weights[0]
    pop = np.where(which[:, None], rng.normal(means[0], sds[0], (n, 2)), rng.normal(means[1], sds[1], (n, 2)))
    rho = mixture(x, weights, means, sds)
    taus = (.1, .05, .025)
    for row, scaled in enumerate((True, False)):
        swarm, marg = cell(0, row)
        before_after(swarm, pop, pop+sigma*(np.sqrt(taus[1]) if scaled else 1.)*rng.normal(size=pop.shape))
        for tau, color in zip(taus, (BLUE, ORANGE, GREEN)):
            extra = sigma**2*(tau if scaled else 1.)
            after = mixture(x, weights, means, [np.sqrt(sd**2+extra) for sd in sds])
            marg.plot(x, (after-rho)/tau, color=color, lw=1., label=rf"$\tau={tau}$")
        marg.axhline(0, color="0.5", lw=.5)
        marg.set_xlabel((r"mutation $y+\sigma\sqrt{\tau}\,Z$" if scaled else r"perturbation $y+\sigma Z$")
                        + "\nchange of density per unit time step", fontsize=6.4, labelpad=1, linespacing=1.05)

    # ---------------- (A2) normalized descent step against a plain gradient step
    rng = np.random.default_rng(0)
    pop = rng.normal(0., 1.3, (140, 2))
    big = rng.normal(0., 1.3, (40000, 2))
    tau = .1
    norm = lambda y: -tau*grad(y)/np.sqrt(1+np.sum(grad(y)**2, axis=1, keepdims=True))
    plain = lambda y: -tau*grad(y)
    bins = np.linspace(-12, 12, 121)
    for row, (ok, move) in enumerate(((True, norm), (False, plain))):
        swarm, marg = cell(1, row, lim=3.2)
        after = pop+move(pop)
        swarm.scatter(pop[:, 0], pop[:, 1], s=5, color="0.55", zorder=3, linewidths=0)
        for a_, b_ in zip(pop, after):
            if np.any(np.abs(a_) >= 3.2):
                continue
            d_ = b_-a_
            with np.errstate(divide="ignore", invalid="ignore"):
                reach = np.where(d_ != 0, (3.2*np.sign(d_)-a_)/d_, np.inf)
            t_ = min(1., float(np.min(reach)))        # stop the arrow at the frame, keeping its direction
            swarm.annotate("", xy=a_+t_*d_, xytext=a_, zorder=4,
                           arrowprops=dict(arrowstyle="-|>", color=BLUE if t_ == 1. else RED, lw=.55, mutation_scale=4,
                                           shrinkA=0, shrinkB=0))
        after_big = big+move(big)
        ratio = float(np.mean(1+np.linalg.norm(after_big, axis=1)**3)/np.mean(1+np.linalg.norm(big, axis=1)**3))
        left = float(np.mean(np.any(np.abs(after_big) > 3.2, axis=1)))
        record[f"A2_{'ok' if ok else 'violated'}"] = {"third_moment_ratio": ratio, "share_outside_frame_after": left}
        marg.hist(big[:, 0], bins=bins, density=True, color="0.6", alpha=.45, lw=0)
        marg.hist(np.clip(after_big[:, 0], -12, 12), bins=bins, density=True, histtype="step", color=BLUE, lw=.9)
        marg.set(xlim=(-12, 12), yscale="log", ylim=(2e-4, 6), yticks=[])
        marg.minorticks_off()
        marg.set_xlabel(("normalized step" if ok else "plain gradient step")
                        + f"\nthird moment $\\times${ratio:.2g} (log. density)", fontsize=6.4, labelpad=1, linespacing=1.05)

    # ---------------- (A3) a share tau replaced by offspring against replacement of the whole population
    rng = np.random.default_rng(0)
    sd, n = .25, 140
    signs = np.where(rng.random(n) < .5, -1., 1.)
    pop = signs[:, None]+sd*rng.normal(size=(n, 2))
    before = mixture(x, (.5, .5), (-1., 1.), (sd, sd))
    offspring = mixture(x, (.25, .5, .25), (-1., 0., 1.), (sd/np.sqrt(2),)*3)
    for row, ok in enumerate((True, False)):
        swarm, marg = cell(2, row, lim=2.2)
        replaced = rng.choice(n, size=int(round(tau*n)) if ok else n, replace=False)
        mids = (pop[rng.integers(0, n, replaced.size)]+pop[rng.integers(0, n, replaced.size)])/2
        keep = np.setdiff1d(np.arange(n), replaced)
        swarm.scatter(pop[:, 0], pop[:, 1], s=5, color="0.55", zorder=3, linewidths=0)
        swarm.scatter(mids[:, 0], mids[:, 1], s=9, color=ORANGE, marker="X", zorder=4, linewidths=0)
        after = (1-tau)*before+tau*offspring if ok else offspring
        distance = w2(x, before, after)
        record[f"A3_{'ok' if ok else 'violated'}"] = {"W2_first_coordinate": distance, "offspring_shown": int(replaced.size)}
        marg.fill_between(x, before, color="0.6", alpha=.45, lw=0)
        marg.plot(x, after, color=BLUE, lw=.9)
        marg.set(xlim=(-2.2, 2.2), ylim=(0, 2.6))
        marg.set_xlabel((r"share $\tau=0.1$ replaced" if ok else "all replaced at once")+f"\ndistance {distance:.2f}",
                        fontsize=6.4, labelpad=1, linespacing=1.05)

    # ---------------- (A4) pull to a weighted average against pull to the best individual
    rng = np.random.default_rng(1)
    base = np.concatenate([np.array([-1., -1.])+.22*rng.normal(size=(11, 2)), np.array([1., 1.])+.22*rng.normal(size=(12, 2))])
    base = base[f2(base) > .02]                      # the best of the others is at a visible distance
    start, end = np.array([-1.32, -1.32]), np.array([-1., -1.])
    path = np.linspace(0., 1., 400)

    def targets(z):
        pop = np.vstack([base, z])
        values = f2(pop)
        wts = np.exp(-5*(values-values.min()))
        return wts@pop/wts.sum(), pop[np.argmin(values)]

    soft = np.array([targets(start+s_*(end-start))[0] for s_ in path])
    hard = np.array([targets(start+s_*(end-start))[1] for s_ in path])
    record["A4"] = {"largest_jump_weighted_average": float(np.max(np.linalg.norm(np.diff(soft, axis=0), axis=1))),
                    "largest_jump_best_individual": float(np.max(np.linalg.norm(np.diff(hard, axis=0), axis=1)))}
    for row, (ok, target) in enumerate(((True, soft), (False, hard))):
        swarm, marg = cell(3, row, lim=2.2)
        swarm.scatter(base[:, 0], base[:, 1], s=6, color="0.55", zorder=3, linewidths=0)
        swarm.annotate("", xy=end, xytext=start, zorder=5,
                       arrowprops=dict(arrowstyle="-|>", color="0.2", lw=.8, mutation_scale=5, shrinkA=0, shrinkB=0))
        swarm.scatter(*start, s=9, color="0.2", zorder=5, linewidths=0)
        swarm.scatter(*target[0], marker="D", s=26, facecolors="none", edgecolors=BLUE, linewidths=1., zorder=6)
        swarm.scatter(*target[-1], marker="D", s=26, color=BLUE, zorder=6, linewidths=0)
        marg.plot(path, target[:, 0], color=BLUE, lw=1.1)
        marg.set(xlim=(0, 1), ylim=(-1.45, 1.45), yticks=[-1, 1], xticks=[0, 1])
        marg.tick_params(axis="y", labelsize=6.5, pad=1)
        marg.set_xticklabels(["start", "end"])
        marg.set_xlabel(("pull to a weighted average" if ok else "pull to the best individual")+"\nfirst coordinate of the target",
                        fontsize=6.4, labelpad=1, linespacing=1.05)
    AUDIT["conditions"] = record
    save(fig, "explanatory_conditions.pdf")


def distance():
    """The two parts of the distance between populations: motion (W_2) and reweighting (bounded-Lipschitz part)."""
    from scipy.optimize import linear_sum_assignment
    rng = np.random.default_rng(3)
    f2 = lambda y: np.sum((y**2-1)**2, axis=-1)
    grad = lambda y: 4*y*(y**2-1)
    pop = rng.normal(.2, 1.1, (36, 2))
    u = np.linspace(-3., 3., 160)
    U1, U2 = np.meshgrid(u, u)
    level = np.log1p((U1**2-1)**2+(U2**2-1)**2)
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.95))
    fig.subplots_adjust(left=.02, right=.98, top=.86, bottom=.14, wspace=.08)
    for ax in axes:
        ax.contour(U1, U2, level, levels=8, colors="0.84", linewidths=.5)
        ax.set(xlim=(-3, 3), ylim=(-3, 3), xticks=[], yticks=[], aspect="equal")
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color("0.6"); spine.set_linewidth(.6)
    # (a) motion: optimal matching of two populations with equal weights
    moved = pop-.45*grad(pop)/np.sqrt(1+np.sum(grad(pop)**2, axis=1, keepdims=True))+.12*rng.normal(size=pop.shape)
    cost = np.sum((pop[:, None, :]-moved[None, :, :])**2, axis=2)
    rows, cols = linear_sum_assignment(cost)
    w2_value = float(np.sqrt(cost[rows, cols].mean()))
    for i, j in zip(rows, cols):
        axes[0].plot([pop[i, 0], moved[j, 0]], [pop[i, 1], moved[j, 1]], color=BLUE, lw=.6, zorder=2)
    axes[0].scatter(pop[:, 0], pop[:, 1], s=7, color="0.5", zorder=3, linewidths=0)
    axes[0].scatter(moved[:, 0], moved[:, 1], s=7, color=BLUE, zorder=4, linewidths=0)
    axes[0].set_title("(a) motion", fontsize=8, pad=3)
    axes[0].set_xlabel(f"matched pairs: $W_2={w2_value:.2f}$", fontsize=7, labelpad=2)
    # (b) reweighting: nobody moves, weights and total mass change
    weights = np.exp(-1.5*f2(pop)/(1+f2(pop)))            # selection without renormalization
    mass = float(weights.mean())
    axes[1].scatter(pop[:, 0], pop[:, 1], s=26, facecolors="none", edgecolors="0.5", linewidths=.5, zorder=3)
    axes[1].scatter(pop[:, 0], pop[:, 1], s=26*weights, color=GREEN, zorder=4, linewidths=0)
    axes[1].set_title("(b) reweighting", fontsize=8, pad=3)
    axes[1].set_xlabel(f"total mass: 1 $\\to$ {mass:.2f}", fontsize=7, labelpad=2)
    AUDIT["distance"] = {"N": 36, "W2_motion": w2_value, "mass_after_reweighting": mass, "seed": 3}
    save(fig, "explanatory_population_distance.pdf")


def evolution():
    """A population evolving under the four operators; the evolution equation for three test functions: the
    change of each population average is the accumulated effect of the operators."""
    from evoscope.operators import Observable
    from evoscope.population import Population
    f = problems.get("quartic_d4")
    ops = {"S": Selection(objective_rate(f.value), name="S"), "R": Recombination(name="R"),
           "D": Drift(normalized_fd_drift(f.value), name="D"), "H": GaussianNoise(.2, name="H")}
    variant = Assembly(list(ops.values()))

    def share_value(x):
        return np.exp(-.5*f.value(x))

    def share_gradient(x):
        return -.5*share_value(x)[..., None]*f.gradient(x)

    def share_laplacian(x):
        return share_value(x)*(.25*np.sum(f.gradient(x)**2, axis=-1)-.5*f.laplacian(x))

    # (key, panel title, label of the vertical axis, test function). Each test function is an error measure
    # that is zero when the whole population sits at an optimum (or, for the second, at the chosen optimum).
    # Tried before: the mean of one coordinate (over 256 runs it is dominated by the optimum that a run
    # reaches) and the squared distance to the origin (it does not measure convergence).
    y_star = np.ones(4)
    tests = (("gap", "mean objective gap", r"$\mathbb{E}_{\mu_t}[f(X)]-f_*$",
              Observable(lambda x: f.value(x)-f.f_star, gradient=f.gradient, laplacian=f.laplacian)),
             ("distance", "mean squared distance to one optimum", r"$\mathbb{E}_{\mu_t}[\|X-y_*\|^2]$",
              Observable(lambda x: np.sum((x-y_star)**2, axis=-1), gradient=lambda x: 2*(x-y_star),
                         laplacian=lambda x: np.full(x.shape[:-1], 2.*x.shape[-1]))),
             ("far", "share not yet near an optimum", r"$1-\mathbb{E}_{\mu_t}[e^{-f(X)/2}]$",
              Observable(lambda x: 1.-share_value(x), gradient=lambda x: -share_gradient(x), laplacian=lambda x: -share_laplacian(x))))
    tau, N, RUNS = ARGS.tau, 256, 256
    suffix = "" if tau == .025 else f"_tau{tau:g}"             # a trial with another time step is saved under its own name
    K = round(6/tau)                                           # time horizon 6
    shown = tuple(round(time/tau) for time in (0, 1, 4))      # generations of the three populations that are drawn
    if ARGS.quick:                                             # short trial run of the script, not the figure of the article
        N, RUNS, suffix = 64, 6, suffix+"_quick"
    # The runs take about a quarter of an hour. Their results are kept, so that a change of the lettering
    # does not repeat them; the file is used again only for the same test functions and settings.
    cache = ROOT/"results/explanatory-cache"/f"evolution_tau{tau:g}_N{N}_runs{RUNS}.npz"
    keys = [key for key, _, _, _ in tests]
    if cache.exists() and list(np.load(cache)["keys"]) == keys:
        data = np.load(cache)
        values, actions = data["values"], data["actions"]
        snapshots = {k: data[f"snapshot_{k}"] for k in shown}
        marginals = {k: [data[f"marginal_{k}"]] for k in shown}
    else:
        rng = np.random.default_rng(0)
        values = np.zeros((len(tests), RUNS, K+1)); actions = np.zeros((len(tests), RUNS, K, 4)); snapshots = {}
        marginals = {k: [] for k in shown}
        for r in range(RUNS):
            pop = broad_gaussian(N, 4, rng)
            for k in range(K+1):
                for i, (_, _, _, obs) in enumerate(tests):
                    values[i, r, k] = pop.expect(obs.value(pop.points))
                if k in marginals:
                    marginals[k].append(pop.points[:, 0].copy())
                if r == 0 and k in shown:
                    snapshots[k] = pop.points[:128].copy()
                if k == K:
                    break
                for i, (_, _, _, obs) in enumerate(tests):
                    acts = variant.generator_actions(pop, obs)
                    actions[i, r, k] = [acts[n] for n in "SRDH"]
                pop = Population.uniform(variant.sample(pop, tau, N, rng))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, keys=np.array(keys), values=values, actions=actions,
                            **{f"snapshot_{k}": snapshots[k] for k in shown},
                            **{f"marginal_{k}": np.concatenate(marginals[k]) for k in shown})
    t = tau*np.arange(K+1)
    V = values[0]
    relative = -(actions[0].sum(axis=2)/V[:, :K])              # summed effect on the mean gap, relative to the gap

    fig = plt.figure(figsize=(WIDTH, 4.25))
    xs = np.linspace(-2.6, 2.6, 300)
    u = np.linspace(-2.6, 2.6, 160)
    U1, U2 = np.meshgrid(u, u)
    level = np.log1p((U1**2-1)**2+(U2**2-1)**2)
    for i, k in enumerate(shown):
        ax = fig.add_axes([.012+.2*i, .615, .19, .345])
        strip = fig.add_axes([.012+.2*i, .5, .19, .095])
        density = kde(xs, np.concatenate(marginals[k]), h=.08)
        strip.fill_between(xs, density, color=BLUE, alpha=.35, lw=0)
        strip.plot(xs, density, color=BLUE, lw=.9)
        strip.set(xlim=(-2.6, 2.6), ylim=(0, None), yticks=[], xticks=[-2, 0, 2])
        strip.tick_params(axis="x", labelsize=6.5, pad=1)
        strip.text(.02, .98, "density of $x_1$", transform=strip.transAxes, ha="left", va="top", fontsize=6.5)
        for spine in ("top", "right", "left"):
            strip.spines[spine].set_visible(False)
        ax.contour(U1, U2, level, levels=8, colors="0.82", linewidths=.5)
        ax.scatter(*np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]).T, marker="*", s=22, color="0.55", zorder=2)
        ax.scatter(snapshots[k][:, 0], snapshots[k][:, 1], s=5, color=BLUE, zorder=3, linewidths=0)
        if i == 0:
            ax.text(.02, .02, "individuals, coordinates $(x_1, x_2)$", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.2, color="0.25")
        ax.set(xlim=(-2.6, 2.6), ylim=(-2.6, 2.6), xticks=[], yticks=[], aspect="equal")
        ax.set_title(f"({'abc'[i]}) time $t={tau*k:g}$", fontsize=8, pad=3)
        for spine in ax.spines.values():
            spine.set_color("0.6"); spine.set_linewidth(.6); spine.set_visible(True)
    rate = fig.add_axes([.74, .56, .245, .345])
    q10, q50, q90 = np.percentile(relative, [10, 50, 90], axis=0)
    rate.fill_between(t[:K], q10, q90, color=DARK, alpha=.18, lw=0)
    rate.plot(t[:K], q50, color=DARK, lw=1.2)
    rate.axhline(0, color=RED, lw=.8, ls="--")
    rate.set_xlabel("time $t$", labelpad=1)
    rate.set_ylabel(r"$-\sum_j G_j[\mu_t](f)\,/\,(\mathbb{E}_{\mu_t}[f(X)]-f_*)$", labelpad=2)
    rate.set_title("(d) summed effect on the mean gap,\nrelative to the mean gap", fontsize=8, pad=3, linespacing=1.05)
    record = {}
    for i, (key, title, ylabel, _) in enumerate(tests):
        ax = fig.add_axes([.08+.325*i, .085, .24, .30])
        observed = values[i].mean(axis=0)                      # the error itself, which is zero at convergence
        start = float(observed[0])
        terms = np.concatenate([np.zeros((1, 4)), np.cumsum(tau*actions[i].mean(axis=0), axis=0)])
        for j, op in enumerate("SRDH"):
            ax.plot(t, start+terms[:, j], color=OPCOLOR[op], lw=1.1, label=op)
        ax.plot(t, start+terms.sum(axis=1), color=DARK, lw=1.6, label="all four operators")
        ax.plot(t[::round(.3/tau)], observed[::round(.3/tau)], ls="", marker="o", ms=2.6, mfc="white", mec=DARK, mew=.7, label="observed")
        ax.axhline(0, color=RED, lw=.8, ls="--")
        ax.set_ylim(bottom=min(0., ax.get_ylim()[0]))
        ax.set_xlabel("time $t$", labelpad=1)
        ax.set_ylabel(ylabel, labelpad=1)
        ax.set_title(f"({'efg'[i]}) {title}", fontsize=8, pad=3)
        record[key] = {"title": title, "vertical_axis": ylabel, "start": start, "observed_end": float(observed[-1]),
                       "start_plus_sum_of_effects_end": float(start+terms.sum(axis=1)[-1]),
                       "accumulated_effects_end": {op: float(terms[-1, j]) for j, op in enumerate("SRDH")}}
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=6, loc="lower center", bbox_to_anchor=(.5, .405),
               handlelength=1.3, columnspacing=1.2)
    AUDIT["evolution"+suffix] = {"problem": "quartic_d4", "tau": tau, "generations": K, "N": N, "runs": RUNS, "seed": 0,
                          "test_functions": record, "median_relative_start": float(q50[0]), "median_relative_end": float(q50[-1]),
                          "mean_gap_start": float(V[:, 0].mean()), "mean_gap_end": float(V[:, -1].mean())}
    save(fig, f"explanatory_evolution_equation{suffix}.pdf")


def targets():
    """Four populations that separate the three convergence targets, and an internal state with its candidates."""
    rng = np.random.default_rng(5)
    f2 = lambda y: np.sum((y**2-1)**2, axis=-1)
    star = np.array([1., 1.])
    n = 40
    cases = (("(a) good solution\nfound", np.vstack([rng.normal(0., 1.2, (n-1, 2)), star+[.03, -.02]])),
             ("(b) good\non average", np.vstack([c+.08*rng.normal(size=(n//4, 2)) for c in ([1, 1], [1, -1], [-1, 1], [-1, -1])])),
             ("(c) agreement\nat an optimum", star+.08*rng.normal(size=(n, 2))),
             ("(d) agreement\nelsewhere", np.array([0., 0.])+.08*rng.normal(size=(n, 2))))
    fig = plt.figure(figsize=(WIDTH, 2.75))
    xs = np.linspace(-2.4, 2.4, 300)
    u = np.linspace(-2.4, 2.4, 160)
    U1, U2 = np.meshgrid(u, u)
    level = np.log1p((U1**2-1)**2+(U2**2-1)**2)
    record = {}

    def panel(i, title):
        ax = fig.add_axes([.008+.198*i, .39, .185, .48])
        strip = fig.add_axes([.008+.198*i, .25, .185, .11])
        strip.set(xlim=(-2.4, 2.4), yticks=[], xticks=[])
        for spine in ("top", "right", "left"):
            strip.spines[spine].set_visible(False)
        ax.contour(U1, U2, level, levels=8, colors="0.82", linewidths=.5)
        ax.scatter(*np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]).T, marker="*", s=22, color="0.55", zorder=2)
        ax.set(xlim=(-2.4, 2.4), ylim=(-2.4, 2.4), xticks=[], yticks=[], aspect="equal")
        ax.set_title(title, fontsize=8, pad=3, linespacing=1.05)
        for spine in ax.spines.values():
            spine.set_color("0.6"); spine.set_linewidth(.6); spine.set_visible(True)
        return ax, strip

    for i, (title, pts) in enumerate(cases):
        ax, strip = panel(i, title)
        ax.scatter(pts[:, 0], pts[:, 1], s=6, color=BLUE, zorder=3, linewidths=0)
        density = kde(xs, pts[:, 0], h=.12)
        strip.fill_between(xs, density, color=BLUE, alpha=.35, lw=0)
        strip.plot(xs, density, color=BLUE, lw=.9)
        strip.set_ylim(0, None)
        values = f2(pts)
        best, mean = float(values.min()), float(values.mean())
        dist = float(np.mean(np.sum((pts-star)**2, axis=1)))
        if i == 0:
            j = int(np.argmin(values))
            ax.scatter(pts[j, 0], pts[j, 1], s=34, facecolors="none", edgecolors=ORANGE, linewidths=1.1, zorder=4)
        record[title[:3]] = {"best_gap": best, "mean_gap": mean, "mean_squared_distance_to_optimum": dist}
        strip.set_xlabel(f"best gap {best:.2f}\nmean gap {mean:.2f}\nmean sq. distance {dist:.2f}", fontsize=7, labelpad=2, linespacing=1.1)
    # (e) internal state and sampled candidates: a Gaussian search distribution
    ax, strip = panel(4, "(e) state and\ncandidates")
    mean_, cov = np.array([-.5, .3]), np.array([[.55, .3], [.3, .4]])
    vals, vecs = np.linalg.eigh(cov)
    angle = np.linspace(0, 2*np.pi, 200)
    for scale in (1., 2.):
        ell = mean_[:, None]+scale*vecs@(np.sqrt(vals)[:, None]*np.vstack([np.cos(angle), np.sin(angle)]))
        ax.plot(ell[0], ell[1], color=DARK, lw=.9)
    ax.scatter(*mean_, marker="+", s=40, color=DARK, linewidths=1.1, zorder=4)
    cand = rng.multivariate_normal(mean_, cov, 24)
    ax.scatter(cand[:, 0], cand[:, 1], s=6, color=BLUE, zorder=3, linewidths=0)
    strip.fill_between(xs, gauss(xs, mean_[0], np.sqrt(cov[0, 0])), color=DARK, alpha=.2, lw=0)
    strip.plot(xs, gauss(xs, mean_[0], np.sqrt(cov[0, 0])), color=DARK, lw=.9)
    strip.plot(cand[:, 0], np.zeros(len(cand)), ls="", marker="|", ms=4, color=BLUE)
    strip.set_ylim(0, None)
    strip.set_xlabel("state: mean and\ncovariance (ellipses)\ncandidates: points", fontsize=7, labelpad=2, linespacing=1.1)
    AUDIT["targets"] = record
    save(fig, "explanatory_convergence_targets.pdf")


if __name__ == "__main__":
    functions = {"operators": operators, "conditions": conditions, "distance": distance, "evolution": evolution,
                 "targets": targets}
    for name in (ARGS.figures or FIGS):
        functions[name]()
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stored = json.loads(AUDIT_PATH.read_text()) if AUDIT_PATH.exists() else {}
    stored.update(AUDIT)
    stored["evoscope_version"] = evoscope.__version__
    AUDIT_PATH.write_text(json.dumps(stored, indent=1)+"\n")
    print("Wrote", AUDIT_PATH)
