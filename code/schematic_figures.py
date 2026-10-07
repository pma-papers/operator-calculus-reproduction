"""Compact schematic: basin geometry and the three canonical operators.

One row, four panels, one shared one-dimensional landscape.  Panel (a)
annotates Assumption (basin regularity); panels (b)-(d) show one balanced
tau-step of mutation, selection and recombination acting on the same
population density.  Illustration only; no experiment data.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/"figures/landscape_operators.pdf"
X = np.linspace(-5, 5, 1201)
Y_STAR = 0.
R0, F_INF, EPS, ETA, NU = 1.6, .58, .18, .42, .5     # Assumption parameters drawn in (a)


def f(y):
    """Smooth landscape: global minimizer at 0 with f_* = 0, two decoy wells."""
    return (1-np.exp(-1.1*y**2))*.85+.15*(1-np.exp(-.6*y**2)) \
        - .22*np.exp(-2.5*(y-3.)**2) - .16*np.exp(-2.5*(y+2.8)**2)


def density(y):
    rho = .5*norm.pdf(y, -1.4, .8)+.5*norm.pdf(y, 1.7, .65)
    return rho/np.trapezoid(rho, y)


def mutation(rho, drift=-.55, sigma=.45):
    dx = X[1]-X[0]
    support = np.arange(-5*sigma, 5*sigma+dx, dx)
    kernel = norm.pdf(support, 0, sigma)
    kernel /= kernel.sum()
    out = np.convolve(rho, kernel, mode="same")
    # Objective-directed drift toward y_*: transport along -sign(y)*|drift|.
    shifted = np.interp(X, X+drift*np.sign(X)*np.minimum(1, np.abs(X)), out, left=0, right=0)
    return shifted/np.trapezoid(shifted, X)


def selection(rho, pressure=2.6):
    out = rho*np.exp(-pressure*f(X))
    return out/np.trapezoid(out, X)


def recombination(rho, tau=.8, rng=np.random.default_rng(0), n=400000):
    cdf = np.cumsum(rho); cdf /= cdf[-1]
    a, b = np.interp(rng.random(n), cdf, X), np.interp(rng.random(n), cdf, X)
    hist, edges = np.histogram((a+b)/2, bins=len(X), range=(X[0], X[-1]), density=True)
    jump = np.interp(X, (edges[:-1]+edges[1:])/2, hist)
    dx = X[1]-X[0]
    k = norm.pdf(np.arange(-.5, .5+dx, dx), 0, .1); k /= k.sum()
    jump = np.convolve(jump, k, mode="same"); jump /= np.trapezoid(jump, X)
    out = (1-tau)*rho+tau*jump
    return out/np.trapezoid(out, X)


def style():
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.6, "axes.labelsize": 7,
                         "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})


def landscape_panel(ax):
    fy = f(X)
    ax.axvspan(Y_STAR-R0, Y_STAR+R0, color="#246f96", alpha=.09, lw=0)
    eps_set = X[fy <= EPS]
    ax.axvspan(eps_set.min(), eps_set.max(), color="#347c63", alpha=.25, lw=0)
    ax.plot(X, fy, color="k", lw=1.1)
    inside = np.abs(X-Y_STAR) <= R0
    ax.plot(X[inside], (ETA*np.abs(X[inside]-Y_STAR))**(1/NU), color="#c16c23", ls="--", lw=.9)
    ax.axhline(F_INF, color="#8a2b2b", ls=":", lw=.8)
    ax.axhline(EPS, color="#347c63", ls=":", lw=.8)
    ax.plot([Y_STAR], [0], "o", color="k", ms=3)
    ax.annotate("", xy=(Y_STAR+R0, .04), xytext=(Y_STAR, .04), arrowprops=dict(arrowstyle="->", lw=.7, color="#246f96"))
    ax.text(Y_STAR+R0/2, .08, "$R_0$", color="#246f96", ha="center", fontsize=6.5)
    ax.text(4.9, F_INF+.03, r"$f_\infty$", color="#8a2b2b", ha="right", fontsize=6.5)
    ax.text(4.9, EPS+.03, r"$f_*+\varepsilon$", color="#347c63", ha="right", fontsize=6.5)
    ax.text(-4.9, .36, r"$(\eta\|y-y_*\|)^{1/\nu}$", color="#c16c23", ha="left", fontsize=6.2)
    ax.text(Y_STAR+.08, -.09, "$y_*$", fontsize=6.5, ha="left", va="top")
    ax.text(0, .92, r"$\mathcal{S}_*^{\varepsilon}$", color="#347c63", ha="center", fontsize=6.5)
    ax.text(-2.8, .86, "local\nmin", ha="center", fontsize=5.6, color="0.35")
    ax.text(3., .85, "local\nmin", ha="center", fontsize=5.6, color="0.35")
    ax.set(title="(a) Landscape and basin geometry", xlim=(-5, 5), ylim=(-.12, 1.02), yticks=[0, .5, 1])
    ax.set_ylabel("$f(y)-f_*$")
    ax.set_xlabel("$y$", labelpad=1)


def operator_panels(axes):
    fy, rho = f(X), density(X)
    panels = (("(b) Mutation (drift, diffusion)", mutation(rho), "#246f96"),
              ("(c) Selection (reweighting)", selection(rho), "#8a2b2b"),
              ("(d) Recombination (midpoint)", recombination(rho), "#6a4c93"))
    for ax, (title, after, color) in zip(axes, panels):
        ax.fill_between(X, 0, rho, color="0.55", alpha=.35, lw=0, label=r"before: $\bar\mu$")
        ax.plot(X, after, color=color, lw=1.2, label=r"after one $\tau$-step")
        ax.plot(X, fy*.62, color="k", lw=.5, alpha=.35)
        ax.axvline(Y_STAR, color="k", lw=.5, ls=":", alpha=.6)
        ax.set(title=title, xlim=(-5, 5), ylim=(0, .68), yticks=[])
        ax.set_xlabel("$y$", labelpad=1)
        ax.legend(frameon=False, loc="upper right", fontsize=5.8, handlelength=1.4)
    axes[0].set_ylabel("density")


def save(fig, path):
    fig.savefig(path, metadata={"Title": path.stem, "CreationDate": None, "ModDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(fig)
    print("Wrote", path)


def main():
    style()
    fig, ax = plt.subplots(figsize=(2.95, 1.5), layout="constrained")
    landscape_panel(ax)
    save(fig, ROOT/"figures/landscape.pdf")
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 1.05), layout="constrained")
    operator_panels(axes)
    save(fig, ROOT/"figures/operators.pdf")
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 1.78), layout="constrained")
    landscape_panel(axes[0]); operator_panels(axes[1:])
    save(fig, OUTPUT)


if __name__ == "__main__":
    main()
