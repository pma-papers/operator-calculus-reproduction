"""The lettering of the figures in the current version of the paper.

The current version renamed several terms after the figures of the study were drawn: "macro-step"
became "time step", "non-discovery" "failure frequency", "objective calls" "function evaluations",
"certified" "proved", "component" "operator" (for S, R, D, H) or "effect", "full assembly" "full
variant", "Gaussian noise" "Gaussian mutation", and the functionals V, E and W of the axis labels are
written as expected values.  `relabel(fig)` applies these replacements to every text object of a
figure just before it is saved.  It changes text only; the plotted data are untouched, and the
scripts that use it check the plotted numbers against the audits of the study.
"""
from matplotlib.text import Text

REPLACEMENTS = (
    # article figures (article_figures.py)
    ("(b) Mutation (drift, diffusion)", "(b) Mutation (directed motion, diffusion)"),
    ("Macro-step", "Time step"),
    ("Non-discovery", "Failure frequency"),
    ("Objective calls", "Function evaluations"),
    ("Certified", "Proved"),
    ("Component coefficients", "Relative effects"),
    ("Component (bars", "Operator (bars"),
    (r"$\max\,|D_{\tau,o}-G[\mu](f)|/(1+V)$", "worst remainder"),
    (r"median $-G_j[\mu](f)/V(\mu)$", "relative effect"),
    ("$V_*$", r"$\mathbb{E}_\mu\|X-y_*\|^2$"),
    (r"$2W_\alpha$", r"$2\,\mathbb{E}_\mu\|X-v_\alpha\|^2$"),
    (r"Mean gap $\bar{\mathcal{E}}_k/\bar{\mathcal{E}}_0$", r"Mean gap $\bar\Psi_k/\bar\Psi_0$"),
    # supplement figures (guide_figures.py, supplement_figures.py)
    ("Archive non-discovery", "Archive failure frequency"),
    ("observed non-discovery", "observed failure frequency"),
    ("full assembly", "full variant"),
    ("Gaussian noise: mean cost", "Gaussian mutation: mean cost"),
    ("Exact noise mean cost", "Exact mean cost of H"),
    ("positive = worse mean gap", "(positive: worse mean gap)"),
    ("positive = better best-so-far value", "(positive: better)"),
    ("(b) Components, PL 4D", "(b) Operators, PL 4D"),
)


def reword(text):
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text


def relabel(fig):
    """Apply REPLACEMENTS to every text object of `fig`; return the list of [old, new] pairs changed."""
    changed = []
    for text in fig.findobj(Text):
        old = text.get_text()
        new = reword(old)
        if new != old:
            text.set_text(new)
            changed.append([old, new])
    return changed
