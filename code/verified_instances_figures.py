"""Figure for the three verified instances (verified-instances protocol); reads saved data only."""
from pathlib import Path
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import verified_instances as experiment


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/"results/verified-instances/verified-instances-full"
OUTPUT = ROOT/"figures/verified_instances.pdf"
AUDIT = ROOT/"results/verified-instances/verified_instances_figure_audit.json"
RES_LABELS = {"separable_pl_d4": "PL 4D", "rotated_anisotropic_pl_d4": "Rot. PL 4D", "quartic_d4": "Quartic 4D (box)"}
RES_COLORS = {"separable_pl_d4": "#246f96", "rotated_anisotropic_pl_d4": "#c16c23", "quartic_d4": "#347c63"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary = json.loads((FOLDER/"summary.json").read_text())
    assert summary["status"] == "full"
    assert summary["sources_sha256"]["runner"] == sha256(experiment.__file__)
    assert summary["sources_sha256"]["protocol"] == sha256(experiment.PROTOCOL)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 6.8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.65), layout="constrained")
    with np.load(FOLDER/"arrays.npz") as data:
        # (a) CBO: attraction to the weighted consensus and its bias.
        ax, cbo = axes[0], data["cbo"]
        t = np.arange(cbo.shape[1])*experiment.CBO["dt"]
        for j, (label, scale, color, ls) in enumerate((("$V_*$", 1, "#222222", "-"),
                                                      (r"$2W_\alpha$", 2, "#246f96", "--"),
                                                      (r"$2\|v_\alpha-y_*\|^2$", 2, "#c16c23", "-."))):
            values = scale*cbo[:, :, j]
            ax.plot(t, np.median(values, axis=0), color=color, ls=ls, lw=1.2, label=label)
            if j == 0:
                ax.fill_between(t, *np.quantile(values, [.1, .9], axis=0), color=color, alpha=.12, lw=0)
        rate = summary["cbo"]["centered_coefficient"]
        start = np.median(cbo[:, 0, 0])
        ax.plot(t, start*np.exp(-rate*t), color="0.5", ls=":", lw=.9, label=r"slope $2\lambda-d\sigma^2$")
        ax.set(yscale="log", xlabel="Time $t$", title="(a) CBO, fixed $\\alpha$", xlim=(0, t[-1]))
        ax.set_ylim(1e-6, 60)
        ax.legend(frameon=False, loc="upper right", handlelength=2.4)
        # (b) CMA-ES-type example: expected flow versus the certified rate.
        ax, times = axes[1], data["cma_times"]
        flow = data["cma_flow_L"]/data["cma_flow_L"][:, :1]
        for i, curve in enumerate(flow):
            ax.plot(times, curve, color="#246f96", lw=1.3 if i == 0 else .7, alpha=1 if i == 0 else .6,
                    label="Expected flow" if i == 0 else None)
        stochastic = data["cma_tau0.05_L"][0]
        steps = np.arange(stochastic.shape[1])*.05
        ratio = stochastic/stochastic[:, :1]
        ax.plot(steps, np.median(ratio, axis=0), color="#c16c23", lw=1., label=r"Finite step $\tau=0.05$ (median)")
        ax.fill_between(steps, *np.quantile(ratio, [.1, .9], axis=0), color="#c16c23", alpha=.15, lw=0)
        ax.plot(times, np.exp(-times/4), color="k", ls="--", lw=1., label="Certified $e^{-t/4}$")
        ax.set(yscale="log", xlabel="Time $t$", title="(b) CMA-ES-type flow, $1-\\cos y$", xlim=(0, times[-1]))
        ax.set_ylabel(r"$L(t)/L(0)$")
        ax.legend(frameon=False, loc="lower left")
        # (c) Recombinative ES: run-average mean gap versus the certified envelope.
        ax = axes[2]
        for name in experiment.RES_INSTANCES:
            gaps, envelope = data[f"res_{name}_gaps"], data[f"res_{name}_envelope"]
            scale = gaps[:, 0].mean()
            k = np.arange(gaps.shape[1])
            ax.plot(k, gaps.mean(axis=0)/scale, color=RES_COLORS[name], lw=1.3, label=RES_LABELS[name])
            ax.plot(k, envelope.mean(axis=0)/scale, color=RES_COLORS[name], lw=1., ls="--")
        ax.plot([], [], color="0.3", ls="--", lw=1., label="Certified envelope")
        ax.set(yscale="log", xlabel="Generation $k$", title="(c) Recombinative ES", xlim=(0, k[-1]))
        ax.set_ylabel(r"Mean gap $\bar{\mathcal{E}}_k/\bar{\mathcal{E}}_0$")
        ax.legend(frameon=False, loc="lower left")
    for ax in axes:
        ax.grid(alpha=.18, lw=.5)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, metadata={"Title": "verified_instances", "CreationDate": None, "ModDate": None})
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=200)
    plt.close(fig)
    audit = {"source_summary_sha256": sha256(FOLDER/"summary.json"), "arrays_sha256": sha256(FOLDER/"arrays.npz"),
             "script_sha256": sha256(__file__), "figure_sha256": sha256(OUTPUT),
             "cma_flow_certificate_holds": summary["cma"]["flow_certificate_holds"],
             "res_certificate_consistent": {n: summary["res"][n]["certificate_consistent"] for n in experiment.RES_INSTANCES}}
    AUDIT.write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
