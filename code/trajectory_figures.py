"""Complete-panel finite-population summaries and conditional-discovery bounds."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import discovery_inference as inference

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/"results/trajectories/finite-population-full"
BUDGETS = np.array([1024, 2048, 4096, 8192])
ENDPOINTS = np.array(inference.ENDPOINTS)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze():
    summary = json.loads((FOLDER/"summary.json").read_text())
    assert summary["status"] == "full" and summary["runs_per_cell"] == 128
    assert summary["generations"] == 200 and summary["probe_size"] == 128
    specs, families = summary["instances"], summary["families"]
    populations = np.array(summary["populations"])
    cells = [(spec, family) for spec in specs for family in families]
    # axes case, N, assembly, order, batch-stream, epsilon, endpoint.
    shape = (20, 4, 8, 2, 5, 3, 4)
    upper = np.empty(shape)
    failure = np.empty(shape)
    discounted = np.empty(shape)
    undiscounted = np.empty(shape)
    joint = np.empty(shape[:-2]+(4,))  # Primary epsilon; endpoint axis retained.
    final_flag_fraction = np.empty((20, 4, 8, 2, 3))
    archive_failure = np.empty((20, 4, 8, 2, 3, 4))
    archive_budget_failure = np.empty((20, 4, 8, 2, 3, 4))
    costs = np.empty((20, 4, 8, 2, 4))
    initial_seen = {}
    for ci, (spec, family) in enumerate(cells):
        for ni, population in enumerate(populations):
            key = f"{spec['id']}__{family}__N{population}"
            path = FOLDER/(key+".npz")
            assert digest(path) == summary["cases"][key]["output_sha256"]
            with np.load(path) as data:
                probe = data["probe_success_counts"]
                assert probe.shape == (128, 8, 2, 200, 3)
                flag_counts = inference.low_mass_flags(probe).sum(axis=0)
                final_flag_fraction[ci, ni] = flag_counts[:, :, -1]/128
                # inference has time last: assembly, order, epsilon, generation.
                flag_counts = np.moveaxis(flag_counts, -1, -2)
                initial = data["initial_points"]
                if ci in initial_seen:
                    old = initial_seen[ci]
                    assert np.array_equal(old, initial[:, :old.shape[1]])
                initial_seen[ci] = initial.copy()
                offspring = 1-data["offspring_cumulative_hit"].mean(axis=0)
                audits = data["audit_cumulative_miss"].mean(axis=0)
                trace = data["traces"]
                all_costs = data["evolution_cumulative_calls_by_category"].sum(axis=-1)
                assert (all_costs[..., -1] >= BUDGETS[-1]).all()
                for ei, endpoint in enumerate(ENDPOINTS):
                    costs[ci, ni, :, :, ei] = np.mean(all_costs[..., endpoint], axis=0)
                    archive_failure[ci, ni, :, :, :, ei] = np.mean(
                        trace[..., endpoint, 0, None] > np.array(summary["epsilons"]), axis=0)
                    for bi, batch in enumerate([int(population), 1, 4, 16, 32]):
                        result = inference.discovery_bound(flag_counts, 128, batch, int(endpoint))
                        upper[ci, ni, :, :, bi, :, ei] = result["upper"]
                        discounted[ci, ni, :, :, bi, :, ei] = result["flag_discounted"]
                        undiscounted[ci, ni, :, :, bi, :, ei] = result["flag_undiscounted"]
                        observed = offspring[:, :, endpoint] if bi == 0 else audits[:, :, endpoint, bi-1]
                        failure[ci, ni, :, :, bi, :, ei] = observed
                        # At the final endpoint this is simultaneous over all 6400
                        # configurations/batch streams for the primary epsilon.
                        simultaneous = inference.discovery_bound(
                            flag_counts[:, :, 0], 128, batch, int(endpoint), alpha=.05/6400)
                        joint[ci, ni, :, :, bi, ei] = simultaneous["upper"]
                for bgi, budget in enumerate(BUDGETS):
                    last = np.sum(all_costs <= budget, axis=-1)-1
                    assert (last >= 0).all()
                    record = np.take_along_axis(trace[..., 0], last[..., None], axis=-1)[..., 0]
                    archive_budget_failure[ci, ni, :, :, :, bgi] = np.mean(
                        record[..., None] > np.array(summary["epsilons"]), axis=0)
    assert np.isfinite(upper).all() and np.all((0 <= upper) & (upper <= 1))
    assert np.all(joint >= upper[..., 0, :]-1e-14)
    assert np.all(np.diff(failure, axis=-1) <= 1e-14)
    assert np.all(np.diff(archive_budget_failure, axis=-1) <= 1e-14)
    target = ROOT/"results/trajectories/finite_population_analysis.npz"
    np.savez_compressed(target, upper=upper, failure=failure, joint_primary=joint,
                        flag_discounted=discounted, flag_undiscounted=undiscounted,
                        final_flag_fraction=final_flag_fraction, archive_failure=archive_failure,
                        archive_budget_failure=archive_budget_failure, mean_calls=costs,
                        endpoints=ENDPOINTS, budgets=BUDGETS, populations=populations,
                        epsilons=summary["epsilons"],
                        cases=np.array([s["id"]+"__"+f for s, f in cells]))
    audit = {"source_summary_sha256": digest(FOLDER/"summary.json"),
             "script_sha256": digest(__file__),
             "inference_sha256": digest(inference.__file__), "arrays_sha256": digest(target),
             "coverage": "Pointwise 95% per fixed cell/tolerance/batch/endpoint; joint_primary at k=200 "
                "is simultaneous over 1280 cells and five streams at epsilon=.1 only.",
             "epsilons": summary["epsilons"], "populations": populations.tolist(),
             "evolution_objective_calls": summary["evolution_total_objective_calls"],
             "diagnostic_objective_calls": summary["diagnostic_total_objective_calls"],
             "by_population": [], "by_problem": {}, "figures": {}}
    for ni, population in enumerate(populations):
        row = {"N": int(population), "cells": 320,
               "offspring_failure_k200": float(failure[:, ni, :, :, 0, 0, -1].mean()),
               "archive_failure_k200": float(archive_failure[:, ni, :, :, 0, -1].mean()),
               "archive_failure_by_budget": archive_budget_failure[:, ni, :, :, 0].mean(axis=(0, 1, 2)).tolist(),
               "mean_operational_calls_k200": float(costs[:, ni, :, :, -1].mean()),
               "low_mass_flag_fraction_k200": float(final_flag_fraction[:, ni, :, :, 0].mean()),
               "pointwise_05_cells_by_stream": (upper[:, ni, :, :, :, 0, -1] <= .05).sum(axis=(0, 1, 2)).tolist(),
               "joint_10_cells_by_stream": (joint[:, ni, :, :, :, -1] <= .1).sum(axis=(0, 1, 2)).tolist(),
               "failure_by_audit_batch": failure[:, ni, :, :, 1:, 0, -1].mean(axis=(0, 1, 2)).tolist(),
               "median_upper_by_audit_batch": np.median(upper[:, ni, :, :, 1:, 0, -1], axis=(0, 1, 2)).tolist()}
        audit["by_population"].append(row)
    for ii, spec in enumerate(specs):
        select = slice(2*ii, 2*ii+2)
        audit["by_problem"][spec["id"]] = {
            "offspring_failure_by_N": failure[select, :, :, :, 0, 0, -1].mean(axis=(0, 2, 3)).tolist(),
            "archive_budget8192_failure_by_N": archive_budget_failure[select, :, :, :, 0, -1].mean(axis=(0, 2, 3)).tolist()}
    plot(summary, failure, upper, discounted, undiscounted, audit)
    (ROOT/"results/trajectories/finite_population_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit["by_population"], indent=2))
    print("PASS: all 80 files/1280 cells; probability bounds, nested starts, monotonicity, costs and hashes")


def plot(summary, failure, upper, discounted, undiscounted, audit):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    labels = []
    names = {"separable_pl": "PL", "rotated_anisotropic_pl": "Rot. PL",
             "quartic": "Quartic", "rastrigin": "Rastrigin", "rosenbrock": "Rosenbrock",
             "wells": "Wells", "periodic": "Periodic"}
    for spec in summary["instances"]:
        short = names[spec["family"]]+f" {spec['dimension']}D"
        labels.extend([short+" / B", short+" / C"])
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.4), sharex=True, sharey=True,
                             layout="constrained")
    for ni, ax in enumerate(axes.flat):
        matrix = failure[:, ni, :, :, 0, 0, -1].reshape(20, 16)
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma_r", aspect="auto")
        certified = upper[:, ni, :, :, 0, 0, -1].reshape(20, 16) <= .05
        y, x = np.nonzero(certified)
        ax.scatter(x, y, s=4, c="#218c36", marker="o")
        ax.set_title(f"Population N={summary['populations'][ni]}")
        ax.set_xticks(np.arange(16), [a["id"].replace("S", "").replace("_R", "").replace("_H", "")
                                    + ("a" if order == "MRS" else "b")
                                    for a in summary["assemblies"] for order in summary["orders"]],
                      rotation=90, fontsize=7)
        ax.set_yticks(np.arange(20), labels, fontsize=7)
        ax.set_xticks(np.arange(.5, 16, 2), minor=True)
        ax.grid(which="minor", axis="x", color="white", lw=.25)
    fig.colorbar(im, ax=axes, location="right", shrink=.65, label="Offspring-only failure frequency by generation 200")
    fig.supxlabel("Switches SRH; a: S then R, b: R then S.  Green dot: pointwise 95% upper bound <= 0.05", fontsize=8)
    destination = ROOT/"figures/trajectory_failures.pdf"
    fig.savefig(destination); plt.close(fig)
    audit["figures"][destination.name] = digest(destination)
    # Controlled audit keeps N=32 fixed while changing the independent tested batch.
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.45), sharex=True, sharey=True,
                             layout="constrained")
    ni = summary["populations"].index(32)
    for bi, (batch, ax) in enumerate(zip((1, 4, 16, 32), axes)):
        x = failure[:, ni, :, :, bi+1, 0, -1].ravel()
        y = upper[:, ni, :, :, bi+1, 0, -1].ravel()
        ax.scatter(x, y, s=11, alpha=.35, color="#16729b", edgecolors="none")
        ax.plot([0, 1], [0, 1], color="0.4", ls="--", lw=.7)
        ax.axhline(.05, color="#218c36", ls=":", lw=.7)
        ax.set_title(f"Audit batch b={batch}")
        ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02))
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Pointwise 95% discovery upper bound")
    fig.supxlabel("Observed audit non-discovery frequency by generation 200", fontsize=8)
    destination = ROOT/"figures/trajectory_discovery_bounds.pdf"
    fig.savefig(destination); plt.close(fig)
    audit["figures"][destination.name] = digest(destination)


if __name__ == "__main__":
    inference.checks()
    analyze()
