"""Synthetic objective panel: instance specifications and objective values.

This is the objective-definition subset of the development module from which the
one-step, exploration, trajectory and verified-instance studies import their test
problems (the test-problem table of the appendix).  instances() returns the ten
problem/dimension specifications with their fixed rotations, weights, reference
minimizers and seed bases; objective(x, spec) evaluates the objective on the last
axis of x.  The runners verify the SHA-256 of this file.  No optimizer is defined here.
"""
import numpy as np

METHODS = ("cma", "es", "cbo")
VALIDATION_SEED_BASE = 972610000


def instances(seed_base=VALIDATION_SEED_BASE):
    rows = []
    for family, name in (("separable_pl", "Separable non-convex PL"),
                         ("rotated_anisotropic_pl", "Rotated anisotropic PL"),
                         ("quartic", "Multimodal quartic")):
        for d in (4, 8):
            rows.append(dict(id=f"{family}_d{d}", family=family, name=name, dimension=d))
    rows.extend(dict(id=f"{family}_d4", family=family, name=name, dimension=4)
                for family, name in (("rastrigin", "Rastrigin"), ("rosenbrock", "Rotated scaled Rosenbrock"),
                                     ("wells", "Unequal wells")))
    rows.append(dict(id="periodic_d1", family="periodic", name="Periodic", dimension=1))
    for j, row in enumerate(rows):
        d, family = row["dimension"], row["family"]
        row["f_star"] = 0.
        row["initial_seed_base"] = seed_base+1000*j
        row["algorithm_seeds"] = {m: row["initial_seed_base"]+(i+1)*100000 for i, m in enumerate(METHODS)}
        row["rotation_seed"] = (260930+d if family == "rotated_anisotropic_pl" else
                                 270929 if family == "rosenbrock" else None)
        Q = np.eye(d) if row["rotation_seed"] is None else np.linalg.qr(
            np.random.default_rng(row["rotation_seed"]).normal(size=(d, d)))[0]
        row["rotation"] = Q.tolist()
        row["weights"] = (np.linspace(1., 2., d) if family == "rotated_anisotropic_pl" else np.ones(d)).tolist()
        target = np.zeros(d)
        if family == "quartic": target = np.ones(d)
        if family == "rosenbrock": target = np.ones(d)@Q.T
        if family == "wells": target[0] = 1.
        row["target"] = target.tolist()
        row["minimizer_note"] = ("reference minimizer +1; all sign vectors are global minimizers" if family == "quartic" else
                                  "reference minimizer zero; all integer multiples of 2*pi are global minimizers" if family == "periodic" else
                                  "unique global minimizer")
        row["objective_formula"] = {
            "separable_pl": "sum_j(x_j^2+1.05*sin(x_j)^2)",
            "rotated_anisotropic_pl": "sum_j a_j*(z_j^2+1.05*sin(z_j)^2), z=Q^T*x, a_j linear from1to2",
            "quartic": "sum_j(x_j^2-1)^2",
            "rastrigin": "sum_j[x_j^2+10*(1-cos(2*pi*x_j))]",
            "rosenbrock": "sum_{j<d}[(z_{j+1}-z_j^2)^2+.01*(1-z_j)^2], z=Q^T*x",
            "wells": "(x_1^2-1)^2+.2*(x_1-1)^2+.5*sum_{j>1}x_j^2",
            "periodic": "1-cos(x_1)"}[family]
    return rows


def objective(x, spec):
    family = spec["family"]
    with np.errstate(over="ignore", invalid="ignore"):
        if family in ("separable_pl", "rotated_anisotropic_pl"):
            z = x if family == "separable_pl" else x@np.asarray(spec["rotation"])
            return np.sum(np.asarray(spec["weights"])*(z*z+1.05*np.sin(z)**2), axis=-1)
        if family == "quartic": return np.sum((x*x-1)**2, axis=-1)
        if family == "rastrigin": return np.sum(x*x+10*(1-np.cos(2*np.pi*x)), axis=-1)
        if family == "rosenbrock":
            z = x@np.asarray(spec["rotation"])
            return np.sum((z[..., 1:]-z[..., :-1]**2)**2+.01*(1-z[..., :-1])**2, axis=-1)
        if family == "wells":
            return (x[..., 0]**2-1)**2+.2*(x[..., 0]-1)**2+.5*np.sum(x[..., 1:]**2, axis=-1)
        if family == "periodic": return 1-np.cos(x[..., 0])
    raise ValueError(family)
