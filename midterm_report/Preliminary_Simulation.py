import numpy as np
import matplotlib.pyplot as plt


def build_array_geometry(
    n_panels: int = 3,
    l_panel: float = 1.0,
    l_hinge: float = 0.15,
    ea_panel: float = 1e4,
    ea_hinge: float = 5e2,
    ei_panel: float = 1e4,
    rho_lin: float = 0.5,
    hinge_mass_scale: float = 0.05,
    phi_init: list[float] | None = None,
    phi_target: list[float] | None = None,
    root_bc: str = "clamped",
):
    """
    Constructs alternating rigid panel and compliant hinge segments (default: 3 panels,
    2 hinges -> 5 edges, 6 nodes, 12 planar DOFs) and computes Voronoi lumped masses.
    """
    n_hinges = max(n_panels - 1, 0)
    if phi_init is None:
        # Folded/stowed relative hinge turn angles (rad)
        phi_init = [0.35 * (-1) ** i for i in range(n_hinges)]
    if phi_target is None:
        # Deployed flat/target relative hinge turn angles (rad)
        phi_target = [0.0 for _ in range(n_hinges)]

    edges = []
    l_rest = []
    ea_seg = []
    ei_seg = []
    m_seg = []

    for p in range(n_panels):
        edges.append(("panel", p))
        l_rest.append(l_panel)
        ea_seg.append(ea_panel)
        ei_seg.append(ei_panel)
        m_seg.append(rho_lin * l_panel)

        if p < n_hinges:
            edges.append(("hinge", p))
            l_rest.append(l_hinge)
            ea_seg.append(ea_hinge)
            ei_seg.append(1.0)  # overwritten during outer stiffness loop
            m_seg.append(rho_lin * hinge_mass_scale * l_hinge)

    l_rest = np.asarray(l_rest, dtype=float)
    ea_seg = np.asarray(ea_seg, dtype=float)
    ei_seg = np.asarray(ei_seg, dtype=float)
    m_seg = np.asarray(m_seg, dtype=float)

    n_edges = len(edges)
    n_nodes = n_edges + 1
    n_dofs = 2 * n_nodes

    # Voronoi lumped nodal masses
    m_nodes = np.zeros(n_nodes, dtype=float)
    for e in range(n_edges):
        m_nodes[e] += 0.5 * m_seg[e]
        m_nodes[e + 1] += 0.5 * m_seg[e]
    m_nodes = np.maximum(m_nodes, 1e-6)

    mass_diag = np.repeat(m_nodes, 2)
    M = np.diag(mass_diag)

    # Constrained and free DOF indices
    fixed_nodes = [0, 1] if root_bc == "clamped" else [0]
    constrained_dofs = []
    for k in fixed_nodes:
        constrained_dofs.extend([2 * k, 2 * k + 1])
    constrained_dofs = np.array(sorted(set(constrained_dofs)), dtype=int)
    all_dofs = np.arange(n_dofs, dtype=int)
    free_dofs = np.setdiff1d(all_dofs, constrained_dofs)

    # Build initial configuration q0 from phi_init
    q0 = np.zeros(n_dofs, dtype=float)
    q0[0], q0[1] = 0.0, 0.0
    if n_nodes > 1:
        q0[2], q0[3] = l_rest[0], 0.0

    # Track cumulative planar tangent angle along edges
    theta = 0.0
    hinge_idx = 0
    for e in range(1, n_edges):
        prev_type, _ = edges[e - 1]
        if prev_type == "hinge" and hinge_idx < len(phi_init):
            theta += phi_init[hinge_idx]
            hinge_idx += 1
        dx = l_rest[e] * np.cos(theta)
        dy = l_rest[e] * np.sin(theta)
        q0[2 * (e + 1)] = q0[2 * e] + dx
        q0[2 * (e + 1) + 1] = q0[2 * e + 1] + dy

    return {
        "edges": edges,
        "n_edges": n_edges,
        "n_nodes": n_nodes,
        "n_dofs": n_dofs,
        "l_rest": l_rest,
        "ea_seg": ea_seg,
        "ei_seg": ei_seg,
        "m_nodes": m_nodes,
        "M": M,
        "constrained_dofs": constrained_dofs,
        "free_dofs": free_dofs,
        "phi_target": np.asarray(phi_target, dtype=float),
        "q0": q0,
    }


def compute_energies(q: np.ndarray, model: dict, ei_hinge_val: float):
    """
    Computes axial stretching energy E_s and discrete bending energy E_b
    across interior vertices relative to phi_target.
    """
    n_edges = model["n_edges"]
    l_rest = model["l_rest"]
    ea_seg = model["ea_seg"]
    ei_seg = model["ei_seg"].copy()
    edges = model["edges"]
    phi_target = model["phi_target"]

    # Assign current hinge stiffness to hinge edges
    for e, (etype, _) in enumerate(edges):
        if etype == "hinge":
            ei_seg[e] = ei_hinge_val

    pts = q.reshape(-1, 2)
    vecs = pts[1:] - pts[:-1]  # shape (n_edges, 2)
    lengths = np.linalg.norm(vecs, axis=1)
    lengths_safe = np.maximum(lengths, 1e-12)

    # Stretching energy: 0.5 * EA * ((L - L0)^2 / L0)
    strain_stretch = (lengths - l_rest) / l_rest
    e_s = 0.5 * np.sum(ea_seg * (strain_stretch ** 2) * l_rest)

    # Bending energy at interior vertices i = 1 .. n_edges - 1
    e_b = 0.0
    hinge_ptr = 0
    for i in range(1, n_edges):
        u_prev = vecs[i - 1] / lengths_safe[i - 1]
        u_next = vecs[i] / lengths_safe[i]
        # Signed planar angle between edge i-1 and edge i
        cross_z = u_prev[0] * u_next[1] - u_prev[1] * u_next[0]
        dot_val = np.clip(np.dot(u_prev, u_next), -1.0, 1.0)
        angle_turn = np.arctan2(cross_z, dot_val)

        target_angle = (
            phi_target[hinge_ptr]
            if (edges[i - 1][0] == "hinge" and hinge_ptr < len(phi_target))
            else 0.0
        )
        if edges[i - 1][0] == "hinge":
            hinge_ptr += 1

        # Effective bending stiffness at vertex i (voronoi-like average / min)
        ei_vert = min(ei_seg[i - 1], ei_seg[i])
        l_eff = 0.5 * (l_rest[i - 1] + l_rest[i])
        kappa = (angle_turn - target_angle) / max(l_eff, 1e-12)
        e_b += 0.5 * ei_vert * (kappa ** 2) * l_eff

    return float(e_s), float(e_b)


def elastic_force_and_hessian(q: np.ndarray, model: dict, ei_hinge_val: float, eps: float = 1e-4):
    """
    Central-difference elastic force F_el = -grad_q(E_s + E_b) with a
    well-conditioned stiffness approximation for stable Newton-Raphson steps.
    """
    n_dofs = len(q)

    def total_potential(x):
        es, eb = compute_energies(x, model, ei_hinge_val)
        return es + eb

    grad = np.zeros(n_dofs, dtype=float)
    for j in range(n_dofs):
        step = np.zeros(n_dofs, dtype=float)
        step[j] = eps
        f_plus = total_potential(q + step)
        f_minus = total_potential(q - step)
        grad[j] = (f_plus - f_minus) / (2.0 * eps)

    # Regularized stiffness estimate (avoids 1/eps^2 noise amplification)
    k_scale = max(float(ei_hinge_val), 10.0)
    H = k_scale * np.eye(n_dofs, dtype=float)
    return -grad, H


def simulate_deployment(
    model: dict,
    ei_hinge_val: float,
    dt: float = 0.005,
    t_end: float = 1.5,
    c_damp: float = 15,
    tol_nr: float = 1e-7,
    iter_max: int = 25,
):
    """
    Inner implicit Euler + Newton-Raphson time-stepping loop.
    """
    n_dofs = model["n_dofs"]
    M = model["M"]
    C = c_damp * np.eye(n_dofs)
    free = model["free_dofs"]
    const = model["constrained_dofs"]

    q = model["q0"].copy()
    v = np.zeros(n_dofs, dtype=float)
    t = 0.0

    es0, eb0 = compute_energies(q, model, ei_hinge_val)
    ekin0 = 0.5 * float(v.T @ M @ v)

    t_hist = [0.0]
    q_hist = [q.copy()]
    v_hist = [v.copy()]
    es_hist = [es0]
    eb_hist = [eb0]
    ekin_hist = [ekin0]
    nr_iters_hist = []

    while t < t_end - 1e-15:
        h = min(dt, t_end - t)
        q_n = q.copy()
        v_n = v.copy()

        q_k = q_n + h * v_n
        q_k[const] = q_n[const]

        iters_used = 0
        for k in range(iter_max):
            iters_used = k + 1
            F_el, H_el = elastic_force_and_hessian(q_k, model, ei_hinge_val)
            v_trial = (q_k - q_n) / h

            r = (M / (h ** 2)) @ (q_k - q_n - h * v_n) - F_el + C @ v_trial
            r_f = r[free]

            if np.linalg.norm(r_f, ord=np.inf) < tol_nr:
                break

            J_ff = (
                (M[np.ix_(free, free)] / (h ** 2))
                + H_el[np.ix_(free, free)]
                + (C[np.ix_(free, free)] / h)
            )
            dq_f = np.linalg.solve(J_ff, -r_f)
            max_step = 0.05
            dq_norm = np.linalg.norm(dq_f, ord=np.inf)
            if dq_norm > max_step:
                dq_f = dq_f * (max_step / dq_norm)
            q_k[free] = q_k[free] + dq_f

        q = q_k.copy()
        v = (q - q_n) / h
        v[const] = 0.0
        t = t + h

        es, eb = compute_energies(q, model, ei_hinge_val)
        ekin = 0.5 * float(v.T @ M @ v)

        t_hist.append(t)
        q_hist.append(q.copy())
        v_hist.append(v.copy())
        es_hist.append(es)
        eb_hist.append(eb)
        ekin_hist.append(ekin)
        nr_iters_hist.append(iters_used)

    q_arr = np.array(q_hist)
    v_arr = np.array(v_hist)
    # Tip speed magnitude at final node (last two DOFs)
    tip_speeds = np.linalg.norm(v_arr[:, -2:], axis=1)
    v_final = float(tip_speeds[-1])

    return {
        "t": np.array(t_hist),
        "q": q_arr,
        "v": v_arr,
        "tip_speeds": tip_speeds,
        "v_final": v_final,
        "E_s": np.array(es_hist),
        "E_b": np.array(eb_hist),
        "E_kin": np.array(ekin_hist),
        "nr_iters": np.array(nr_iters_hist),
        "ei_hinge": ei_hinge_val,
    }


def optimize_hinge_stiffness(
    model: dict,
    ei_low: float = 5.0,
    ei_high: float = 500.0,
    v_target: float = 0.15,
    tol_v: float = 0.01,
    opt_max: int = 12,
    sim_kwargs: dict | None = None,
):
    """
    Outer bracketed/bisection loop tuning EI_hinge so final tip lock-out speed
    approaches v_target within tol_v.
    """
    if sim_kwargs is None:
        sim_kwargs = {}

    low = ei_low
    high = ei_high
    best_res = None
    best_err = np.inf
    history = []

    for opt_iter in range(1, opt_max + 1):
        ei_curr = 0.5 * (low + high)
        print(f"Running outer iter {opt_iter}/{opt_max} with EI_hinge = {ei_curr:.2f}...", flush=True)
        res = simulate_deployment(model, ei_curr, **sim_kwargs)
        err = res["v_final"] - v_target
        abs_err = abs(err)

        history.append((opt_iter, ei_curr, res["v_final"], err))

        if abs_err < best_err:
            best_err = abs_err
            best_res = res

        if abs_err <= tol_v:
            return ei_curr, res, history

        # Higher stiffness typically increases deployment/lock-out speed
        if err > 0:
            high = ei_curr
        else:
            low = ei_curr

    return best_res["ei_hinge"], best_res, history


if __name__ == "__main__":
    model = build_array_geometry(
        n_panels=3,
        l_panel=1.0,
        l_hinge=0.15,
        ea_panel=1e4,
        ea_hinge=5e2,
        ei_panel=1e4,
        rho_lin=0.5,
        phi_init=[0.35, -0.25],
        phi_target=[0.0, 0.0],
        root_bc="clamped",
    )

    sim_params = {
        "dt": 0.005,
        "t_end": 1.2,
        "c_damp": 15,
        "tol_nr": 1e-7,
        "iter_max": 25,
    }

    ei_opt, best_run, opt_log = optimize_hinge_stiffness(
        model,
        ei_low=10.0,
        ei_high=400.0,
        v_target=0.20,
        tol_v=0.01,
        opt_max=10,
        sim_kwargs=sim_params,
    )

    print(f"Optimized EI_hinge: {ei_opt:.3f} N*m^2 | Final tip speed: {best_run['v_final']:.4f} m/s", flush=True)
    for it, ei_val, vf, er in opt_log:
        print(f"  Iter {it:2d}: EI={ei_val:8.3f}, v_final={vf:.4f} m/s, err={er:+.4f} m/s", flush=True)

    # Preliminary Results Plots: Tip Speed & Energy Evolution
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(best_run["t"], best_run["tip_speeds"], linewidth=2)
    axes[0].set_title("Tip Speed During Deployment")
    axes[0].set_xlabel("Time t [s]")
    axes[0].set_ylabel("Tip Speed [m/s]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(best_run["t"], best_run["E_s"], label="Stretching $E_s$")
    axes[1].plot(best_run["t"], best_run["E_b"], label="Bending $E_b$")
    axes[1].plot(best_run["t"], best_run["E_kin"], label="Kinetic $E_{kin}$")
    axes[1].plot(
        best_run["t"],
        best_run["E_s"] + best_run["E_b"] + best_run["E_kin"],
        "--",
        label="Total $E_s + E_b + E_{kin}$",
    )
    axes[1].set_title("System Energy Evolution")
    axes[1].set_xlabel("Time t [s]")
    axes[1].set_ylabel("Energy [J]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("preliminary_results.png", dpi=200)
    plt.show()


    # --- Temporal Convergence Study ---
dt_list = [0.02, 0.01, 0.005, 0.0025, 0.001, 0.0005]
conv_results = []

for dt_val in dt_list:
    res_dt = simulate_deployment(
        model,
        ei_hinge_val=ei_opt,
        dt=dt_val,
        t_end=1.2,
        c_damp=8.0,
        tol_nr=1e-7,
        iter_max=25,
    )
    conv_results.append((dt_val, res_dt["v_final"], float(np.max(res_dt["tip_speeds"]))))

v_ref = conv_results[-1][1]
print("\nTemporal Convergence Study (Reference dt = 0.0005 s):", flush=True)
for dt_val, vf, vmax in conv_results:
    rel_err = abs(vf - v_ref) / max(abs(v_ref), 1e-12)
    print(f"  dt = {dt_val:7.4f} s | v_final = {vf:.5f} m/s | v_max = {vmax:.5f} m/s | rel_err = {rel_err:.3e}", flush=True)