import numpy as np
import matplotlib.pyplot as plt


# Representative distributions
def p_gaussian(n_samples: int, rng: np.random.Generator) -> np.ndarray:

    return rng.normal(size=n_samples)

def p_rademacher(n_samples: int, rng: np.random.Generator) -> np.ndarray:

    return rng.choice([-1.0, 1.0], size=n_samples)

def p_uniform(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    
    return rng.uniform(low=-np.sqrt(3.0), high=np.sqrt(3.0), size=n_samples)

def p_student_t(n_samples: int, rng: np.random.Generator, nu: int) -> np.ndarray:

    assert nu > 2.0, "nu must be greater than 2 to obtain unit variance."

    return (np.sqrt((nu - 2.0) / nu) * rng.standard_t(df=nu, size=n_samples))



def generate_random_array(shape: tuple | list, v: float, tau: complex, 
                          rng: np.random.Generator, p_dist: callable = None) -> np.ndarray:
    """
    Generate random complex random variables with E[z] = 0, E[|z|^2] = v, E[z^2] = tau.
    Must satisfy |tau| <= v.
    
    p_dist: Callable that returns a one-dimensional NumPy array of independent
            real-valued random variables with mean zero and variance one.
            default (None): standard normal distribution N(0, 1)
    """
    if p_dist is None:
        p_dist = p_gaussian

    assert v >= np.abs(tau) - 1e-12, (
        f"Invalid generation: |tau|={np.abs(tau)} exceeds v={v}"
    )

    cov = np.array([
        [0.5 * (v + np.real(tau)), 0.5 * np.imag(tau)],
        [0.5 * np.imag(tau),       0.5 * (v - np.real(tau))]
    ])

    n_samples = int(np.prod(shape))
    U = np.asarray(p_dist(n_samples=n_samples, rng=rng))
    V = np.asarray(p_dist(n_samples=n_samples, rng=rng))
    assert U.shape == (n_samples,) and V.shape == (n_samples,)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    sqrt_cov = (
        eigenvectors
        @ np.diag(np.sqrt(eigenvalues))
        @ eigenvectors.T
    )

    standard_samples = np.column_stack((U, V))
    samples = standard_samples @ sqrt_cov.T

    X = samples[:, 0].reshape(shape)
    Y = samples[:, 1].reshape(shape)

    return X + 1j * Y



def single_SVD(params: dict, seed: int, p_dist: callable = None) -> tuple:

    required_keys = {
        'v_x', 'v_w', 'v_z', 
        'tau_x', 'tau_w', 'tau_z', 
        'c_1', 'c_2', 'n', 'd', 'p'
    }
    missing_keys = required_keys - set(params.keys())
    assert not missing_keys, f"Missing keys in 'params': {missing_keys}"

    n, d, p = params['n'], params['d'], params['p']
    rng = np.random.default_rng(seed)
    X = generate_random_array(shape=(n, d), v=params["v_x"], 
                              tau=params["tau_x"], rng=rng, p_dist=p_dist)
    W = generate_random_array(shape=(d, p), v=params["v_w"], 
                              tau=params["tau_w"], rng=rng, p_dist=p_dist)
    H = (X @ W) / np.sqrt(d)

    if params["v_z"] > 1e-10:
        Z = generate_random_array(shape=(n, p), v=params["v_z"], 
                                  tau=params["tau_z"], rng=rng, p_dist=p_dist)
    else:
        Z = np.zeros((n, p), dtype=complex)
        
    Q = params["c_1"] * H + params["c_2"] * np.conj(H) + Z

    singvals = np.linalg.svd(Q / np.sqrt(n), compute_uv=False)
    rank_Q = np.count_nonzero(singvals > 1e-10)
    nonzero_singvals = singvals[:rank_Q]
    nonzero_eigvals = nonzero_singvals ** 2

    return nonzero_eigvals, nonzero_singvals, rank_Q



def iteration_solver(E_array: np.ndarray, params: dict, rank_Q: int,
                     eta: float = 3e-5, tol: float = 1e-7) -> np.ndarray:

    Sigma_W = np.array([[params["v_w"], params["tau_w"]], 
                        [np.conj(params["tau_w"]), params["v_w"]]], dtype=complex)
        
    Sigma_CX = np.array([[np.abs(params["c_1"]) ** 2 * params["v_x"],    
                          np.conj(params["c_1"]) * params["c_2"] * np.conj(params["tau_x"])], 
                         [params["c_1"] * np.conj(params["c_2"]) * params["tau_x"], 
                          np.abs(params["c_2"]) ** 2 * params["v_x"]]], dtype=complex)

    phi = params["n"] / params["d"]
    psi = params["p"] / params["d"]
    
    rho_theory = []
    I2 = np.eye(2)
    eps_mat = 1e-9 * I2
    assert np.all(E_array[1:] - E_array[:-1] > 0.)
    z_init = E_array[-1] + 1j * eta
    g1 = 1j
    g2 = (psi / phi) * z_init * g1 + (psi / phi) - 1.0
    
    for E in E_array[::-1]:

        z = E + 1j * eta
        converged = False
        strategies = [(0.5, 1000), (0.1, 5000), (0.02, 10000)]

        for current_alpha, current_max_iter in strategies:

            for _ in range(current_max_iter):

                g1_old, g2_old = g1, g2
                
                Γ_W = psi * g1 * Sigma_W
                Γ_X = g2 * Sigma_CX
                GG = Γ_X @ np.linalg.inv(I2 - Γ_W @ Γ_X + eps_mat)
                
                g1_new = - 1.0 / (z + params["v_z"] * g2 + np.trace(Sigma_W @ GG))
                g1 = (1 - current_alpha) * g1_old + current_alpha * g1_new
                g2 = (psi / phi) * z * g1 + (psi / phi) - 1.0
                
                if np.abs(g1 - g1_old) < tol and np.abs(g2 - g2_old) < tol:
                    converged = True
                    break
            
            if converged:
                break
                
        if not converged:
            print(f"Warning: Iteration did not converge at E = {E:.5f}")
                
        raw_density = np.imag(g1) / np.pi
        if np.imag(g1) < -1e-10:
            print(f"Warning: negative Im(g1) at E={E:.5f}")

        nonzero_mass = rank_Q / params["p"]
        zero_mass = 1.0 - nonzero_mass

        if zero_mass <= 0.0:
            rho = raw_density
        else:
            lorentzian = eta / (np.pi * (E ** 2 + eta ** 2))
            rho = (raw_density - zero_mass * lorentzian) / nonzero_mass
            rho = np.maximum(0.0, rho)
            
        rho_theory.append(rho)
    
    return np.array(rho_theory)[::-1]



def cc(z, precision=3, tol=1e-12) -> str:    # format a complex number
        
        real = 0.0 if abs(z.real) < tol else z.real
        imag = 0.0 if abs(z.imag) < tol else z.imag

        if imag == 0:
            return f"{real:.{precision}g}"

        imag_abs = abs(imag)
        imag_str = "i" if abs(imag_abs - 1) < tol else f"{imag_abs:.{precision}g}i"

        if real == 0:
            return imag_str if imag > 0 else f"-{imag_str}"

        sign = "+" if imag > 0 else "-"

        return f"{real:.{precision}g}{sign}{imag_str}"



def plot_distribution(data: np.ndarray, params: dict, return_pdf: bool, 
                      ext: str | int, n_bins: int, rank_Q: int,
                      x_range: tuple | None = None, y_range: tuple | None = None,
                      is_sv: bool = True, split_outliers: bool = False) -> None:

    n, d, p = params["n"], params["d"], params["p"]
    # spike_ratio = min(1.0 / min(n, p), 0.02)
    # safe_percentile = 100.0 * (1.0 - 1.1 * spike_ratio)
    # d_max_plot = 1.12 * np.percentile(data, safe_percentile)
    if split_outliers:

        q1, q3 = np.quantile(data, [0.25, 0.75])
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        is_outlier = data > upper
        outliers = data[is_outlier]
        inliers = data[~is_outlier]
        d_max_plot = 1.1 * np.max(inliers)
        print("portion of outliers :", len(outliers) / len(data))
        print(np.sort(outliers)[::-1])

    else:
        d_max_plot = 1.1 * np.max(data)

    val_range = np.linspace(0.0, d_max_plot, 300)
    E_range = val_range ** 2 if is_sv else val_range
    rho_nonzero = iteration_solver(E_array=E_range, params=params, rank_Q=rank_Q)
    
    if is_sv:
        rho_theory_final = 2 * val_range * rho_nonzero
        filename = f"fg_SV_{ext}."
    else:
        rho_theory_final = rho_nonzero
        filename = f"fg_E_{ext}."

    filename += "pdf" if return_pdf else "png"
    title = (
        rf"$(n,d,p)=({n},{d},{p}),\quad N_{{\mathrm{{MC}}}}={params["n_mc"]},$"
        "\n"
        rf"$(c_2,\tau_x,\tau_w,v_z,\tau_z)"
        rf"=({cc(params['c_2'])},\,"
        rf"{cc(params['tau_x'])},\,"
        rf"{cc(params['tau_w'])},\,"
        rf"{params['v_z']},\,"
        rf"{cc(params['tau_z'])})$"
    )
    bins = np.linspace(0.0, d_max_plot, n_bins + 1)
    # hist_vals, _ = np.histogram(data, bins=bins, density=True)
    hist_vals, _ = np.histogram(inliers if split_outliers else data, bins=bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.figure(figsize=(7, 5))
    
    # Empirical data
    plt.fill_between(bin_centers, 0, hist_vals, step='mid', alpha=0.25, color='royalblue')
    plt.plot(bin_centers, hist_vals, color='royalblue', linewidth=1.75, 
             drawstyle='steps-mid', label="Empirical")

    # Theory
    plt.plot(val_range, rho_theory_final, 'r-', lw=2, label="Theory")
    
    # Formatting
    plt.title(title)
    if x_range is None:
        plt.xlim(0.0, d_max_plot)
    else:
        plt.xlim(x_range[0], x_range[1])
    if y_range is not None:
        plt.ylim(y_range[0], y_range[1])
    plt.xlabel("Singular value" if is_sv else "Eigenvalue")
    plt.ylabel("Density")
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(color="gray", linestyle=":")
    plt.savefig(filename, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.show()

    if split_outliers:
        # Outlier data
        w = np.max(outliers) - np.min(outliers)
        bins = np.linspace(np.min(outliers) - 0.15 * w, np.max(outliers) + 0.15 * w, 20)
        hist_vals, _ = np.histogram(outliers, bins=bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        plt.figure(figsize=(7, 5))
        plt.fill_between(bin_centers, 0, hist_vals, step='mid', alpha=0.25, color='royalblue')
        plt.plot(bin_centers, hist_vals, color='royalblue', linewidth=1.75, 
                 drawstyle='steps-mid', label="Empirical")
        plt.show()

