import numpy as np

def shrink(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

def svt(X, tau):
    U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
    return U @ np.diag(shrink(sigma, tau)) @ Vt

def rpca_ialm(M, lam=None, tol=1e-7, max_iter=500, rho=1.5, verbose=False):
    M = np.asarray(M, dtype=np.float64)
    m, n = M.shape
 
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))
 
    norm_two = np.linalg.norm(M, 2)
    norm_inf = np.abs(M).max() / lam
    Y = M / max(norm_two, norm_inf)
    S = np.zeros_like(M)
    mu = 1.25 / norm_two
    mu_max = mu * 1e7
    norm_M = np.linalg.norm(M, "fro")
 
    history = []
    for it in range(max_iter):
        L = svt(M - S + Y / mu, 1.0 / mu)
 
        S = shrink(M - L + Y / mu, lam / mu)
 
        R = M - L - S
        err = np.linalg.norm(R, "fro") / norm_M
        history.append(err)
        if verbose and it % 5 == 0:
            print(f"iter {it:3d}  rel_residual {err:.3e}  mu {mu:.3e}")
        if err < tol:
            break
 
        Y = Y + mu * R
        mu = min(rho * mu, mu_max)
 
    return L, S, history