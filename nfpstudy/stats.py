"""Small statistics helpers (no statsmodels dependency)."""
import numpy as np
from scipy import stats


def ols_hc3(y, X):
    """OLS with HC3 robust covariance. X must include the constant column.

    p-values use the normal distribution (same convention as statsmodels'
    cov_type='HC3' default). The joint test is a robust Wald test that all slopes = 0 (F reference).
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    S = XtXi @ (X.T * (e**2 / (1 - h) ** 2)) @ X @ XtXi
    se = np.sqrt(np.diag(S))
    t = b / se
    p = 2 * stats.norm.sf(np.abs(t))
    r2 = 1 - (e @ e) / ((y - y.mean()) ** 2).sum()
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    W = b[1:] @ np.linalg.solve(S[1:, 1:], b[1:])
    q = k - 1
    return dict(b=b, se=se, t=t, p=p, r2=r2, adj=adj, n=n,
                joint_p=stats.f.sf(W / q, q, n - k))  # robust Wald/F test of all slopes = 0


def fit(df, y, cols):
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols])
    return ols_hc3(df[y].to_numpy(float), X)


def clopper_pearson(k, n, alpha=0.05):
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def binom_p(k, n):
    return stats.binomtest(int(k), int(n), 0.5).pvalue
