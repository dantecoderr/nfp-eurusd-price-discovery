"""Whole-sample test of post-announcement drift (the thesis tests only a few subgroups).

Rule under test (no parameters): after the first minute closes, trade in the direction implied by
the NFP surprise (short EUR/USD after a beat, long after a miss) and exit h minutes later.
P&L in pips = D * POST_h, where POST_h = close(08:30+h) - close(08:30). Cost is NOT deducted unless stated."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from scipy import stats
from nfpstudy.data import load

d = load(); d = d[d.D != 0].copy()
HS = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60]
print("A. Signed drift by exit horizon, all releases with a non-zero NFP surprise (n=%d)" % len(d))
rows = []
for h in HS:
    v = (d.D * d[f"POST_{h}"]).dropna(); t = stats.ttest_1samp(v, 0)
    rows.append((h, len(v), v.mean(), v.median(), t.statistic, t.pvalue, (v > 0).mean()))
tab = pd.DataFrame(rows, columns=["h", "n", "mean_pips", "median", "t", "p", "win_rate"]); print(tab.round(3).to_string(index=False))
tab.to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "02_drift_by_horizon.csv"), index=False)

print("\nB. By side (beat -> short, miss -> long)")
for lab, m in {"beats (short)": d.D < 0, "misses (long)": d.D > 0}.items():
    print("  ", lab, " | ".join(f"T{h}: {(d.D*d[f'POST_{h}'])[m].dropna().mean():+.2f} (t={stats.ttest_1samp((d.D*d[f'POST_{h}'])[m].dropna(), 0).statistic:+.2f})" for h in (5, 10, 15, 30, 60)))

print("\nC. Multiple-testing check: max t over exit horizons 1..60, sign-flip null (5000 draws)")
M = np.column_stack([d[f"POST_{h}"].values for h in range(1, 61)]); D = d.D.values
ok = ~np.isnan(M).any(1); M, D = M[ok], D[ok]; n = len(D)
tmax = lambda Dv: ((Dv[:, None] * M).mean(0) / ((Dv[:, None] * M).std(0, ddof=1) / np.sqrt(n)))
obs = tmax(D); rng = np.random.default_rng(0)
null = np.array([tmax(rng.choice([-1, 1], n)).max() for _ in range(5000)])
print(f"   observed max t = {obs.max():.2f} at T{obs.argmax()+1}; adjusted p = {(null >= obs.max()).mean():.3f}")

print("\nD. Stability: signed mean (pips) and net of a 3-pip round-trip cost")
for lab, m in {"2010-2017": d.date < "2018-01-01", "2018-2026": d.date >= "2018-01-01", "excluding 2020": d.date.dt.year != 2020}.items():
    for h in (10, 15, 20):
        v = (d.D * d[f"POST_{h}"])[m].dropna()
        print(f"   {lab:<15} T{h:<3} n={len(v):>3} mean={v.mean():+.2f}  t={stats.ttest_1samp(v, 0).statistic:+.2f}  net@3pips={v.mean()-3:+.2f}")
print("\nE. Signed T15 mean by year (pips, n)")
v = (d.D * d["POST_15"]); g = pd.DataFrame({"y": d.date.dt.year, "v": v}).dropna().groupby("y").v.agg(["mean", "count"])
print(g.round(1).T.to_string())

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.6))
    allh = list(range(1, 61)); mu, se = [], []
    for h in allh:
        v = (d.D * d[f"POST_{h}"]).dropna(); mu.append(v.mean()); se.append(v.std(ddof=1) / np.sqrt(len(v)))
    mu, se = np.array(mu), np.array(se)
    ax.plot(allh, mu, lw=1.6); ax.fill_between(allh, mu - 1.96 * se, mu + 1.96 * se, alpha=.2)
    ax.axhline(0, c="k", lw=.6); ax.axhline(3, c="r", ls="--", lw=.8, label="3-pip round-trip cost")
    ax.set_xlabel("exit minute after the 08:30 candle close"); ax.set_ylabel("mean signed P&L (pips)")
    ax.set_title("Post-announcement drift in the NFP-implied direction (n=192)\n95% CI is pointwise, not adjusted for 60 horizons", fontsize=9)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__), "..", "results", "02_drift_by_horizon.png"), dpi=140)
except Exception as e:
    print("figure skipped:", e)
