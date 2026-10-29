"""Robustness of the first-minute and alignment results (all inputs from the event CSV)."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from scipy import stats
from nfpstudy.data import load, reg_sample, Z, HORIZONS
from nfpstudy.stats import fit, clopper_pearson, binom_p

d = load(); r = reg_sample(d)
k = 3
print("A. R2 by horizon, with adjusted R2 (regression sample N=%d)" % len(r))
print("   Expected R2 of a pure-noise model with 3 regressors = k/(n-1) = %.4f\n" % (k / (len(r) - 1)))
print(" horizon   n    R2    adjR2  joint p")
for h in [1] + HORIZONS:
    y = "R1" if h == 1 else f"R{h}"; g = fit(r.dropna(subset=[y]), y, Z)
    print(f"   T{h:<3} {g['n']:>4} {g['r2']:6.4f} {g['adj']:+7.4f}  {g['joint_p']:.3f}")

print("\nB. Directional accuracy with exact (Clopper-Pearson) 95% CI; p = two-sided binomial vs 50%")
groups = {"NFP beat (expect down)": (d.beat, -1), "beat + all-3 bullish": (d.beat & d.bull3, -1),
          "NFP miss (expect up)": (d.miss, 1), "miss + all-3 bearish": (d.miss & d.bear3, 1)}
for g, (mask, sgn) in groups.items():
    print(g)
    for h in [1] + HORIZONS:
        y = "R1" if h == 1 else f"R{h}"; v = d.loc[mask, y].dropna(); kk = int((sgn * v > 0).sum()); n = len(v)
        lo, hi = clopper_pearson(kk, n)
        print(f"   T{h:<3} {kk:>3}/{n:<3} = {kk/n:5.1%}  CI [{lo:5.1%}, {hi:5.1%}]  p={binom_p(kk, n):.3f}")

print("\nC. Is the 'full alignment' effect just a bigger NFP surprise?")
b = d[d.beat].copy(); b["full"] = b.bull3.astype(int)
print("   NFP surprise (k jobs), beats: median fully aligned %.0f vs not %.0f ; mean %.0f vs %.0f" % (
    b[b.full == 1].NFP_Surprise.median(), b[b.full == 0].NFP_Surprise.median(),
    b[b.full == 1].NFP_Surprise.mean(), b[b.full == 0].NFP_Surprise.mean()))
bb = b.dropna(subset=Z)
print("   Beats with Z-scores: n=%d, fully aligned=%d. Dependent variable: first-minute return (pips)" % (len(bb), bb.full.sum()))
for cols in (["full"], ["NFP_Z_HIST"], ["NFP_Z_HIST", "full"], Z, Z + ["full"]):
    g = fit(bb, "R1", cols)
    print(f"   {'+'.join(cols):<48} R2={g['r2']:.3f}  coefs={np.round(g['b'][1:], 2)}  HC3 p={np.round(g['p'][1:], 3)}")
print("   -> the full-alignment dummy stays large and significant after controlling for NFP size and the other surprises.")
print("   Caveat: 13-14 events, and the 'fully aligned' group was defined after looking at the data.")

print("\nD. Sensitivity of the first-minute regression")
def row(lab, df, cols):
    g = fit(df, "R1", cols); print(f"   {lab:<40} n={g['n']:>3} R2={g['r2']:.3f} beta={np.round(g['b'][1:], 1)} p={np.round(g['p'][1:], 3)}")
row("thesis spec (Z clipped at +/-4)", r, Z)
row("excluding all 2020 releases", r[r.date.dt.year != 2020], Z)
row("excluding Apr-Jun 2020 only", r[~r.date.between("2020-04-01", "2020-06-30")], Z)
row("Z clipped at +/-3 (notebook NFP_Z cols)", r, ["NFP_Z", "AHE_Z", "U3_Z"])
raw = d.copy()
for c, s in (("NFP_Surprise", "nz"), ("AHE_Surprise", "az"), ("U3_USD_Surprise", "uz")):
    x = raw[c]; raw[s] = (x - x.expanding(24).mean().shift(1)) / x.expanding(24).std().shift(1)
row("Z unclipped (raw expanding z)", raw.dropna(subset=["nz", "az", "uz"]), ["nz", "az", "uz"])
def mad_z(x, w=40):
    med = x.rolling(w, min_periods=24).median().shift(1)
    mad = x.rolling(w, min_periods=24).apply(lambda a: np.median(np.abs(a - np.median(a))), raw=True).shift(1) * 1.4826
    return ((x - med) / mad).clip(-4, 4)
q = d.copy(); q["nz"], q["az"], q["uz"] = mad_z(q.NFP_Surprise), mad_z(q.AHE_Surprise), mad_z(q.U3_USD_Surprise)
row("rolling-40 median/MAD z (clip +/-4)", q.dropna(subset=["nz", "az", "uz"]), ["nz", "az", "uz"])
row("raw surprises (k jobs, %, %), all 194", d, ["NFP_Surprise", "AHE_Surprise", "U3_USD_Surprise"])
d["nfp_sign"] = np.sign(d.NFP_Surprise); d["ahe_sign"] = np.sign(d.AHE_Surprise); d["u3_sign"] = np.sign(d.U3_USD_Surprise)
row("NFP sign only (192 events)", d[d.nfp_sign != 0], ["nfp_sign"])
row("three signs", d[(d.nfp_sign != 0)], ["nfp_sign", "ahe_sign", "u3_sign"])
rs = r.copy(); rs["nfp_sign"] = np.sign(rs.NFP_Surprise); rs = rs[rs.nfp_sign != 0]
row("NFP sign only, same sample as thesis", rs, ["nfp_sign"])
print("   -> sign-based models (R2 0.24-0.31) explain as much or more than the clipped Z-score model (0.21); the unclipped Z-score collapses to 0.06 because three 2020 releases dominate. Magnitude information adds little; the result is mostly a sign effect.")
