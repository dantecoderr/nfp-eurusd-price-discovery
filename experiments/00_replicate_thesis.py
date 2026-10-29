"""Re-derive the thesis' headline numbers from data/master_event_dataset.csv
and print them next to the values reported in the thesis (Chapter 4)."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from scipy import stats
from nfpstudy.data import load, reg_sample, Z, HORIZONS
from nfpstudy.stats import fit

d = load(); r = reg_sample(d)
rows = []
def chk(name, mine, thesis, tol):
    rows.append((name, mine, thesis, "OK" if abs(mine - thesis) <= tol else "DIFF"))

# 4.1 sample
chk("matched releases", len(d), 194, 0)
chk("NFP beats", d.beat.sum(), 106, 0); chk("NFP misses", d.miss.sum(), 86, 0)
chk("NFP in line", (d.NFP_Surprise == 0).sum(), 2, 0)
chk("beat + all-3 bullish", (d.beat & d.bull3).sum(), 14, 0)
chk("miss + all-3 bearish", (d.miss & d.bear3).sum(), 10, 0)
# 4.2 / 4.6 first minute
b, m = d[d.beat], d[d.miss]; fb, fm = d[d.beat & d.bull3], d[d.miss & d.bear3]
chk("beat mean R1 (pips)", b.R1.mean(), -14.39, 0.01); chk("beat median R1", b.R1.median(), -11.90, 0.01)
chk("beat % down", 100 * (b.R1 < 0).mean(), 66.04, 0.01)
chk("miss mean R1", m.R1.mean(), 14.85, 0.01); chk("miss median R1", m.R1.median(), 13.65, 0.01)
chk("miss % up", 100 * (m.R1 > 0).mean(), 69.77, 0.01)
chk("full-bull mean R1", fb.R1.mean(), -34.52, 0.01); chk("full-bull median", fb.R1.median(), -33.30, 0.01)
chk("full-bull % down", 100 * (fb.R1 < 0).mean(), 92.86, 0.01)
chk("full-bear mean R1", fm.R1.mean(), 20.38, 0.01); chk("full-bear % up", 100 * (fm.R1 > 0).mean(), 70.0, 0.01)
# 4.7 alignment test
nf = b[~b.bull3]
chk("non-aligned beat mean", nf.R1.mean(), -11.33, 0.01)
w = stats.ttest_ind(fb.R1, nf.R1, equal_var=False)
chk("Welch t", w.statistic, -3.2085, 0.001); chk("Welch p", w.pvalue, 0.0053, 0.0001)
chk("Mann-Whitney p", stats.mannwhitneyu(fb.R1, nf.R1).pvalue, 0.0030, 0.0001)
# 4.9 OLS
f = fit(r, "R1", Z)
chk("OLS R2", f["r2"], 0.2108, 0.0001); chk("beta NFP", f["b"][1], -13.59, 0.01)
chk("HC3 se NFP", f["se"][1], 3.31, 0.01); chk("beta AHE", f["b"][2], -8.64, 0.01)
chk("p AHE", f["p"][2], 0.0652, 0.0001); chk("beta U3", f["b"][3], 2.32, 0.01); chk("p U3", f["p"][3], 0.5945, 0.0001)
# 4.10 decay
for h, tr2, tp in zip(HORIZONS, [.0157, .0223, .0091, .0153, .0094, .0164], [.581, .283, .710, .533, .769, .449]):
    g = fit(r.dropna(subset=[f"R{h}"]), f"R{h}", Z)
    chk(f"R2 T{h}", g["r2"], tr2, 0.0001); chk(f"joint p T{h}", g["joint_p"], tp, 0.001)
# 4.11 Table 4.2 directional accuracy (a few cells)
def hit(df, sgn, h): v = df[f"R{h}"].dropna(); return 100 * (sgn * v > 0).mean()
for h, t in zip([5, 10, 15], [50.00, 59.43, 65.09]): chk(f"hit beat T{h}", hit(b, -1, h), t, 0.01)
for h, t in zip([5, 10], [35.71, 28.57]): chk(f"hit full-bull T{h}", hit(fb, -1, h), t, 0.01)
# 4.16 continuation
for h, tg, tp in zip([5, 10, 15, 30, 60], [.0559, -.0185, .0651, .0220, -.0580], [.281, .748, .327, .805, .575]):
    rr = r.dropna(subset=[f"R{h}"]); g = fit(rr, f"R{h}", ["R1"] + Z)
    # The control set of Eq. 3.40 is not documented in the notebook; with Z-score controls the
    # coefficients are close but not identical. Conclusion (all insignificant) is the same.
    rows.append((f"gamma T{h} (approx.)", g["b"][1], tg, "QUAL-OK" if g["p"][1] > 0.05 else "DIFF"))
    rows.append((f"gamma p T{h} (approx.)", g["p"][1], tp, "QUAL-OK" if g["p"][1] > 0.05 else "DIFF"))
out = pd.DataFrame(rows, columns=["quantity", "re-derived", "thesis", "status"])
print(out.round(4).to_string(index=False))
print("\nexact matches:", (out.status == "OK").sum(), "| qualitative matches:", (out.status == "QUAL-OK").sum(), "| differences:", (out.status == "DIFF").sum(), "| total:", len(out))
print("\nNOT STATED IN THESIS: regression sample N =", len(r),
      "(first 24 releases are warm-up for the expanding Z-score); Z-scores are clipped at +/-4:")
raw = d.NFP_Surprise
z = (raw - raw.expanding(24).mean().shift(1)) / raw.expanding(24).std().shift(1)
print(pd.DataFrame({"date": d.date, "NFP_raw_z": z.round(2), "NFP_Z_HIST_used": d.NFP_Z_HIST.round(2)})
      [z.abs() > 4].to_string(index=False))
