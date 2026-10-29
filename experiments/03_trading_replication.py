"""Replicate the thesis' 'fully aligned' entry/exit result and test how much of it is selection.

Notation (as in the original notebook): C_k = close of the k-th minute after the 08:30 candle, C_0 = close of
the 08:30 candle. In the CSV, POST_k = (C_k - C_0) in pips. A trade 'entry e, exit x' earns
D * (POST_x - POST_e), D = -1 (short EUR/USD) when all three surprises are USD-bullish, +1 when all bearish.

NOTE: entry e=1 means the CLOSE OF THE SECOND CANDLE (08:31), i.e. a fill about two minutes after the release,
not 'the close of the first minute' as the thesis text (Sec. 4.18) says. The CSV only holds exits up to T60, so
exits 75/90 from the original grid are not available here."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from scipy import stats
from nfpstudy.data import load

d = load(); al = d.NFP_Sign + d.AHE_Sign + d.U3_USD_Sign
fa = d[al.abs() == 3].copy().sort_values("date"); fa["Dm"] = np.where(al[fa.index] == 3, -1, 1)
ENT, EXI, COST = [0, 1, 2, 5], [15, 30, 45, 60], 3.0
P = lambda df, e: 0.0 if e == 0 else df[f"POST_{e}"]
print("Fully aligned releases:", len(fa), "(bull %d, bear %d)" % ((fa.Dm == -1).sum(), (fa.Dm == 1).sum()))
print("\nA. Gross P&L grid (pips): mean / median / n  [cost 0]")
res = {}
for e in ENT:
    for x in EXI:
        v = (fa.Dm * (fa[f"POST_{x}"] - P(fa, e))).dropna(); res[(e, x)] = v
        print(f"   entry T{e} exit T{x}: mean {v.mean():6.2f}  median {v.median():6.2f}  n {len(v)}  t {stats.ttest_1samp(v, 0).statistic:5.2f}")
v = res[(1, 60)]
print(f"\nB. Thesis 'descriptive' trade (entry C1, exit C60), cost 3 pips: mean {v.mean()-COST:.2f}  median {v.median()-COST:.2f}  n {len(v)}  win {((v-COST)>0).mean():.1%}  total {(v-COST).sum():.1f}")
bs = np.random.default_rng(1).choice((v - COST).values, (10000, len(v))).mean(1)
print(f"   bootstrap 95% CI of the net mean: [{np.percentile(bs, 2.5):.1f}, {np.percentile(bs, 97.5):.1f}] pips; share of total P&L from best 3 trades: {np.sort((v-COST).values)[-3:].sum()/(v-COST).sum():.0%}")
for c in (0, 1, 2, 3, 5): print(f"   cost {c}: mean {v.mean()-c:6.2f} median {v.median()-c:6.2f} total {(v-c).sum():7.1f}")

print("\nC. Selection bias: best cell of the 4x4 grid vs a sign-flip null (random direction per event)")
cells = [(e, x) for e in ENT for x in EXI]
Mx = np.column_stack([(fa[f"POST_{x}"] - P(fa, e)).values for e, x in cells]); Dm = fa.Dm.values
ok = ~np.isnan(Mx).any(1); Mx, Dm = Mx[ok], Dm[ok]; n = len(Dm)
tst = lambda Dv: (Dv[:, None] * Mx).mean(0) / ((Dv[:, None] * Mx).std(0, ddof=1) / np.sqrt(n))
obs = tst(Dm); rng = np.random.default_rng(0)
null = np.array([tst(rng.choice([-1, 1], n)).max() for _ in range(5000)])
print(f"   best cell: entry T{cells[obs.argmax()][0]} exit T{cells[obs.argmax()][1]}  t={obs.max():.2f}; grid-adjusted p = {(null >= obs.max()).mean():.3f}  (n={n})")
print("   By side for the same trade: bull-aligned shorts mean %.2f (n=%d), bear-aligned longs mean %.2f (n=%d)" % (
    v[fa.Dm == -1].mean(), (fa.Dm.loc[v.index] == -1).sum(), v[fa.Dm == 1].mean(), (fa.Dm.loc[v.index] == 1).sum()))

print("\nD. Expanding walk-forward parameter selection (needs >=10 prior aligned events; best mean net pnl at 3 pips)")
fa = fa.reset_index(drop=True); curve = []; picks = []
for i in range(10, len(fa)):
    hist = fa.iloc[:i]; best, bc = None, -1e9
    for e in ENT:
        for x in EXI:
            vv = (hist.Dm * (hist[f"POST_{x}"] - P(hist, e))).dropna() - COST
            if len(vv) >= 5 and vv.mean() > bc: bc, best = vv.mean(), (e, x)
    row = fa.iloc[i]; pnl = row.Dm * (row[f"POST_{best[1]}"] - (0 if best[0] == 0 else row[f"POST_{best[0]}"])) - COST
    if not np.isnan(pnl): curve.append((row.date.date(), best, round(pnl, 1)))
cw = pd.DataFrame(curve, columns=["date", "chosen (entry, exit)", "net pips"]); cw["cum"] = cw["net pips"].cumsum()
print(cw.to_string(index=False)); print("   total walk-forward net pips: %.1f over %d trades (thesis Fig 4.17 used exits up to T90; ends near -58)" % (cw["net pips"].sum(), len(cw)))
cw.to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "03_walk_forward_trades.csv"), index=False)
