# Macroeconomic surprises and EUR/USD price discovery around US Non-Farm Payrolls

Code and reproducible results for my MSc thesis *A Study in Programmable Trading Signals: Macroeconomic
Information, High-Frequency Price Discovery and Trading Opportunities in EUR/USD* (University of Milan, 2025/26).

**Question.** NFP, average hourly earnings (AHE) and the unemployment rate are released together at 08:30 ET.
How much of the EUR/USD reaction is explained by the surprises, how fast does it disappear, and can a trader
who reacts after the first minute capture anything after costs?

**Sample.** 194 releases, Jan 2010 - Jul 2026, 1-minute EUR/USD. Surprise = actual - consensus
(unemployment sign-flipped so that positive always means USD-bullish).

*Academic research code, not investment advice.*

## Findings (all re-derived from `data/master_event_dataset.csv`)

| | Result |
|---|---|
| First minute | NFP beats: mean -14.4 pips, 66% down. Misses: +14.8 pips, 70% up. Beat + all three USD-bullish: -34.5 pips, 13/14 down. |
| Regression (HC3) | First-minute R² = 0.211 (N=170); NFP -13.6 pips per SD (p<0.001), AHE -8.6 (p=0.065), unemployment n.s. |
| After the first minute | R² 0.9-2.2%, adjusted R² between -0.9% and +0.5%, joint p 0.28-0.77 at every horizon T5-T60. A pure-noise model with 3 regressors has an expected R² of 1.8% here, so this is indistinguishable from zero. |
| Trading | The "fully aligned" rule (23 trades) shows +5.3 pips net of a 3-pip cost, but the median is +0.8, the bootstrap CI is [-11.5, +23.5], and expanding walk-forward is negative. |

## What this repo adds beyond the thesis text

Run `./run_all.sh`; outputs are in `results/`.

1. **Replication (`00`).** 45 of 55 reported numbers reproduce exactly (R², coefficients, HC3 errors, Welch and
   Mann-Whitney tests, directional accuracy, decay R² and joint p-values). The 10 continuation-regression
   coefficients (Eq. 3.40) reproduce only qualitatively because the control set is not documented in the notebook.
2. **Uncertainty on the headline hit rates (`01`).** 13/14 = 92.9%, exact 95% CI [66%, 99.8%]. The T5 and T10
   aligned-group hit rates (5/14, 4/14) are not distinguishable from 50% (p = 0.42, 0.18).
3. **Full alignment is not just a bigger NFP surprise (`01`).** Surprises are not larger in the aligned group
   (median 73k vs 55k jobs). Among the 97 NFP beats with Z-scores, the full-alignment dummy is -21.7 pips
   (HC3 p = 0.011) after controlling for all three Z-scores. Caveat: 13 events, and the subgroup was defined
   after looking at the data.
4. **Sensitivity to 2020 and to scaling (`01`).** Excluding 2020: R² 0.296 and all coefficients larger
   (AHE p<0.001). Rolling median/MAD scaling: R² 0.291. Sign-only specifications: R² 0.24-0.31. Unclipped
   expanding Z-scores: R² 0.06, because three 2020 releases dominate. The effect is mostly about the *sign* of
   the surprise, not its size.
5. **Whole-sample post-minute drift (`02`).** Trading in the NFP-implied direction from the close of the first
   minute earns +3.4 pips at T15 (t = 2.0) and +4.3 at T20 (t = 2.4), almost all from NFP beats (+5.7 pips at T15,
   t = 2.8; misses +0.5). After adjusting for choosing the best of 60 horizons the p-value is 0.063. Net of a
   3-pip round-trip it is about +0.4 pips at T15 and it is negative in 2018-2026 at T10-T15. This is weak,
   marginal continuation that does not survive costs, not strictly zero predictability.
6. **Selection in the trading result (`03`).** The best cell of the 4x4 entry/exit grid has t = 0.92 and a
   grid-adjusted p = 0.42. The top three trades contribute more than 100% of total P&L, and the result comes
   from the 10 bear-aligned longs (+19.0 pips average) while the 13 bull-aligned shorts average +0.1.
   Expanding walk-forward with exits up to T60: -29.9 pips over 13 trades.

## Notes on the thesis text (found while re-deriving)

The thesis was submitted as written and is not changed here. These are differences between its text and what the code does, recorded so the results can be read correctly:

* The regression runs on **170 events, not 194**: the first 24 releases are warm-up for the expanding Z-score.
* Z-scores are **clipped at +/-4**. Three 2020 releases (Apr-Jun) have raw expanding Z-scores of -9.5, +17.7 and +67.7.
* The thesis says the trade enters at "the close of the first minute". The notebook's entry `C1` is the close of the
  **second** candle (08:31), a fill roughly two minutes after the release, using candle-close mid prices with no
  spread. Entry at the first-minute close (`C0`) gives +4.0 gross instead of +8.3.
* Table 4.2 shows NFP-beat directional accuracy of 59% at T10 and 65% at T15 (binomial p = 0.064 and 0.002). The
  statement that accuracy "converges toward 50%" holds for misses and for the aligned groups, not for plain beats.
* The notebook contains no HC3 / Welch / Mann-Whitney code (those results were computed elsewhere); `nfpstudy/stats.py`
  is a from-scratch implementation checked against the reported values.

## Layout

```
data/master_event_dataset.csv      derived event table (see data/README.md)
nfpstudy/                          HC3 OLS, exact binomial CIs, data loader
experiments/00_replicate_thesis.py thesis numbers vs re-derived numbers
experiments/01_robustness.py       adjusted R², CIs, alignment control, 2020 and scaling sensitivity
experiments/02_post_minute_drift.py whole-sample drift, multiple-testing check, sub-periods, figure
experiments/03_trading_replication.py 23-trade result, selection bias, walk-forward
notebooks/original_colab_exploration.ipynb  the original exploration notebook, outputs stripped (needs Colab + raw price data)
results/                           text, CSV and PNG outputs of the scripts
```

```bash
pip install -r requirements.txt
./run_all.sh
```

## Limitations

* 194 events (170 in the regressions); the fully aligned groups have 14 and 10.
* 1-minute OHLC: nothing is known about the path inside the announcement minute, and mid prices ignore spread and
  latency. A 3-pip round-trip cost is an assumption, not a measurement.
* Consensus forecasts come from a single public aggregator; revisions to previous months are not modelled.
* No placebo test (for example the same regression on 08:29 or on non-NFP days) is included because the raw
  minute data is not in the event table.
* The CSV only contains exits up to T60, so the original grid's T75/T90 exits are not replicated.
* The original exploration notebook searched many specifications; treat any full-sample result as exploratory.

## License

MIT, see `LICENSE`. The thesis text itself is not part of this repository.
