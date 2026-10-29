# Data

`master_event_dataset.csv` (194 rows, one per NFP release, Jan 2010 - Jul 2026) is a **derived** event table:
actual/forecast values for NFP, average hourly earnings and unemployment (collected from investing.com),
the surprise and Z-score columns computed in the notebook, and EUR/USD paths in **pips** around the 08:30 ET
release (first-minute OHLC plus `POST_1..POST_60`, relative to the close of the 08:30 candle; built from
1-minute histdata.com bars).

Not included: the raw 1-minute EUR/USD price file (`filtered_eurusd_nfp_events.csv`, about 202k rows) that the
original Colab notebook reads from Google Drive. Download 1-minute EUR/USD from histdata.com yourself if you
want to rebuild the event table; check the provider's terms before redistributing any of it.

Before publishing: check the investing.com and histdata terms for redistributing even derived tables. If you
are not comfortable, delete this file and keep it local (the scripts take a path via `nfpstudy.data.load(path)`).
