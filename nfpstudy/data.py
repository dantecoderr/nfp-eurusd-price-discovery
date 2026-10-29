"""Load the derived event dataset (one row per NFP release)."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "..", "data", "master_event_dataset.csv")
Z = ["NFP_Z_HIST", "AHE_Z_HIST", "U3_Z_HIST"]
HORIZONS = [5, 10, 15, 30, 45, 60]


def load(path=None):
    d = pd.read_csv(path or DEFAULT, parse_dates=["date"]).copy()
    d["R1"] = d["First_Minute_Return"]          # pips, 08:30 open -> 08:30 close
    # POST_h = (close of minute h after the 08:30 candle) - (close of 08:30 candle), pips
    for h in HORIZONS:
        d[f"R{h}"] = d[f"POST_{h}"]
    d["beat"] = d["NFP_Surprise"] > 0
    d["miss"] = d["NFP_Surprise"] < 0
    d["bull3"] = d["All_3_USD_Bullish"].astype(bool)
    d["bear3"] = d["All_3_USD_Bearish"].astype(bool)
    d["D"] = -np.sign(d["NFP_Surprise"])         # +1 long EUR/USD (miss), -1 short (beat)
    return d.copy()


def reg_sample(d):
    """Events with a defined historical Z-score (first 24 releases are warm-up)."""
    return d.dropna(subset=Z).copy()
