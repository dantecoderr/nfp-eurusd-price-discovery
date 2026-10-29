#!/usr/bin/env bash
# Re-runs every experiment from data/master_event_dataset.csv (about 10 seconds).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
python experiments/00_replicate_thesis.py       > results/00_replication.txt
python experiments/01_robustness.py             > results/01_robustness.txt
python experiments/02_post_minute_drift.py      > results/02_post_minute_drift.txt
python experiments/03_trading_replication.py    > results/03_trading_replication.txt
echo "done - see results/"
