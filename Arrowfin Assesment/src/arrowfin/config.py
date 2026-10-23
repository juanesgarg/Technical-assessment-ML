"""Dataset constants. Everything 'magic' lives here, not scattered through the code."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
OUTPUTS_DIR = ROOT / "outputs"

# Dataset README: everything is as of this instant. We treat it as "now".
CUT = pd.Timestamp("2026-08-25T14:30:00Z")

# CME Globex: the session opens at 17:00 Chicago time and closes at 16:00 the next day.
# A session is named by the day it CLOSES (the trade date).
EXCHANGE_TZ = "America/Chicago"
SESSION_OPEN_HOUR = 17
