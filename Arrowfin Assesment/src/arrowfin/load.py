"""Load the CSVs with correct types and validate the schema.

Rule: if the schema is broken we fail loudly (assert).
If a value is valid but odd, it is not fixed here: it is reported in quality.py.
"""
from pathlib import Path

import pandas as pd

from .config import DATASET_DIR

TIMESTAMP_COLS = {
    "brokers": ["created_at"],
    "traders": ["created_at"],
    "accounts": ["created_at"],
    "market_prices": ["as_of"],
    "fills": ["filled_at"],
}

EXPECTED_COLUMNS = {
    "brokers": {"id", "name", "type", "white_label_name", "created_at"},
    "traders": {"id", "broker_id", "first_name", "last_name", "email", "phone", "kyc_status", "created_at"},
    "accounts": {"id", "trader_id", "account_number", "account_type", "balance", "buying_power",
                 "max_position_size", "status", "created_at"},
    "instruments": {"symbol", "description", "exchange", "point_value_usd", "tick_size", "tick_value_usd",
                    "initial_margin_usd", "maintenance_margin_usd"},
    "market_prices": {"symbol", "mark_price", "as_of"},
    "fills": {"id", "account_id", "instrument_symbol", "side", "quantity", "price", "filled_at", "order_id",
              "liquidity", "commission_usd"},
}


def load_all(dataset_dir: Path = DATASET_DIR) -> dict[str, pd.DataFrame]:
    """Read the 6 CSVs, parse timestamps as UTC and validate the schema."""
    tables = {}
    for name, expected in EXPECTED_COLUMNS.items():
        df = pd.read_csv(Path(dataset_dir) / f"{name}.csv")
        missing = expected - set(df.columns)
        assert not missing, f"{name}.csv: missing columns {missing}"
        for col in TIMESTAMP_COLS.get(name, []):
            # utc=True: timestamps are ISO 8601 with 'Z'. We never work with naive datetimes.
            df[col] = pd.to_datetime(df[col], utc=True, errors="raise")
        tables[name] = df
    validate_schema(tables)
    return tables


def validate_schema(t: dict[str, pd.DataFrame]) -> None:
    """Basic integrity: unique keys, foreign keys that exist, values within their domain."""
    f, a, tr, ins = t["fills"], t["accounts"], t["traders"], t["instruments"]

    for name, key in [("fills", "id"), ("accounts", "id"), ("traders", "id"), ("instruments", "symbol"),
                      ("market_prices", "symbol"), ("brokers", "id")]:
        assert t[name][key].is_unique, f"{name}.{key} has duplicates"

    assert f.notna().all().all(), "fills has nulls"
    assert f["account_id"].isin(a["id"]).all(), "fill with unknown account_id"
    assert f["instrument_symbol"].isin(ins["symbol"]).all(), "fill with unknown symbol"
    assert a["trader_id"].isin(tr["id"]).all(), "account with unknown trader_id"
    assert tr["broker_id"].isin(t["brokers"]["id"]).all(), "trader with unknown broker_id"
    assert set(ins["symbol"]) == set(t["market_prices"]["symbol"]), "missing mark for an instrument"

    assert f["side"].isin(["BUY", "SELL"]).all()
    assert f["liquidity"].isin(["maker", "taker"]).all()
    assert (f["quantity"] > 0).all() and (f["price"] > 0).all() and (f["commission_usd"] >= 0).all()

    # One order = one account, one symbol, one side. Otherwise grouping by order_id would be wrong.
    per_order = f.groupby("order_id")[["account_id", "instrument_symbol", "side"]].nunique()
    assert (per_order == 1).all().all(), "an order_id mixes accounts, symbols or sides"
