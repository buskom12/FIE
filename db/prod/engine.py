from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine


def _sqlite_columns(engine, table: str) -> set[str]:
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(f"PRAGMA table_info({table});").fetchall()
    # row: (cid, name, type, notnull, dflt_value, pk)
    return {r[1] for r in rows}


def _sqlite_add_column(engine, table: str, column: str, col_type: str) -> None:
    with engine.connect() as conn:
        conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {col_type};")
        conn.commit()


def _default_db_path() -> str:
    root = Path(__file__).resolve().parents[2]  # .../FIE
    return str(root / "db" / "fie_prod.sqlite")


@lru_cache(maxsize=1)
def get_engine():
    db_path = os.environ.get("FIE_PROD_DB_PATH", "").strip() or _default_db_path()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", echo=False)


def init_db() -> None:
    engine = get_engine()
    SQLModel.metadata.create_all(engine)

    # Minimal, safe SQLite migrations (add-only columns)
    try:
        cols = _sqlite_columns(engine, "trades")
        add = []
        if "edge" not in cols:
            add.append(("edge", "REAL"))
        if "edge_score" not in cols:
            add.append(("edge_score", "REAL"))
        if "size" not in cols:
            add.append(("size", "REAL"))
        if "variance" not in cols:
            add.append(("variance", "REAL"))
        if "kelly_fraction" not in cols:
            add.append(("kelly_fraction", "REAL"))
        if "regime_key" not in cols:
            add.append(("regime_key", "TEXT"))
        if "p_model" not in cols:
            add.append(("p_model", "REAL"))
        if "p_market" not in cols:
            add.append(("p_market", "REAL"))
        if "edge_real" not in cols:
            add.append(("edge_real", "REAL"))
        # Diagnostic columns (add-only)
        if "funding_stress_48" not in cols:
            add.append(("funding_stress_48", "REAL"))
        if "liq_spike" not in cols:
            add.append(("liq_spike", "REAL"))
        if "scen_breakout_suspicious" not in cols:
            add.append(("scen_breakout_suspicious", "REAL"))
        if "momentum_up" not in cols:
            add.append(("momentum_up", "REAL"))
        if "oi_strength" not in cols:
            add.append(("oi_strength", "REAL"))
        # Allocation context (add-only)
        if "hour_utc" not in cols:
            add.append(("hour_utc", "INTEGER"))
        if "p_bucket" not in cols:
            add.append(("p_bucket", "REAL"))
        if "hour_weight" not in cols:
            add.append(("hour_weight", "REAL"))
        if "hour_n" not in cols:
            add.append(("hour_n", "INTEGER"))
        if "is_prior" not in cols:
            add.append(("is_prior", "INTEGER"))  # sqlite boolean-like
        for c, t in add:
            _sqlite_add_column(engine, "trades", c, t)
    except Exception:
        # If DB/table doesn't exist yet or non-sqlite engine, create_all is enough
        pass

    try:
        rcols = _sqlite_columns(engine, "regime_stats")
        if "last_update_ts" not in rcols:
            _sqlite_add_column(engine, "regime_stats", "last_update_ts", "REAL")
    except Exception:
        pass


def get_session() -> Session:
    engine = get_engine()
    return Session(engine)

