"""TSV I/O. Safe against the two silent-corruption traps in this dataset:

1. pandas' default quote handling: a business name that begins with `"` (Source 2 has heavy
   punctuation noise) makes the C parser swallow following lines. We use QUOTE_NONE.
2. pandas' default NA handling: names/countries like "NA", "None", "null" become NaN.
   We use keep_default_na=False and dtype=str.

Every loader asserts that the number of parsed rows equals the number of data lines.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def _count_lines(path) -> int:
    n = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            n += block.count(b"\n")
    return n


def read_tsv(path, expected_cols=None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                     quoting=csv.QUOTE_NONE, encoding="utf-8", engine="c")
    if expected_cols is not None and list(df.columns) != list(expected_cols):
        raise ValueError(f"{path}: header {list(df.columns)} != {expected_cols}")
    # Row-count integrity check (header + rows; tolerate missing final newline)
    n_lines = _count_lines(path)
    if len(df) not in (n_lines - 1, n_lines):
        raise ValueError(f"{path}: parsed {len(df):,} rows but file has {n_lines:,} lines — "
                         "possible embedded newline/quote corruption")
    return df


def read_source(path) -> pd.DataFrame:
    df = read_tsv(path, SOURCE_COLS)
    if df["entity_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate entity_id values")
    return df


def read_ground_truth(path) -> Dict[str, List[str]]:
    df = read_tsv(path, ["source1_entity_id", "matched_entity_ids"])
    return {s: ([x for x in m.split(",") if x] if m else [])
            for s, m in zip(df["source1_entity_id"].values, df["matched_entity_ids"].values)}


def write_id_list_file(path, header_col: str, s1_ids: np.ndarray, groups: Dict[int, List[str]]):
    """Write one row per S1 id (in the given order). groups maps S1 row index -> list of ids."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header_col}\n")
        for k, sid in enumerate(s1_ids):
            ids = groups.get(k)
            f.write(f"{sid}\t{','.join(ids) if ids else ''}\n")
