"""Fingerprints of the DataFrame a report was built from.

A report never stores rows, so later steps that need data (`verify`, `holdout`)
take the DataFrame again. The fingerprint proves it is the same frame: five
SHA-256 digests (no data) are kept in the manifest, and any mismatch raises —
prefit never re-splits or silently continues on a different frame.

    column_schema_digest   column names and their order
    dtype_digest           dtypes, in column order
    index_digest           index values, in row order
    target_digest          target values, in row order
    content_digest         every cell, row-order sensitive

Per-row hashes are kept in memory only (never serialized) to say *how* a frame
differs: reordered rows, changed rows, or a different length.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

DIGESTS = ("column_schema_digest", "dtype_digest", "index_digest", "target_digest", "content_digest")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def row_hashes(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(df, index=False).to_numpy()


def fingerprint(df: pd.DataFrame, target: str) -> dict[str, Any]:
    rows = row_hashes(df)
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "column_schema_digest": _sha(json.dumps([str(c) for c in df.columns]).encode()),
        "dtype_digest": _sha(json.dumps([str(t) for t in df.dtypes]).encode()),
        "index_digest": _sha(pd.util.hash_pandas_object(df.index, index=False).to_numpy().tobytes()),
        "target_digest": _sha(pd.util.hash_pandas_object(df[target], index=False).to_numpy().tobytes()),
        "content_digest": _sha(rows.tobytes()),
    }


class FingerprintMismatch(ValueError):
    """The DataFrame given is not the one the report was built from."""


def check(df: Any, expected: dict[str, Any], target: str, what: str,
          expected_rows: np.ndarray | None = None) -> None:
    """Raise FingerprintMismatch, explaining the difference, unless `df` matches."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{what} expects a pandas DataFrame, got {type(df).__name__}.")
    if not expected:
        raise FingerprintMismatch(f"{what}: this report has no source fingerprint; re-run pf.analyze.")
    if target not in df.columns:
        raise FingerprintMismatch(f"{what}: target column '{target}' is missing — this is not the DataFrame "
                                  "passed to pf.analyze.")
    got = fingerprint(df, target)
    bad = [k for k in DIGESTS if got[k] != expected[k]]
    if not bad and got["n_rows"] == expected["n_rows"]:
        return
    reasons = []
    if got["n_rows"] != expected["n_rows"] or got["n_columns"] != expected["n_columns"]:
        reasons.append(f"shape is {got['n_rows']}x{got['n_columns']}, expected "
                       f"{expected['n_rows']}x{expected['n_columns']}")
    if "column_schema_digest" in bad:
        reasons.append("column names or their order differ")
    elif "dtype_digest" in bad:
        reasons.append("column dtypes differ")
    if "index_digest" in bad:
        reasons.append("the index differs")
    if "target_digest" in bad:
        reasons.append("target values differ")
    if "content_digest" in bad and expected_rows is not None and len(df) == len(expected_rows) \
            and "column_schema_digest" not in bad:
        mine = row_hashes(df)
        if np.array_equal(np.sort(mine), np.sort(expected_rows)):
            reasons.append("the same rows are present but in a different order")
        else:
            reasons.append(f"{int((mine != expected_rows).sum())} row(s) have different values")
    elif "content_digest" in bad:
        reasons.append("cell values differ")
    raise FingerprintMismatch(
        f"{what}: this is not the DataFrame passed to pf.analyze ({'; '.join(reasons)}). Nothing was re-split. "
        "Pass the exact same frame, or re-run pf.analyze on the new data.")
