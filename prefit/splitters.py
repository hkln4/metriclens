"""Cross-validation splitters prefit builds itself.

DuplicateAwareKFold is a repeated (stratified) k-fold whose unit is a set of
identical rows rather than a row: exact duplicates (features + target) always
fall in the same fold, so a copy never validates its twin. It hashes the rows
itself, so no `groups=` is needed, and each repeat reshuffles with a new draw
from one seeded generator. That keeps the repeated k-fold protocol (and the
corrected resampled t-test in verify.py) valid, which a single run of
StratifiedGroupKFold would not.

It follows the scikit-learn splitter protocol (`split`, `get_n_splits`), so it
works with `cross_validate`, `cross_val_score` and `GridSearchCV`; numpy and
pandas are all it needs.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd


def duplicate_units(X: Any, y: Any = None) -> np.ndarray:
    """Unit id per row: rows with identical values in X (and y) share one id."""
    frame = X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    frame = frame.copy()
    frame.columns = range(frame.shape[1])
    if y is not None:
        frame[frame.shape[1]] = np.asarray(y).ravel()
    codes, _ = pd.factorize(pd.util.hash_pandas_object(frame, index=False).to_numpy())
    return codes


class DuplicateAwareKFold:
    """Repeated k-fold that keeps exact duplicate rows in one fold.

    n_splits      folds per repeat
    n_repeats     repeats, each with a fresh shuffle
    stratify      balance class ratios across folds (classification); a unit has
                  one class, since identical rows share the target
    random_state  seed of the generator all repeats draw from

    Within each class (or over all rows), units are shuffled, then placed
    largest-first into the fold with the fewest rows of that class, so fold
    sizes and class ratios stay close to plain (stratified) k-fold.
    """

    def __init__(self, n_splits: int = 5, n_repeats: int = 1, stratify: bool = False,
                 random_state: int | None = 0) -> None:
        if int(n_splits) < 2:
            raise ValueError("n_splits must be at least 2.")
        if int(n_repeats) < 1:
            raise ValueError("n_repeats must be at least 1.")
        self.n_splits = int(n_splits)
        self.n_repeats = int(n_repeats)
        self.stratify = bool(stratify)
        self.random_state = random_state

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits * self.n_repeats

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train, test) row positions. `groups` is accepted for API
        compatibility and ignored: the units come from the row hashes."""
        units = duplicate_units(X, y)
        n_units = int(units.max()) + 1 if len(units) else 0
        if n_units < self.n_splits:
            raise ValueError(f"Cannot make {self.n_splits} folds from {n_units} distinct rows.")
        if self.stratify and y is None:
            raise ValueError("stratify=True needs y.")
        sizes = np.bincount(units, minlength=n_units)
        first = np.unique(units, return_index=True)[1]
        if self.stratify:
            labels, _ = pd.factorize(pd.Series(np.asarray(y).ravel()[first]), use_na_sentinel=False)
        else:
            labels = np.zeros(n_units, dtype=int)
        rng = np.random.default_rng(self.random_state)
        all_rows = np.arange(len(units))
        for _ in range(self.n_repeats):
            fold_of_unit = self._assign(sizes, labels, rng)
            fold_of_row = fold_of_unit[units]
            for k in range(self.n_splits):
                test = fold_of_row == k
                yield all_rows[~test], all_rows[test]

    def _assign(self, sizes: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        fold = np.empty(len(sizes), dtype=int)
        total = np.zeros(self.n_splits, dtype=int)
        for lab in np.unique(labels):
            members = np.flatnonzero(labels == lab)
            rng.shuffle(members)
            members = members[np.argsort(-sizes[members], kind="stable")]  # largest first; ties stay shuffled
            in_class = np.zeros(self.n_splits, dtype=int)
            multi = members[sizes[members] > 1]
            for u in multi:
                k = int(np.lexsort((total, in_class))[0])  # fewest rows of this class, then fewest overall
                fold[u] = k
                in_class[k] += sizes[u]
                total[k] += sizes[u]
            singles = members[len(multi):]
            per_fold = _water_fill(in_class, total, len(singles))
            fold[singles] = np.repeat(np.arange(self.n_splits), per_fold)
            total += per_fold
        return fold

    def __repr__(self) -> str:
        return (f"DuplicateAwareKFold(n_splits={self.n_splits}, n_repeats={self.n_repeats}, "
                f"stratify={self.stratify}, random_state={self.random_state})")


def _water_fill(counts: np.ndarray, total: np.ndarray, m: int) -> np.ndarray:
    """How many of `m` single rows each fold gets so the emptiest folds fill up
    first (ties: fewest rows overall) — the greedy result, without a loop over rows."""
    order = np.lexsort((total, counts))
    s = counts[order].astype(np.int64)
    k = len(s)
    give = np.zeros(k, dtype=np.int64)
    if m <= 0:
        return give
    for j in range(1, k + 1):
        cap = j * (s[j] if j < k else np.iinfo(np.int64).max // (2 * k)) - s[:j].sum()
        if cap >= m:
            level, extra = divmod(int(s[:j].sum()) + m, j)
            target = np.full(j, level)
            target[:extra] += 1
            give[order[:j]] = target - s[:j]
            break
    return give
