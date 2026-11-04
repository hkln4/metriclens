"""Statistics against independent references, and their calibration by simulation.

The decisions rest on these numbers, so "the direction looks right" is not
enough: each statistic is compared with a reference implementation (scipy,
pandas, a by-hand formula), and the tests that make claims are checked for
their error rates on data where the truth is known."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from prefit.profiling.missingness import _target_matrix, _target_relation
from prefit.profiling.stats import (
    _lookup_predict,
    bootstrap_ci,
    excess_kurtosis,
    iqr_outlier_fraction,
    skewness,
    wilson_ci,
)
from prefit.profiling.tabular import _spearman
from prefit.verify import classify_outcome, corrected_ttest, equivalence_test, holm

D = np.array([0.01, 0.03, -0.01, 0.02, 0.00, 0.04, 0.01, 0.02, 0.03, -0.02])


# --------------------------------------------------------------------------- #
# reference values
# --------------------------------------------------------------------------- #

def test_corrected_ttest_reduces_to_the_paired_t_test_without_correction():
    base = np.linspace(0.7, 0.8, 10)
    mean, ci, p = corrected_ttest(base, base + D, 0.0)
    ref = stats.ttest_rel(base + D, base)
    assert mean == pytest.approx(D.mean()) and p == pytest.approx(ref.pvalue, rel=1e-10)
    lo, hi = stats.t.interval(0.95, len(D) - 1, loc=D.mean(), scale=stats.sem(D))
    assert ci == pytest.approx((lo, hi), rel=1e-10)


def test_corrected_ttest_matches_the_nadeau_bengio_formula():
    _, ci, p = corrected_ttest(np.zeros(10), D, 0.25)
    se = np.sqrt((1 / 10 + 0.25) * np.var(D, ddof=1))       # Nadeau & Bengio (2003), eq. for the corrected variance
    assert p == pytest.approx(2 * stats.t.sf(abs(D.mean() / se), 9), rel=1e-10)
    assert ci[1] - ci[0] == pytest.approx(2 * stats.t.ppf(0.975, 9) * se, rel=1e-10)


def test_equivalence_test_is_two_one_sided_tests():
    margin = 0.03
    p = equivalence_test(np.zeros(10), D, 0.0, margin)
    lower = stats.ttest_1samp(D, -margin, alternative="greater").pvalue
    upper = stats.ttest_1samp(D, margin, alternative="less").pvalue
    assert p == pytest.approx(max(lower, upper), rel=1e-10)


def test_wilson_interval_matches_scipy():
    for k, n in [(0, 20), (3, 40), (20, 100), (97, 100), (500, 1000)]:
        ref = stats.binomtest(k, n).proportion_ci(method="wilson")
        assert wilson_ci(k, n) == pytest.approx((ref.low, ref.high), abs=1e-6)


def test_holm_matches_the_step_down_definition():
    p = [0.01, 0.04, 0.03, 0.005]
    order = np.argsort(p)
    ref = np.empty(4)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (4 - rank) * p[i]))
        ref[i] = running
    assert holm(p) == pytest.approx(list(ref))
    assert classify_outcome(0.02, 0.01, 0.005, 0.05) == "improved"
    assert classify_outcome(-0.02, 0.01, 0.005, 0.05) == "worse"
    assert classify_outcome(0.001, 0.01, 0.005, 0.05) == "negligible"


def test_row_wise_moments_match_pandas():
    rng = np.random.default_rng(1)
    m = rng.exponential(size=(6, 300))
    sk, ku, out = skewness(m), excess_kurtosis(m), iqr_outlier_fraction(m)
    for i in range(6):
        s = pd.Series(m[i])
        assert sk[i] == pytest.approx(s.skew()) and ku[i] == pytest.approx(s.kurt())
        assert out[i] == pytest.approx(iqr_outlier_fraction(m[i]))
    assert skewness(np.ones(10)) == 0.0 and excess_kurtosis(np.ones(10)) == 0.0


def test_vectorized_bootstrap_equals_the_loop():
    v = np.random.default_rng(2).lognormal(size=12_000)
    looped = bootstrap_ci(v, lambda a: skewness(a), 200, 5_000, seed=3)   # a lambda takes the loop path
    assert bootstrap_ci(v, skewness, 200, 5_000, seed=3) == pytest.approx(looped, abs=1e-6)


def test_fast_spearman_equals_pandas_with_missing_values_and_ties():
    rng = np.random.default_rng(4)
    d = pd.DataFrame(rng.normal(size=(800, 6)))
    d[1] = d[0] * 2 + rng.normal(size=800) * 0.1
    d.loc[rng.random(800) < 0.2, 1] = np.nan
    d.loc[rng.random(800) < 0.1, 3] = np.nan
    d[5] = np.round(d[5])
    np.testing.assert_allclose(_spearman(d).to_numpy(), d.corr(method="spearman").to_numpy(), atol=1e-12)


def test_vectorized_lookup_equals_groupby_means():
    rng = np.random.default_rng(5)
    c_fit, c_ev = rng.integers(0, 40, 2000), rng.integers(0, 45, 500)
    y = rng.normal(size=(2000, 2))
    g = pd.DataFrame(y).assign(_c=c_fit).groupby("_c")
    means, sizes = g.mean(), g.size()
    table = {c: means.loc[c].to_numpy() for c in sizes.index[sizes >= 5]}
    ref = np.array([table.get(c, y.mean(0)) for c in c_ev])
    np.testing.assert_allclose(_lookup_predict(c_fit, y, c_ev), ref)


# --------------------------------------------------------------------------- #
# calibration on data where the truth is known
# --------------------------------------------------------------------------- #

def test_corrected_ttest_controls_false_positives_where_the_naive_test_does_not():
    """Two copies of one learner that differ only in their random seed have the same
    expected score. Over 100 data sets, the corrected test rejects at most ~alpha;
    the uncorrected paired t-test on overlapping folds rejects far more often."""
    from sklearn.model_selection import RepeatedKFold, cross_val_score
    from sklearn.tree import DecisionTreeRegressor

    rej_c = rej_u = 0
    S = 100
    for s in range(S):
        g = np.random.default_rng(100 + s)
        X = g.normal(size=(200, 5))
        y = X @ g.normal(size=5) + g.normal(size=200)
        cv = list(RepeatedKFold(n_splits=5, n_repeats=2, random_state=s).split(X))
        a = cross_val_score(DecisionTreeRegressor(max_depth=4, splitter="random", random_state=2 * s), X, y, cv=cv)
        b = cross_val_score(DecisionTreeRegressor(max_depth=4, splitter="random", random_state=2 * s + 1), X, y,
                            cv=cv)
        rej_c += corrected_ttest(a, b, 0.25)[2] < 0.05
        rej_u += corrected_ttest(a, b, 0.0)[2] < 0.05
    assert rej_c / S <= 0.10
    assert rej_u / S > 2 * rej_c / S


def test_skewness_bootstrap_interval_coverage():
    """Percentile bootstrap intervals of the skewness of normal samples (true skew 0)
    cover it a little less often than 95% at n = 500 (about 92% in this simulation);
    the README states this. A drop below 88% means the interval code broke."""
    rng = np.random.default_rng(0)
    cov = 0
    for i in range(200):
        lo, hi = bootstrap_ci(rng.normal(size=500), skewness, 200, 5_000, i)
        cov += lo <= 0 <= hi
    assert 0.88 <= cov / 200 <= 0.99


def test_wilson_interval_coverage_is_near_nominal():
    for p, n in [(0.1, 100), (0.5, 50), (0.02, 400)]:
        k = np.arange(n + 1)
        inside = np.array([lo <= p <= hi for lo, hi in (wilson_ci(int(x), n) for x in k)])
        assert stats.binom.pmf(k, n, p)[inside].sum() >= 0.92


@pytest.mark.parametrize("classification", [True, False])
def test_missingness_permutation_test_is_calibrated_when_values_are_missing_at_random(classification):
    """Missing completely at random: the target relation's p-value rejects at about alpha."""
    rej, S = 0, 200
    for s in range(S):
        g = np.random.default_rng(s)
        y = pd.Series(g.integers(0, 2, 500) if classification else g.normal(size=500))
        m = g.random(500) < 0.1
        out = _target_relation(m, _target_matrix(y, classification), np.random.default_rng(1000 + s))
        rej += out["p_value"] < 0.05
    assert rej / S <= 0.09


def test_missingness_permutation_test_detects_a_real_relation():
    g = np.random.default_rng(0)
    y = pd.Series(g.integers(0, 2, 1000))
    m = (g.random(1000) < np.where(y == 1, 0.25, 0.05))          # missing more often when y = 1
    out = _target_relation(m, _target_matrix(y, True), g)
    assert out["p_value"] < 0.01 and abs(out["missing"] - out["observed"]) > 0.2


def test_weighted_auc_and_ap_bootstrap_match_sklearn_on_unit_weights():
    """With every unit drawn once, the weighted AUC / AP equal scikit-learn's; the
    bootstrap interval then brackets the full-sample value."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    from prefit.posttrain import _score_intervals

    g = np.random.default_rng(0)
    yb = (g.random(600) < 0.3).astype(float)
    sc = np.round(yb * 0.8 + g.normal(size=600), 1)                  # rounded: many ties
    from prefit.posttrain import _ranking_order, _weighted_auc_ap

    auc, ap = _weighted_auc_ap(yb, np.ones(600), _ranking_order(sc))
    assert auc == pytest.approx(roc_auc_score(yb, sc), abs=1e-12)
    assert ap == pytest.approx(average_precision_score(yb, sc), abs=1e-12)
    w = g.integers(0, 3, 600).astype(float)                         # integer weights = repeated rows
    rep = np.repeat(np.arange(600), w.astype(int))
    auc_w, ap_w = _weighted_auc_ap(yb, w, _ranking_order(sc))
    assert auc_w == pytest.approx(roc_auc_score(yb[rep], sc[rep]), abs=1e-12)
    assert ap_w == pytest.approx(average_precision_score(yb[rep], sc[rep]), abs=1e-12)
    iv = _score_intervals(yb, sc, np.arange(600), 400, 0)
    assert iv["roc_auc"][0] < roc_auc_score(yb, sc) < iv["roc_auc"][1]
