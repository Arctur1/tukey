import numpy as np
from scipy import stats
from tukey import tukey_generic


def test_tukey_generic_vs_tukey_hsd():
    np.random.seed(42)
    groups = [
        np.random.normal(0, 1, 10),
        np.random.normal(1, 1, 10),
        np.random.normal(2, 1, 10),
    ]

    means = [np.mean(group) for group in groups]
    sizes = [len(group) for group in groups]
    variances = [np.var(group, ddof=1) for group in groups]

    generic_result = tukey_generic(means, sizes, variances)
    scipy_result = stats.tukey_hsd(*groups)

    # Verify array shapes
    assert generic_result.statistic.shape == (3, 3)
    assert generic_result.pvalue.shape == (3, 3)

    # Compare with scipy implementation
    np.testing.assert_allclose(generic_result.statistic, scipy_result.statistic)
    np.testing.assert_allclose(generic_result.pvalue, scipy_result.pvalue)
