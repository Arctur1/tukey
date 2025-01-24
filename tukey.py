import numpy as np
import numpy.typing as ntypes
import scipy
import scipy.stats
from scipy.stats._result_classes import TukeyHSDResult


# taken from https://github.com/scipy/scipy/blob/v1.15.0/scipy/stats/_hypotests.py#L1989
def tukey_generic(
    means: ntypes.ArrayLike, sample_sizes: ntypes.ArrayLike, variance: ntypes.ArrayLike
) -> TukeyHSDResult:
    ntreatments = len(means)
    means = np.asarray(means)
    nsamples_treatments = np.asarray(sample_sizes)
    nobs = np.sum(nsamples_treatments)

    # determine mean square error [5]. Note that this is sometimes called
    # mean square error within.
    mse = np.sum(variance * (nsamples_treatments - 1)) / (nobs - ntreatments)

    # The calculation of the standard error differs when treatments differ in
    # size. See ("Unequal sample sizes")[1].
    if np.unique(nsamples_treatments).size == 1:
        # all input groups are the same length, so only one value needs to be
        # calculated [1].
        normalize = 2 / nsamples_treatments[0]
    else:
        # to compare groups of differing sizes, we must compute a variance
        # value for each individual comparison. Use broadcasting to get the
        # resulting matrix. [3], verified against [4] (page 308).
        normalize = 1 / nsamples_treatments + 1 / nsamples_treatments[None].T

    # the standard error is used in the computation of the tukey criterion and
    # finding the p-values.
    stand_err = np.sqrt(normalize * mse / 2)

    # the mean difference is the test statistic.
    mean_differences = means[None].T - means

    # Calculate the t-statistic to use within the survival function of the
    # studentized range to get the p-value.
    t_stat = np.abs(mean_differences) / stand_err

    params = t_stat, ntreatments, nobs - ntreatments
    pvalues = scipy.stats.distributions.studentized_range.sf(*params)

    return TukeyHSDResult(mean_differences, pvalues, ntreatments, nobs, stand_err)
