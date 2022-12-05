"""
Implementation of paired bootstrap test
`(Efron & Tibshirani, 1994) <https://cds.cern.ch/record/526679/files/0412042312_TOC.pdf>`_.
"""
from sklearn.metrics import accuracy_score
# STD
from typing import Optional
import scipy

# EXT
from joblib import Parallel, delayed
import numpy as np

# PKG
from conversions import ArrayLike, score_pair_conversion


@score_pair_conversion
def bootstrap_CI(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    alpha: float = 0.95,
    num_samples: int = 1000,
    num_jobs: int = 1,
    seed: Optional[int] = None,
) -> (float, float):
    """
    Implementation of paired bootstrap test. A p-value is being estimated by comparing the mean of scores
    for two algorithms to the means of resampled populations, where `num_samples` determines the number of
    times we resample.
    The test is single-tailed, where we want to verify that the algorithm corresponding to `scores_a` is better than
    the one `scores_b` originated from.
    Parameters
    ----------
    scores_a: ArrayLike
        Scores of algorithm A.
    scores_b: ArrayLike
        Scores of algorithm B.
    alpha: float
        significance level
    num_samples: int
        Number of bootstrap samples used for estimation.
    num_jobs: int
        Number of threads that bootstrap iterations are divided among.
    seed: Optional[int]
        Set seed for reproducibility purposes. Default is None (meaning no seed is used).
    Returns
    -------
    float
        lower CI bound, upper CI bound.
    """
    assert len(scores_a) == len(scores_b), "Scores have to be of same length."
    assert (
        len(scores_a) > 0 and len(scores_b) > 0
    ), "Both lists of scores must be non-empty."
    assert num_samples > 0, "num_samples must be positive, {} found.".format(
        num_samples
    )

    N = len(scores_a)

    # Set seeds for different jobs if applicable
    # "Sub-seeds" for jobs are just seed argument + job index
    seeds = (
        [None] * num_samples
        if seed is None
        else [seed + offset for offset in range(1, num_samples + 1)]
    )

    def _bootstrap_iter(seed: Optional[int] = None):
        """
        One bootstrap iteration. Wrapped in a function so it can be handed to joblib.Parallel.
        """
        # When running multiple jobs, modules have to be re-imported for some reason to avoid an error
        # Use dir() to check whether module is available in local scope:
        # https://stackoverflow.com/questions/30483246/how-to-check-if-a-module-has-been-imported
        if "np" not in dir():
            import numpy as np

        if seed is not None:
            np.random.seed(seed)

        boot_sample_inds = np.random.randint(0, N, N)  # sample with repetitions
        temp_gold_y = np.zeros(N)
        temp_pred_y = np.zeros(N)
        for ii, ind in enumerate(boot_sample_inds):
            temp_gold_y[ii] = scores_a[ind]
            temp_pred_y[ii] = scores_b[ind]

        # resampled_scores_a = np.random.choice(scores_a, N)
        # resampled_scores_b = np.random.choice(scores_b, N)
        new_delta = accuracy_score(temp_gold_y, temp_pred_y)
        return new_delta

    # Initialize worker pool and start iterations
    parallel = Parallel(n_jobs=num_jobs)
    samples = parallel(
        delayed(_bootstrap_iter)(seed)
        for _, seed in zip(range(num_samples), seeds)
    )
    # Percentile Bootstrap
    p = (alpha / 2.0) * 100
    lower = max(0.0, np.percentile(samples, p))
    p = (1.0 - (alpha / 2.0)) * 100
    upper = min(1.0, np.percentile(samples, p))

    # for normal bootstrap
    std_sample = np.std(samples)
    z = scipy.stats.norm.ppf((1 + alpha) / 2.0)
    avg_delta = np.average(samples)
    lower_normal = avg_delta - std_sample * z
    upper_normal = avg_delta + std_sample * z

    # Empirical Bootstrap

    return lower, upper, lower_normal, upper_normal



@score_pair_conversion
def weighted_bootstrap_CI(
    scores_a: ArrayLike,
    scores_b: ArrayLike,
    scores_unlabeled: ArrayLike,
    weights: ArrayLike,
    alpha: float = 0.95,
    num_samples: int = 1000,
    num_jobs: int = 1,
    seed: Optional[int] = None,
) -> (float, float):
    """
    Implementation of paired bootstrap test. A p-value is being estimated by comparing the mean of scores
    for two algorithms to the means of resampled populations, where `num_samples` determines the number of
    times we resample.
    The test is single-tailed, where we want to verify that the algorithm corresponding to `scores_a` is better than
    the one `scores_b` originated from.
    Parameters
    ----------
    scores_a: ArrayLike
        Scores of algorithm A.
    scores_b: ArrayLike
        Scores of algorithm B.
    alpha: float
        significance level
    num_samples: int
        Number of bootstrap samples used for estimation.
    num_jobs: int
        Number of threads that bootstrap iterations are divided among.
    seed: Optional[int]
        Set seed for reproducibility purposes. Default is None (meaning no seed is used).
    Returns
    -------
    float
        lower CI bound, upper CI bound.
    """
    assert len(scores_a) == len(scores_b), "Scores have to be of same length."
    assert (
        len(scores_a) > 0 and len(scores_b) > 0
    ), "Both lists of scores must be non-empty."
    assert num_samples > 0, "num_samples must be positive, {} found.".format(
        num_samples
    )

    N = len(scores_a)
    N_u = len(scores_unlabeled)

    # Set seeds for different jobs if applicable
    # "Sub-seeds" for jobs are just seed argument + job index
    seeds = (
        [None] * num_samples
        if seed is None
        else [seed + offset for offset in range(1, num_samples + 1)]
    )

    def _bootstrap_iter(seed: Optional[int] = None):
        """
        One bootstrap iteration. Wrapped in a function so it can be handed to joblib.Parallel.
        """
        # When running multiple jobs, modules have to be re-imported for some reason to avoid an error
        # Use dir() to check whether module is available in local scope:
        # https://stackoverflow.com/questions/30483246/how-to-check-if-a-module-has-been-imported
        if "np" not in dir():
            import numpy as np

        if seed is not None:
            np.random.seed(seed)

        unlabeled_boot_sample_inds = np.random.randint(0, N_u, N)  # sample with repetitions
        boot_sample_inds = np.zeros(N)
        for ind, unlabeled_ind in enumerate(unlabeled_boot_sample_inds):
            boot_sample_inds[ind] = np.argmax(np.random.multinomial(1, weights[unlabeled_ind], size=1))
        resampled_scores_a = np.zeros(N)
        resampled_scores_b = np.zeros(N)
        for ii, ind in enumerate(boot_sample_inds):
            resampled_scores_a[ii] = scores_a[int(ind)]
            resampled_scores_b[ii] = scores_b[int(ind)]

        # resampled_scores_a = np.random.choice(scores_a, N)
        # resampled_scores_b = np.random.choice(scores_b, N)
        new_delta = accuracy_score(resampled_scores_a, resampled_scores_b)
        return new_delta

    # Initialize worker pool and start iterations
    parallel = Parallel(n_jobs=num_jobs)
    samples = parallel(
        delayed(_bootstrap_iter)(seed)
        for _, seed in zip(range(num_samples), seeds)
    )
    # Percentile Bootstrap
    p = (alpha / 2.0) * 100
    lower = max(0.0, np.percentile(samples, p))
    p = (1.0 - (alpha / 2.0))* 100
    upper = min(1.0, np.percentile(samples, p))

    # for normal bootstrap
    std_sample = np.std(samples)
    z = scipy.stats.norm.ppf((1 + alpha) / 2.0)
    avg_delta = np.average(samples)
    lower_normal = avg_delta - std_sample * z
    upper_normal = avg_delta + std_sample * z

    # Empirical Bootstrap


    return lower, upper, lower_normal, upper_normal
