import numpy as np


def test_paired_bootstrap_and_holm_are_deterministic():
    from experiments.report import holm_bonferroni, paired_bootstrap_delta
    mean, interval = paired_bootstrap_delta([3, 4, 5], [1, 3, 4], n=500, seed=7)
    assert mean == 4 / 3
    assert interval[0] <= mean <= interval[1]
    assert holm_bonferroni([0.01, 0.02, 0.9]) == [0.03, 0.04, 0.9]
