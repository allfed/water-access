"""Tests for canonical Monte Carlo sampling parameters."""

import numpy as np
import pytest

from src.monte_carlo_parameters import (
    PARAMETER_NAMES,
    RURAL_BOREHOLE_LOSE_FRACTION_LOWER_ESTIMATE,
    RURAL_BOREHOLE_LOSE_FRACTION_UPPER_ESTIMATE,
    TIME_GATHERING_LOWER_ESTIMATE,
    TIME_GATHERING_UPPER_ESTIMATE,
    URBAN_BOREHOLE_LOSE_FRACTION_LOWER_ESTIMATE,
    URBAN_BOREHOLE_LOSE_FRACTION_UPPER_ESTIMATE,
    UNALLOCATED_VULNERABLE_FRACTION_LOWER_ESTIMATE,
    UNALLOCATED_VULNERABLE_FRACTION_UPPER_ESTIMATE,
    sample_parameters,
)


def test_sample_parameters_returns_all_keys():
    params = sample_parameters(5)
    assert list(params.keys()) == list(PARAMETER_NAMES)
    for name, values in params.items():
        assert len(values) == 5, name


def test_borehole_and_unallocated_ranges():
    params = sample_parameters(200)
    urban = params["urban_borehole_lose_fractions"]
    rural = params["rural_borehole_lose_fractions"]
    g = params["unallocated_vulnerable_fractions"]
    assert urban.min() >= URBAN_BOREHOLE_LOSE_FRACTION_LOWER_ESTIMATE
    assert urban.max() <= URBAN_BOREHOLE_LOSE_FRACTION_UPPER_ESTIMATE
    assert rural.min() >= RURAL_BOREHOLE_LOSE_FRACTION_LOWER_ESTIMATE
    assert rural.max() <= RURAL_BOREHOLE_LOSE_FRACTION_UPPER_ESTIMATE
    assert g.min() >= UNALLOCATED_VULNERABLE_FRACTION_LOWER_ESTIMATE
    assert g.max() <= UNALLOCATED_VULNERABLE_FRACTION_UPPER_ESTIMATE


def test_time_gathering_is_positive_with_median_in_range():
    params = sample_parameters(500)
    times = params["time_gatherings"]
    assert times.min() > 0
    assert TIME_GATHERING_LOWER_ESTIMATE <= np.median(times) <= TIME_GATHERING_UPPER_ESTIMATE
