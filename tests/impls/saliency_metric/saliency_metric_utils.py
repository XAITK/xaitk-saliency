from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np


def saliency_metric_assertions(
    computation: Callable[
        [np.ndarray],
        float,
    ],
    sal_map: np.ndarray,
) -> float:
    """Test that the inputs are not modified while computing an image metric.

    :param computation: Interface to test the compute() function on
    :param sal_map: Input saliency map.
    """
    original_sal_map = copy.deepcopy(sal_map)

    metric_value = computation(sal_map)

    assert np.array_equal(original_sal_map, sal_map), "sal_map modified, data changed"

    return metric_value
