"""
Provides an implementation of the `SaliencyMetric` interface for computing
entropy over a saliency map using `scipy.stats.entropy`.

Classes:
    Entropy: Computes the entropy of a given saliency map.

Example:
    >> entropy_saliency_metric = Entropy(clip_range=(0, 1))
    >> result = entropy_saliency_metric(sal_map)  # entropy_saliency_metric.compute(sal_map)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import entropy
from typing_extensions import override

from xaitk_saliency.interfaces.saliency_metric import SaliencyMetric


class Entropy(SaliencyMetric):
    """
    Implementation of the `SaliencyMetric` interface to calculate entropy.

    Entropy is a statistical measure of randomness that quantifies the amount of
    information in a saliency map. High entropy indicates complex, detailed information,
    while low entropy suggests uniform or less informative regions.

    Attributes:
        clip_range (tuple[float, float] | None): Optional (min, max) range to clip
            saliency values before computing entropy.
    """

    def __init__(self, clip_range: tuple[float, float] | None = None) -> None:
        """
        Initialize the Entropy metric.

        Args:
            clip_range (tuple[float, float] | None): Optional (min, max) value range to
                clip the saliency map before computing entropy. The metric computation
                does a min-max normalization if `clip_range` is not provided. If input
                already contains a [-1,1] normalized saliency map, the following clip
                ranges can be used:
                - Full Saliency: clip_range = (-1, 1)
                - Positive Saliency: clip_range = (0, 1)
                - Negative Saliency: clip_range = (-1, 0)
        """
        self.clip_range = clip_range
        if self.clip_range is not None and len(self.clip_range) != 2:
            raise ValueError("Invalid clip range. 2-element tuple of floats")

    @override
    def compute(
        self,
        sal_map: np.ndarray,
    ) -> float:
        """
        Compute the entropy of a saliency map.

        Entropy is computed using `scipy.stats.entropy` on the clipped/normalized saliency map.
        If `clip_range` is provided, the metric computation only performs range clipping and does
        not perform normalization. However, if `clip_range` is not provided, the metric computation
        performs only a min-max normalization.

        Args:
            sal_map (np.ndarray): The input saliency map.

        Returns:
            float: The entropy value of the saliency map.

        Warnings:
            UserWarning: If `ref_sal_map` is provided, it is ignored for entropy computation.

        """
        if self.clip_range is None:
            # The provided 1e-10 is a small non-zero value to prevent division by zero.
            s = (sal_map - sal_map.min()) / max((sal_map.max() - sal_map.min()), 1e-10)
        else:
            s = np.clip(sal_map, self.clip_range[0], self.clip_range[1])
        return float(entropy(s.ravel(), base=2))

    @override
    def get_config(self) -> dict[str, Any]:
        """
        Generates a configuration dictionary for the Entropy metric instance.

        Returns:
            dict[str, Any]: Configuration data representing the sensor and scenario.
        """
        return {
            "clip_range": self.clip_range,
        }
