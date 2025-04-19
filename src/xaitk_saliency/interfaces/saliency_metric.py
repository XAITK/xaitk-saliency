"""
This module defines the `SaliencyMetric` abstract base class, an interface for computing
metrics for a single saliency map input. Implementations of `SaliencyMetric` should
define the specific metric computation in the `compute` method, which can be called directly or
via the `__call__` method.

Classes:
    SaliencyMetric: An interface outlining the computation of a given metric for
    saliency map analysis.

Dependencies:
    - numpy for numerical operations on saliency maps.
    - smqtk_core for configuration management and plugin compatibility.

Example usage:
    class SpecificMetric(SaliencyMetric):
        def compute(self, sal_map):
            # Define metric calculation logic here.
            pass

    metric = SpecificMetric()
    score = metric(sal_map)
"""

from __future__ import annotations

import abc

import numpy as np
from smqtk_core import Plugfigurable


class SaliencyMetric(Plugfigurable):
    """
    This interface outlines the computation of a given metric when provided with
    a single input saliency map.
    """

    @abc.abstractmethod
    def compute(
        self,
        sal_map: np.ndarray,
    ) -> float:
        """
        Given up to two saliency maps, and additional parameters, return some given metric about
        the saliency map(s).

        :param sal_map: An input saliency map.

        :return: Returns a single scalar value representing an implementation's computed metric.
                 Implementations should impart no side effects upon the input saliency map.
        """

    def __call__(
        self,
        sal_map: np.ndarray,
    ) -> float:
        """Calls compute() with the given input saliency map(s) and additional parameters.

        :param sal_map: An input saliency map.

        :return: Returns a single scalar value representing an implementation's computed metric.
                 Implementations should impart no side effects upon the input saliency map.
        """
        return self.compute(sal_map)

    @property
    def name(self) -> str:
        """
        Returns the name of the SaliencyMetric instance.

        This property provides a convenient way to retrieve the name of the
        class instance, which can be useful for logging, debugging, or display purposes.

        Returns:
            str: The name of the SaliencyMetric instance.
        """
        return self.__class__.__name__
