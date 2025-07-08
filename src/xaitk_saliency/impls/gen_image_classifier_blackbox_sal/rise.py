"""Encapsulation of the perturbation-occlusion method using specifically the
RISE implementations of the component algorithms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from smqtk_classifier.interfaces.classify_image import ClassifyImage

from xaitk_saliency.impls.gen_classifier_conf_sal.rise_scoring import RISEScoring
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.occlusion_based import PerturbationOcclusion
from xaitk_saliency.impls.perturb_image.rise import RISEGrid
from xaitk_saliency.interfaces.gen_image_classifier_blackbox_sal import GenerateImageClassifierBlackboxSaliency


class RISEStack(GenerateImageClassifierBlackboxSaliency):
    """
    Encapsulation of the perturbation-occlusion method using specifically the
    RISE implementations of the component algorithms.

    This more specifically encapsulates the original RISE method as presented
    in their paper and code. See references in the :class:`RISEGrid`
    and :class:`RISEScoring` documentation.

    This implementation shares the `p1` probability with the internal
    `RISEScoring` instance use, effectively causing this implementation to
    utilize debiased RISE.
    """

    def __init__(
        self,
        n: int,
        s: int,
        p1: float,
        seed: int | None = None,
        threads: int = 0,
        debiased: bool = True,
    ) -> None:
        """
        Initialization of the perturbation-occlusion method using specifically the
        RISE implementations of the component algorithms.

        :param n:
            Number of random masks used in the algorithm. E.g. 1000.
        :param s:
            Spatial resolution of the small masking grid. E.g. 8.
            Assumes square grid.
        :param p1:
            Probability of the grid cell being set to 1 (otherwise 0).
            This should be a float value in the [0, 1] range. E.g. 0.5.
        :param seed:
            A seed to pass into the constructed random number generator to allow
            for reproducibility
        :param threads: The number of threads to utilize when generating masks.
            If this is <=0 or None, no threading is used and processing
            is performed in-line serially.
        :param debiased: If we should pass the provided debiasing parameter to the
            RISE saliency map generation algorithm. See the :meth:`.RISEScoring`
            documentation for more details on debiasing.
        """
        self._debiased = debiased  # retain for config output
        self._po = PerturbationOcclusion(
            RISEGrid(n=n, s=s, p1=p1, seed=seed, threads=threads),
            RISEScoring(p1=p1 if debiased else 0.0),
            threads=threads,
        )

    @property
    def fill(self) -> int | Sequence[int] | None:
        """Gets the fill value"""
        return self._po.fill

    @fill.setter
    def fill(self, v: int | Sequence[int] | None) -> None:
        self._po.fill = v

    def _generate(self, ref_image: np.ndarray, blackbox: ClassifyImage) -> np.ndarray:
        return self._po.generate(ref_image, blackbox)

    def get_config(self) -> dict[str, Any]:
        """
        Get the configuration dictionary of the RISEStack instance.

        Returns:
            dict[str, Any]: Configuration dictionary.
        """
        # It turns out that our configuration here is equivalent to that given
        # and retrieved from the RISEGrid implementation that is known set to
        # the internal ``PerturbationOcclusion.perturber``.
        # noinspection PyProtectedMember
        po_config = self._po.get_config()
        c = po_config["perturber"][po_config["perturber"]["type"]]
        c["debiased"] = self._debiased
        return c
