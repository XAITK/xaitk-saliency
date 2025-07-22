"""Encapsulation of the perturbation-occlusion method using specifically
sliding windows and the occlusion-scoring method."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from smqtk_classifier.interfaces.classify_image import ClassifyImage

from xaitk_saliency.impls.gen_classifier_conf_sal.occlusion_scoring import OcclusionScoring
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.occlusion_based import PerturbationOcclusion
from xaitk_saliency.impls.perturb_image.sliding_window import SlidingWindow
from xaitk_saliency.interfaces.gen_image_classifier_blackbox_sal import GenerateImageClassifierBlackboxSaliency


class SlidingWindowStack(GenerateImageClassifierBlackboxSaliency):
    """
    Encapsulation of the perturbation-occlusion method using specifically
    sliding windows and the occlusion-scoring method.
    See the :class:`SlidingWindow` and :class:`OcclusionScoring` documentation
    for more details.
    """

    def __init__(
        self,
        window_size: tuple[int, int] = (50, 50),
        stride: tuple[int, int] = (20, 20),
        threads: int = 0,
    ) -> None:
        """
        Initialization of the perturbation-occlusion method using specifically
        sliding windows and the occlusion-scoring method.

        :param window_size: The block window size as a tuple with format
            `(height, width)`.
        :param stride: The sliding window striding step as a tuple with format
            `(height_step, width_step)`.
        :param threads: Optional number threads to use to enable parallelism in
            applying perturbation masks to an input image.
            If 0, a negative value, or `None`, work will be performed on the
            main-thread in-line.
        """
        self._po = PerturbationOcclusion(
            perturber=SlidingWindow(
                window_size=window_size,
                stride=stride,
            ),
            generator=OcclusionScoring(),
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

    @classmethod
    def get_default_config(cls) -> dict[str, Any]:
        """
        Returns the default configuration for the SlidingWindowStack.

        This method provides a default configuration dictionary, specifying default
        values for key parameters in the factory. It can be used to create an instance
        of the factory with preset configurations.

        Returns:
            dict[str, Any]: A dictionary containing default configuration parameters.
        """
        # Minor override to curry tuple defaults into lists, which are the
        # JSON-parsed types. This is to allow successful equality between
        # default, get_config() and JSON-parsed outputs.
        cfg = super().get_default_config()
        cfg["window_size"] = list(cfg["window_size"])
        cfg["stride"] = list(cfg["stride"])
        return cfg

    def get_config(self) -> dict[str, Any]:
        """
        Get the configuration dictionary of the SlidingWindowStack instance.

        Returns:
            dict[str, Any]: Configuration dictionary.
        """
        # It turns out that our configuration here is nearly equivalent to that
        # given and retrieved from the SlidingWindow implementation that is
        # known set to the internal ``PerturbationOcclusion.perturber``.
        # noinspection PyProtectedMember
        po_config = self._po.get_config()
        c = po_config["perturber"][po_config["perturber"]["type"]]
        # noinspection PyProtectedMember
        c["threads"] = po_config["threads"]
        return c
