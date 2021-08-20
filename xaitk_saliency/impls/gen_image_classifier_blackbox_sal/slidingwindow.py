from typing import Any, Dict, Optional, Sequence, Tuple, Union


import numpy as np
from smqtk_classifier import ClassifyImage

from xaitk_saliency.impls.gen_classifier_conf_sal.occlusion_scoring import OcclusionScoring
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.occlusion_based import PerturbationOcclusion
from xaitk_saliency.impls.perturb_image.sliding_window import SlidingWindow
from xaitk_saliency.interfaces.gen_image_classifier_blackbox_sal import GenerateImageClassifierBlackboxSaliency


class SlidingWindowStack (GenerateImageClassifierBlackboxSaliency):
    """
    Encapsulation of the perturbation-occlusion method using specifically
    sliding windows and the occlusion-scoring method.

    :param window_size: The block window size as a tuple with format
        `(height, width)`.
    :param stride: The sliding window striding step as a tuple with format
        `(height_step, width_step)`.
    :param threads: Optional number threads to use to enable parallelism in
        applying perturbation masks to an input image. If 0, a negative value,
        or `None`, work will be performed on the main-thread in-line.
    """

    def __init__(
        self,
        window_size: Tuple[int, int] = (50, 50),
        stride: Tuple[int, int] = (20, 20),
        threads: int = 0
    ):
        self._po = PerturbationOcclusion(
            perturber=SlidingWindow(
                window_size=window_size,
                stride=stride,
            ),
            generator=OcclusionScoring(),
            threads=threads,
        )

    @property
    def fill(self) -> Optional[Union[int, Sequence[int]]]:
        return self._po.fill

    @fill.setter
    def fill(self, v: Optional[Union[int, Sequence[int]]]) -> None:
        self._po.fill = v

    def _generate(self, ref_image: np.ndarray, blackbox: ClassifyImage) -> np.ndarray:
        return self._po.generate(ref_image, blackbox)

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        # Minor override to curry tuple defaults into lists, which are the
        # JSON-parsed types. This is to allow successful equality between
        # default, get_config() and JSON-parsed outputs.
        cfg = super().get_default_config()
        cfg['window_size'] = list(cfg['window_size'])
        cfg['stride'] = list(cfg['stride'])
        return cfg

    def get_config(self) -> Dict[str, Any]:
        # It turns out that our configuration here is nearly equivalent to that
        # given and retrieved from the SlidingWindow implementation that is
        # known set to the internal ``PerturbationOcclusion.perturber``.
        # noinspection PyProtectedMember
        c = self._po._perturber.get_config()
        # noinspection PyProtectedMember
        c['threads'] = self._po._threads
        return c
