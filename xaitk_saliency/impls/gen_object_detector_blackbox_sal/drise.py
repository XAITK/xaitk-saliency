import numpy as np
from typing import Optional, Any, Dict, Union, Sequence

from xaitk_saliency.interfaces.gen_object_detector_blackbox_sal import GenerateObjectDetectorBlackboxSaliency
from xaitk_saliency.impls.gen_object_detector_blackbox_sal.occlusion_based import PerturbationOcclusion
from xaitk_saliency.impls.perturb_image.rise import RISEGrid
from xaitk_saliency.impls.gen_detector_prop_sal.drise_scoring import DRISEScoring

from smqtk_detection import DetectImageObjects


class DRISEStack (GenerateObjectDetectorBlackboxSaliency):
    """
    Encapsulation of the perturbation-occlusion method using the RISE image
    perturbation and DRISE scoring algorithms to generate visual saliency maps
    for object detections.
    See references in the :class:`RISEGrid` and :class:`DRISEScoring`
    documentation.

    :param n: Number of random masks used in the algorithm.
    :param s: Spatial resolution of the small masking grid.
        Assumes square grid.
    :param p1: Probability of the grid cell being set to 1 (not occluded).
        This should be a float value in the [0, 1] range.
    :param seed: A seed to pass to the constructed random number generator to
        allow for reproducibility.
    :param threads: The number of threads to utilize when generating masks.
        If this is <=0 or None, no threading is used and processing
        is performed in-line serially.
    """

    def __init__(
        self,
        n: int,
        s: int,
        p1: float,
        seed: Optional[int] = None,
        threads: Optional[int] = 0,
    ):
        self._po = PerturbationOcclusion(
            RISEGrid(n=n, s=s, p1=p1, seed=seed, threads=threads),
            DRISEScoring(),
            threads=threads
        )

    @property
    def fill(self) -> Optional[Union[int, Sequence[int]]]:
        return self._po.fill

    @fill.setter
    def fill(self, v: Optional[Union[int, Sequence[int]]]) -> None:
        self._po.fill = v

    def _generate(
        self,
        ref_image: np.ndarray,
        bboxes: np.ndarray,
        scores: np.ndarray,
        blackbox: DetectImageObjects,
        objectness: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self._po.generate(
            ref_image,
            bboxes,
            scores,
            blackbox,
            objectness,
        )

    def get_config(self) -> Dict[str, Any]:
        # It turns out that our configuration here is equivalent to that given
        # and retrieved from the RISEGrid implementation that is known set to
        # the internal ``PerturbationOcclusion.perturber``.
        # noinspection PyProtectedMember
        return self._po._perturber.get_config()
