from typing import Tuple, Optional, Sequence, Union, Dict, Any
import numpy as np

from smqtk_descriptors.interfaces.image_descriptor_generator import ImageDescriptorGenerator

from xaitk_saliency import GenerateImageSimilarityBlackboxSaliency
from xaitk_saliency.impls.gen_image_similarity_blackbox_sal.occlusion_based import PerturbationOcclusion
from xaitk_saliency.impls.gen_descriptor_sim_sal.similarity_scoring import SimilarityScoring
from xaitk_saliency.impls.perturb_image.sliding_window import SlidingWindow


class SBSMStack (GenerateImageSimilarityBlackboxSaliency):
    """
    Encapsulation of the perturbation-occlusion method using specifically the
    sliding window image perturbation and similarity scoring algorithms to
    generate similarity-based visual saliency maps.
    See the documentation of :class:`SlidingWindow` and
    :class:`SimilarityScoring` for details.

    :param window_size: The block window size as a tuple with format
        `(height, width)`.
    :param stride: The sliding window striding step as a tuple with format
        `(height_step, width_step)`.
    :param proximity_metric: The type of comparison metric used
        to determine proximity in feature space. The type of comparison
        metric supported is restricted by scipy's cdist() function. The
        following metrics are supported in scipy.

        ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
        ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’,
        ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’,
        ‘sqeuclidean’, ‘wminkowski’, ‘yule’.
    :param threads: Optional number threads to use to enable parallelism in
        applying perturbation masks to an input image.
        If 0, a negative value, or `None`, work will be performed on the
        main-thread in-line.
    """
    def __init__(
        self,
        window_size: Tuple[int, int] = (50, 50),
        stride: Tuple[int, int] = (20, 20),
        proximity_metric: str = 'euclidean',
        fill: Optional[Union[int, Sequence[int], np.ndarray]] = None,
        threads: Optional[int] = None
    ):
        self._po = PerturbationOcclusion(
            perturber=SlidingWindow(
                window_size=window_size,
                stride=stride
            ),
            generator=SimilarityScoring(
                proximity_metric=proximity_metric
            ),
            fill=fill,
            threads=threads
        )

    @property
    def fill(self) -> Optional[Union[int, Sequence[int], np.ndarray]]:
        return self._po.fill

    @fill.setter
    def fill(self, v: Optional[Union[int, Sequence[int], np.ndarray]]) -> None:
        self._po.fill = v

    def _generate(
        self,
        ref_image: np.ndarray,
        query_images: Sequence[np.ndarray],
        blackbox: ImageDescriptorGenerator
    ) -> np.ndarray:
        return self._po.generate(
            ref_image,
            query_images,
            blackbox
        )

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
        c_p = self._po._perturber.get_config()
        c_g = self._po._generator.get_config()
        c = dict(c_p, **c_g)
        c['fill'] = self._po.fill
        c['threads'] = self._po._threads
        return c
