"""Implementation of SimilarityScoring scorer"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import maxabs_scale

from xaitk_saliency import GenerateDescriptorSimilaritySaliency
from xaitk_saliency.utils.masking import weight_regions_by_scalar


class SimilarityScoring(GenerateDescriptorSimilaritySaliency):
    """
    This saliency implementation transforms proximity in feature space into
    saliency heatmaps.
    This should require feature vectors for the reference image, for each query
    image, and for perturbed versions of the reference image, as well as the
    masks of the reference image perturbations (as would be output from a
    `PerturbImage` implementation).

    The resulting saliency maps are relative to the reference image.
    As such, each map denotes regions in the reference image that make it more
    or less similar to the corresponding query image.
    """

    def __init__(self, proximity_metric: str = "euclidean") -> None:
        """
        Initialization for SimilarityScoring

        :param proximity_metric: The type of comparison metric used
            to determine proximity in feature space. The type of comparison
            metric supported is restricted by scipy's cdist() function. The
            following metrics are supported in scipy.

            ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
            ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
            ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’,
            ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’,
            ‘sqeuclidean’, ‘wminkowski’, ‘yule’.
        """
        try:
            # NOTE: cdist attempts to type-check with the wrong overloaded function
            # and hence throws an error for a str input for the metric, although it is
            # supported by the function as per the docs here:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

            # Attempting to use chosen comparison metric
            cdist([[1], [1]], [[1], [1]], proximity_metric)  # type: ignore
            self.proximity_metric: str = proximity_metric
        except ValueError as err:
            raise ValueError("Chosen comparison metric not supported or may not be available in scipy") from err

    def generate(
        self,
        ref_descr: np.ndarray,
        query_descrs: np.ndarray,
        perturbed_descrs: np.ndarray,
        perturbed_masks: np.ndarray,
    ) -> np.ndarray:
        """
        Generate visual saliency heatmaps for similarity from vectors

        :param ref_descr: np.ndarray
            Feature vectors from the reference image
        :param query_descrs: np.ndarray
            Query vectors from the reference image
        :param perturbed_descrs: np.ndarray
            Perturbed vectors from the reference image
        :param perturbed_masks: np.ndarray
            Perturbation masks `numpy.ndarray` over the reference image.

        :return: np.ndarray
            Generated visual saliency heatmap.
        """
        if len(perturbed_descrs) != len(perturbed_masks):
            raise ValueError("Number of perturbation masks and respective feature vector do not match.")

        if len(ref_descr) != query_descrs.shape[1]:
            raise ValueError("Size of feature vectors between reference and query images do not match.")

        # Computing original proximity between reference image feature vector
        # and each query image feature vector.
        original_proximity = cdist(ref_descr.reshape(1, -1), query_descrs, metric=self.proximity_metric)  # type: ignore

        # Computing proximity between query feature vectors and perturbed
        # reference image feature vectors.
        perturbed_proximity = cdist(query_descrs, perturbed_descrs, metric=self.proximity_metric)  # type: ignore

        # Iterating through each distance and compare it with
        # its perturbed twin
        diff = perturbed_proximity - np.transpose(original_proximity)

        diff = np.transpose(np.clip(diff, 0, None))

        # Weighting perturbed regions with respective difference in confidence
        sal = weight_regions_by_scalar(diff, perturbed_masks)

        # Normalize final saliency maps
        sal = maxabs_scale(sal.reshape(sal.shape[0], -1), axis=1).reshape(sal.shape)

        # Ensure saliency map in range [-1, 1]
        return np.clip(sal, -1, 1)

    def get_config(self) -> dict:
        """
        Get the configuration dictionary of the SimilarityScoring instance.

        Returns:
            dict[str, Any]: Configuration dictionary.
        """
        return {
            "proximity_metric": self.proximity_metric,
        }
