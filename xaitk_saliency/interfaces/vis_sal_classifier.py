import abc

import numpy as np
from smqtk_core import Plugfigurable

T = TypeVar("T", bound="ImageClassifierSaliencyMapGenerator")

class ImageClassifierSaliencyMapGenerator(Plugfigurable)
    """
    This interface proposes that implementations transform black-box
    image classification scores into saliency heatmaps. This should
    require a sequence of per-class confidences predicted on the
    reference image, a number of per-class confidences as predicted
    on perturbed images, as well as the masks of the reference image
    perturbations (as would be output from a
    `PerturbImage` implementation.
    """
    @abc.abstractmethod
    def generate(
            self,
            image_conf: np.ndarray,
            perturbed_conf: np.ndarray,
            perturbed_masks: np.ndarray,
        ) -> np.ndarray:
        """
        Generate an image saliency heat-map matrix given a the original classifier
        confidence on image, confidences on pertrubed images and finally the masks
        used to pertrube the image.

        :param image_conf:
            Numpy Original classifier confidence on Image for all classes
            that require saliency map generation: [nClasses], float, [0, 1] range
        :param perturbed_conf:
            Numpy Classifier confidences on pertrubed version of Image
            for nClasses: [nMasks x nClasses], float, [0, 1] range
        :param perturbed_masks:
            Numpy Array of masks used to pertrube Image in the same order
            as their their respective 'pertutbed_conf's: [nMasks x H x W],
            float, [0, 1] range.

        :return: Numpy array of heat-map matrices for each input class in nClasses.

            :py:class:`numpy.ndarray` matrix of the shape [nClasses, height, width]
            shape as the input image matrix but of floating-point type within
            the range of [0,1], where areas of higher value represent more
            salient regions according to the classifier that generated confidence
            values.
        """

