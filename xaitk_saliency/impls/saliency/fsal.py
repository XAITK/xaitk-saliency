from xaitk_saliency import ImageClassifierSaliencyMapGenerator
from xaitk_saliency.utils.masking import (
    generate_masked_images,
    weight_regions_by_scalar
)
import numpy as np


class Fsal (ImageClassifierSaliencyMapGenerator):
    """
    This saliency implementation transforms black-box
    image classification scores into saliency heatmaps. This should
    require a sequence of per-class confidences predicted on the
    reference image, a number of per-class confidences as predicted
    on perturbed images, as well as the masks of the reference image
    perturbations (as would be output from a 
    `PerturbImage` implementation.
    """

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

        sal = np.empty((len(image_conf), *perturbed_masks.shape[-2:]))
        for i, base_conf in enumerate(image_conf):
            diff = np.empty(len(perturbed_conf[:, i]))
            for j in range(len(perturbed_conf[:, i])):
                diff[i]= perturbed_conf[j, i] - base_conf
            sal[i] = weight_regions_by_scalar(diff, perturbed_masks)
        return sal

    def get_config(self):
        return {}