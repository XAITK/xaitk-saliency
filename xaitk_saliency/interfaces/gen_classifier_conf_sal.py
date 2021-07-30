import abc

import numpy as np
from smqtk_core import Plugfigurable


class GenerateClassifierConfidenceSaliency(Plugfigurable):
    """
    Visual saliency map generation interface whose implementations transform
    black-box image classification scores into saliency heatmaps.

    This should require a sequence of per-class confidences predicted on the
    reference image, a number of per-class confidences as predicted on
    perturbed images, as well as the masks of the reference image perturbations
    (as would be output from a
    :class:`xaitk_saliency.interfaces.perturb_image.PerturbImage`
    implementation).

    Implementations should use this input to generate a visual saliency
    heat-map for each input “class” in the input.
    This is both an effort to vectorize the operation for optimal performance,
    as well as to allow some algorithms to take advantage of differences in
    classification behavior for other classes to influence heatmap generation.
    For classifiers that generate many class label predictions, it is intended
    that only a subset of relevant class predictions need be provided here if
    computational performance is a consideration.
    """

    @abc.abstractmethod
    def generate(
        self,
        image_conf: np.ndarray,
        perturbed_conf: np.ndarray,
        perturbed_masks: np.ndarray,
    ) -> np.ndarray:
        """
        Generate an visual saliency heat-map matrix given the black-box
        classifier output on a reference image, the same classifier output on
        perturbed images and the masks of the visual perturbations.

        Perturbation mask input into the `perturbed_masks` parameter here is
        equivalent to the perturbation mask output from a
        :meth:`xaitk_saliency.interfaces.perturb_image.PerturbImage.perturb`
        method implementation.
        These should have the shape `[nMasks x H x W]`, and values in range
        [0, 1], where a value closer to 1 indicate areas of the image that
        are *unperturbed*.
        Note the type of values in masks can be either integer, floating point
        or boolean within the above range definition.
        Implementations are responsible for handling these expected variations.

        Generated saliency heat-map matrices should be floating-point typed and
        be composed of values in the [-1,1] range.
        Positive values of the saliency heat-maps indicate regions which increase
        class confidence scores, while negative values indicate regions which
        decrease class confidence scores according to the model that generated
        input confidence values.

        :param image_conf:
            Reference image predicted class-confidence vector, as a
            `numpy.ndarray`, for all classes that require saliency map
            generation.
            This should have a shape `[nClasses]`, be float-typed and with
            values in the [0,1] range.
        :param perturbed_conf:
            Perturbed image predicted class confidence matrix.
            Classes represented in this matrix should be congruent to classes
            represented in the `image_conf` vector.
            This should have a shape `[nMasks x nClasses]`, be float-typed and
            with values in the [0,1] range.
        :param perturbed_masks:
            Perturbation masks `numpy.ndarray` over the reference image.
            This should be parallel in association to the classification
            results input into the `perturbed_conf` parameter.
            This should have a shape `[nMasks x H x W]`, and values in range
            [0, 1], where a value closer to 1 indicate areas of the image that
            are *unperturbed*.

        :return: Generated visual saliency heat-map for each input class as a
            float-type `numpy.ndarray` of shape `[nClasses x H x W]`.
        """

    def __call__(
        self,
        image_conf: np.ndarray,
        perturbed_conf: np.ndarray,
        perturbed_masks: np.ndarray,
    ) -> np.ndarray:
        """
        Alias for :meth:`.GenerateClassifierConfidenceSaliency.generate`.
        """
        return self.generate(image_conf, perturbed_conf, perturbed_masks)
