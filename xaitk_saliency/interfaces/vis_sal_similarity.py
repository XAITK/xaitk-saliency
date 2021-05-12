import abc

import numpy as np
from smqtk_core import Plugfigurable


class ImageSimilaritySaliencyMapGenerator (Plugfigurable):
    """
    Visual saliency map generation interface whose implementations transform
    black-box feature-vectors from multiple references and perturbations into
    a saliency heat-maps.

    This transformation requires two reference images, translated into
    feature-vectors via some black-box means, between which we are trying to
    discern the feature-space saliency.
    This also requires the feature-vectors for perturbed images as well as the
    masks of the perturbations as would be output from a
    :class:`xaitk_saliency.interfaces.perturb_image.PerturbImage`
    implementation.
    We expect perturbations to be relative to the second reference image.
    """

    @abc.abstractmethod
    def generate(
        self,
        ref_descr_1: np.ndarray,
        ref_descr_2: np.ndarray,
        perturbed_descrs: np.ndarray,
        perturbed_masks: np.ndarray,
    ) -> np.ndarray:
        """
        Generate a visual saliency heat-map matrix given the black-box
        descriptor generation output on two reference images, the same
        descriptor output on perturbed images and the masks of the visual
        perturbations.

        Perturbation mask input into the `perturbed_masks` parameter here is
        equivalent to the perturbation mask output from a
        :meth:`xaitk_saliency.interfaces.perturb_image.PerturbImage.perturb`
        method implementation.
        We expect perturbations to be relative to the second reference image.

        :param ref_descr_1:
            First image reference float feature-vector, shape `[nFeats]`
        :param ref_descr_2:
            Second image reference float feature-vector, shape `[nFeats]'
        :param perturbed_descrs:
            Feature vectors of second reference image perturbations, float
            typed of shape `[nMasks x nFeats]`.
        :param perturbed_masks:
            Perturbation masks `numpy.ndarray`, float-typed with shape
            `[nMasks x H x W]` in the [0,1] range.
        :return: Generated saliency heat-map as a float-typed `numpy.ndarray`
            with shape `[H x W]`.
        """

    def __call__(
        self,
        ref_descr_1: np.ndarray,
        ref_descr_2: np.ndarray,
        perturbed_descrs: np.ndarray,
        perturbed_masks: np.ndarray,
    ) -> np.ndarray:
        """
        Alias for :meth:`.ImageSimilaritySaliencyMapGenerator.generate`.
        """
        return self.generate(
            ref_descr_1,
            ref_descr_2,
            perturbed_descrs,
            perturbed_masks
        )
