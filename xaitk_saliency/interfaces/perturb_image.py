import abc

import numpy as np
from smqtk_core import Plugfigurable


class PerturbImage (Plugfigurable):
    """
    Interface abstracting the behavior of taking a reference image and
    generating some number perturbations in the form of mask matrices
    indicating where perturbations should occur and to what amount.

    Implementations should impart no side effects upon the input image.
    """

    @abc.abstractmethod
    def perturb(
        self,
        ref_image: np.ndarray
    ) -> np.ndarray:
        """
        Transform an input reference image into a number of mask matrices
        indicating the perturbed regions.

        Output mask matrix should be three-dimensional with the format
        [nMasks x Height x Width], sharing the same height and width to the
        input reference image.
        The implementing algorithm may determine the quantity of output masks
        per input image.
        These masks should indicate the regions in the corresponding perturbed
        image that have been modified.
        Values should be in the [0, 1] range, where a value closer to 1.0
        indicates areas of the image that are unperturbed.
        Note that output mask matrices may be of a floating-point type
        to allow for fractional perturbation.

        :param ref_image:
            Reference image to generate perturbations from.
        :return: Mask matrix with shape [nMasks x Height x Width].
        """

    def __call__(
        self,
        ref_image: np.ndarray
    ) -> np.ndarray:
        """
        Alias for :meth:`.PerturbImage.perturb`.
        """
        return self.perturb(ref_image)
