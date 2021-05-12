import abc
from typing import List, Tuple

import numpy as np
import PIL.Image
from smqtk_core import Plugfigurable


class PerturbImage (Plugfigurable):
    """
    Interface abstracting the behavior of taking a reference image and
    generating some number perturbations of the image along with paired mask
    matrices indicating where perturbations have occurred and to what amount.

    Implementations should impart no side effects upon the input image.
    """

    @abc.abstractmethod
    def perturb(
        self,
        ref_image: PIL.Image.Image
    ) -> Tuple[List[PIL.Image.Image], np.ndarray]:
        """
        Transform an input reference image into a number of perturbed
        variations along with mask matrices indicating the perturbed regions.

        Output images should have the same shape as the input reference image,
        including channels.

        Output mask matrices should be congruent in length to the number of
        perturbed images output, as well as share the same height and width
        dimensions.
        These masks should indicate the regions in the corresponding perturbed
        image that has been modified.
        Values should be in the [0, 1] range, where a value closer to 1.0
        indicate areas of the image that are *unperturbed*.
        Note that output mask matrices *may be* of a floating-point type in
        order to allow for fractional perturbation.

        :param ref_image:
            Reference image to generate perturbations from.
        :return: Tuple of perturbed images and the masks detailing perturbation
            areas.
        """

    def __call__(
        self,
        ref_image: PIL.Image.Image
    ) -> Tuple[List[PIL.Image.Image], np.ndarray]:
        """
        Alias for :meth:`.PerturbImage.perturb`.
        """
        return self.perturb(ref_image)
