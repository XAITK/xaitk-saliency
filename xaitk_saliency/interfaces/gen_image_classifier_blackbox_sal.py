import abc

import numpy as np
from smqtk_classifier import ClassifyImage
from smqtk_core import Plugfigurable

from xaitk_saliency.exceptions import ShapeMismatchError


class GenerateImageClassifierBlackboxSaliency (Plugfigurable):
    """
    Interface for algorithms that take a reference image and a image classifier
    blackbox algorithm, and generates a number of visual saliency heatmap
    matrices, one for each class output by the classifier blackbox.

    An input classifier blackbox needs to be input, which requires some
    specification in how to operate the blackbox.
    The `smqtk_classifier.ClassifyImage` abstract interface is utilized for
    this purpose in order to provide some minimal form that a blackbox
    classifier must have: be able to classify an image into confidences for
    some number of class labels.

    Generated visual saliency heat-map for each input class as a
    float-type `numpy.ndarray` of shape `[nClasses x H x W]`.
    """

    def generate(
        self,
        ref_image: np.ndarray,
        blackbox: ClassifyImage
    ) -> np.ndarray:
        """
        Generate per-class visual saliency heatmaps for some classifier
        blackbox over some image of interest.

        The input reference image is expected to be in matrix form and be in
        either a `H x W` or `H x W x C` shape format.

        Output saliency map matrix should be in the shape `nClasses x H x W`,
        be floating-point typed and be composed of values in the `[-1, 1]`
        range.
        `nClasses` should be the quantity of unique class labels output by the
        given classifier blackbox.
        While specific algorithms determine the quantity of heatmaps returned
        the height and width of returned heatmaps should be consistent with the
        input image, i.e. the `H` and `W` dimensions should match in size to
        the reference image's `H` and `W` dimensions.
        Positive values of the saliency heat-maps indicate regions which
        increase respective class confidence scores, while negative values
        indicate regions which decrease respective class confidence scores
        according to the given blackbox classifier.

        :param ref_image: Reference image over which visual saliency heatmaps
            will be generated.
        :param blackbox: The blackbox classifier handle to perform arbitrary
            operations on in order to deduce visual saliency.

        :raises ShapeMismatchError: The implementation result visual saliency
            heatmap matrix did not have matching height and width components to
            the reference image.

        :return: A number of visual saliency heatmaps equivalent in number to
            the quantity of class labels output by the blackbox classifier.
        """
        # This is a template method that enforces standard input and result
        # checks.
        if ref_image.ndim not in (2, 3):
            raise ValueError(f"Input image matrix has an unexpected number of dimensions: {ref_image.ndim}")
        output = self._generate(ref_image, blackbox)
        # Check that the saliency heatmaps' shape matches the reference image.
        if output.shape[1:] != ref_image.shape[:2]:
            raise ShapeMismatchError(
                f"Output saliency heatmaps did not have matching height and "
                f"width shape components: "
                f"(ref) {ref_image.shape[:2]} != {output.shape[1:]} (output)"
            )
        return output

    def __call__(
        self,
        ref_image: np.ndarray,
        blackbox: ClassifyImage
    ) -> np.ndarray:
        """
        Alias to the :meth:`generate` method.
        See :meth:`generate` for more details.
        """
        return self.generate(ref_image, blackbox)

    @abc.abstractmethod
    def _generate(
        self,
        ref_image: np.ndarray,
        blackbox: ClassifyImage
    ) -> np.ndarray:
        """
        Internal method for implementing the generation logic.
        This is invoked by the above `generate` method as a template method.

        The doc-string for the `generate` method also applies here aside from
        the `ShapeMismatchError` which is specific to `generate`.

        :param ref_image: Reference image over which visual saliency heatmaps
            will be generated.
        :param blackbox: The blackbox classifier handle to perform arbitrary
            operations on in order to deduce visual saliency.

        :return: A number of visual saliency heatmaps equivalent in number to
            the quantity of class labels output by the blackbox classifier.
        """
