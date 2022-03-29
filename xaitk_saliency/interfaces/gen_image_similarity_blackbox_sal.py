import abc
import numpy as np

from smqtk_core import Plugfigurable
from smqtk_descriptors.interfaces.image_descriptor_generator import ImageDescriptorGenerator

from xaitk_saliency.exceptions import ShapeMismatchError, UnexpectedDimensionsError


class GenerateImageSimilarityBlackboxSaliency (Plugfigurable):
    """
    This interface describes the generation of visual saliency heatmaps based on
    the similarity of two images.
    Similarity is deduced from the output of a black-box image feature vector
    generator that transforms each image to an embedding space.

    The resulting saliency map is relative to the query image.
    As such, it denotes regions in the query image that make it more or less
    similar to the other image, called the reference image.

    The `smqtk_descriptors.ImageDescriptorGenerator` interface is used to
    provide a common format for image feature vector generation.
    """

    def generate(
        self,
        ref_image: np.ndarray,
        query_image: np.ndarray,
        blackbox: ImageDescriptorGenerator
    ) -> np.ndarray:
        """
        Generates visual saliency map based on the similarity of two images
        determined by the output of the blackbox feature vector generator.

        The input reference and query image are expected to be matrices in
        either a `H x W` or `H x W x C` shape format.

        The output saliency map matrix should be (1) equal in height and width
        to the query image, (2) floating-point typed, and (3) composed of values
        in the `[-1, 1]` range.

        The `(0, 1]` range is intended to describe regions that are positively
        salient, and the `[-1, 0)` range is intended to describe regions that
        are negatively salient.
        Positive values of the saliency heatmaps indicate regions of the query
        image that increase its similarity to the reference image, while
        negative values indicate regions that actively decrease its similarity
        to the reference image.

        Similarity is determined by the output of the feature vector generator
        and implementation specifics.

        :param ref_image: Reference image to compare the `query_image` to.
        :param query_image: Image to compute saliency for.
        :param blackbox: Black-box image feature vector generator.

        :raises ShapeMismatchError: The implementation's resulting heatmap
            matrix did not have matching height and width components to the
            query image.

        :return: A single visual saliency heatmap relative to the query image.
        """
        if ref_image.ndim not in (2, 3):
            raise ValueError(f"Input reference image matrix has an unexpected number of dimensions: {ref_image.ndim}")
        if query_image.ndim not in (2, 3):
            raise ValueError(f"Input query image matrix has an unexpected number of dimensions: {query_image.ndim}")

        output = self._generate(ref_image, query_image, blackbox)

        if output.ndim != 2:
            raise UnexpectedDimensionsError(
                f"Expected output to be a 2D heatmap matrix but got matrix with shape: {output.shape}"
            )

        if output.shape != query_image.shape[:2]:
            raise ShapeMismatchError(
                f"Output saliency heatmaps did not have matching height and "
                f"width shape components: (query) {query_image.shape[:2]} != "
                f"{output.shape} (output)"
            )

        return output

    def __call__(
        self,
        ref_image: np.ndarray,
        query_image: np.ndarray,
        blackbox: ImageDescriptorGenerator
    ) -> np.ndarray:
        """
        Alias to :meth:`generate` method.
        See :meth:`generate` for details.
        """
        return self.generate(ref_image, query_image, blackbox)

    @abc.abstractmethod
    def _generate(
        self,
        ref_image: np.ndarray,
        query_image: np.ndarray,
        blackbox: ImageDescriptorGenerator
    ) -> np.ndarray:
        """
        Internal method for implementing the generation logic.
        This is invoked by the above `generate` method as a template method.

        The doc-string for `generate` also applies here aside from the
        `ShapeMismatchError` which is specific to `generate`.
        Due to the input value checks done in `generate` we can assume that the
        inputs to this method conform to the constraints defined by the
        interface.

        :param ref_image: Reference image to compare the `query_image` to.
        :param query_image: Image to compute saliency for.
        :param blackbox: Black-box image feature vector generator.

        :return: A single visual saliency heatmap relative to the query image.
        """
