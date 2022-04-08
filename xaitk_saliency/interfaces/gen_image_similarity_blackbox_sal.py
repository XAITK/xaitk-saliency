import abc
import numpy as np
from typing import Sequence

from smqtk_core import Plugfigurable
from smqtk_descriptors.interfaces.image_descriptor_generator import ImageDescriptorGenerator

from xaitk_saliency.exceptions import ShapeMismatchError


class GenerateImageSimilarityBlackboxSaliency (Plugfigurable):
    """
    This interface describes the generation of visual saliency heatmaps based on
    the similarity of a reference image to a number of query images.
    Similarity is deduced from the output of a black-box image feature vector
    generator that transforms each image to an embedding space.

    The resulting saliency maps are relative to the reference image.
    As such, each map denotes regions in the reference image that make it more
    or less similar to the corresponding query image.

    The `smqtk_descriptors.ImageDescriptorGenerator` interface is used to
    provide a common format for image feature vector generation.
    """

    def generate(
        self,
        ref_image: np.ndarray,
        query_images: Sequence[np.ndarray],
        blackbox: ImageDescriptorGenerator
    ) -> np.ndarray:
        """
        Generates visual saliency maps based on the similarity of the reference
        image to each query image determined by the output of the blackbox
        feature vector generator.

        The input reference image is expected to be a matrix in
        either a `H x W` or `H x W x C` shape format.
        The input query images should be a sequence of matrices, each of which
        should also be in either `H x W` or `H x W x C` format.
        Each query image is not required to have the same shape, however.

        The output saliency map matrix should be (1) of shape
        `nQueryImgs x H x W` with matching height and width to the reference
        image, (2) floating-point typed, and (3) composed of values in the
        `[-1, 1]` range.

        The `(0, 1]` range is intended to describe regions that are positively
        salient, and the `[-1, 0)` range is intended to describe regions that
        are negatively salient.
        Positive values of each saliency heatmap indicate regions of the
        reference image that increase its similarity to the corresponding query
        image, while negative values indicate regions that actively decrease its
        similarity to the corresponding query image.

        Similarity is determined by the output of the feature vector generator
        and implementation specifics.

        :param ref_image: Reference image to compute saliency for.
        :param query_images: Query images to compare the reference image to.
        :param blackbox: Black-box image feature vector generator.

        :raises ShapeMismatchError: The implementation's resulting heatmap
            matrix did not have matching height and width components to the
            reference image.

        :return: A matrix of saliency heatmaps relative to the reference image
            with shape `nQueryImgs x H x W`.
        """
        if ref_image.ndim not in (2, 3):
            raise ValueError(f"Input reference image matrix has an unexpected number of dimensions: {ref_image.ndim}")

        output = self._generate(ref_image, query_images, blackbox)

        if output.shape[1:] != ref_image.shape[:2]:
            raise ShapeMismatchError(
                f"Output saliency heatmaps did not have matching height and "
                f"width shape components: (reference) {ref_image.shape[:2]} != "
                f"{output.shape[1:]} (output)"
            )

        if len(output) != len(query_images):
            raise ShapeMismatchError(
                f"Number of output saliency heatmaps did not match number of "
                f"input query images: (heatmaps) {len(output)} != "
                f"{len(query_images)} (query images)"
            )

        return output

    def __call__(
        self,
        ref_image: np.ndarray,
        query_images: Sequence[np.ndarray],
        blackbox: ImageDescriptorGenerator
    ) -> np.ndarray:
        """
        Alias to :meth:`generate` method.
        See :meth:`generate` for details.
        """
        return self.generate(ref_image, query_images, blackbox)

    @abc.abstractmethod
    def _generate(
        self,
        ref_image: np.ndarray,
        query_images: Sequence[np.ndarray],
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

        :param ref_image: Reference image to compute saliency for.
        :param query_images: Query images to compare the reference image to.
        :param blackbox: Black-box image feature vector generator.

        :return: A matrix of saliency heatmaps relative to the reference image
            with shape `nQueryImgs x H x W`.
        """
