import numpy as np
import abc
from typing import Optional

from smqtk_core import Plugfigurable
from smqtk_detection import DetectImageObjects

from xaitk_saliency.exceptions import ShapeMismatchError


class GenerateObjectDetectorBlackboxSaliency (Plugfigurable):
    """
    This interface describes the generation of visual saliency heatmaps for
    input object detections with respect to a given black box object detection
    and classification model.

    This transformation requires reference detections to focus on explaining,
    and the image those detections were drawn from.
    For compatibility, the input detections specification are split into three
    separate inputs: bounding boxes, scores, and objectness.
    A visual saliency heatmap is generated for each input detection.

    The `smqtk_detection.DetectImageObjects` abstract interface is used to
    provide a common format for a black-box object detector.
    """

    def generate(
        self,
        ref_image: np.ndarray,
        bboxes: np.ndarray,
        scores: np.ndarray,
        blackbox: DetectImageObjects,
        objectness: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate per-detection visual saliency heatmaps for some object
        detector black-box over some input reference detections from some input
        reference image.

        The input reference image is expected to be a matrix in either
        `[H x W]` or `[H x W x C]` shape format.

        The reference detections are represented by three separate inputs:
        bounding boxes, scores, and objectness.
        The input bounding boxes are expected to be a matrix with shape
        `[nDets x 4]` where each row is the bounding box of a single detection
        in xyxy format.
        The input scores are expected to be a matrix with shape
        `[nDets x nClasses]` where each row is the scores for each class for a
        single detection.
        The order of each class score should match the order returned by the
        input black-box algorithm.
        The optional input objectness is expected to be a vector of length
        `nDets` containing the objectness score (single float value) for each
        reference detection.
        If this is not provided, it is assumed that each detection has an
        objectness score of 1.

        If your detections consist of a single class prediction and confidence
        score instead of scores for each class, it is best practice to replace
        the objectness score with the confidence score and use a one-hot
        encoding of the prediction as the class scores.

        The output saliency map matrix should be (1) in the shape
        `[nDets x H x W]` where `H` and `W` are the height and width
        respectively of the input reference image, (2) floating-point typed,
        and (3) composed of values in the `[-1, 1]` range.

        The `(0, 1]` range is intended to describe regions that are positively
        salient, and the `[-1, 0)` range is intended to describe regions that
        are negatively salient.
        Positive values of the saliency heatmaps indicate regions that increase
        detection locality and class confidence scores, while negative values
        indicate regions that actively decrease detection locality and class
        confidence scores.

        :param ref_image: Reference image that the input reference detections
            belong to.
        :param bboxes: The bounding boxes in xyxy format of the reference
            detections to generate visual saliency maps for.
            This should be a matrix with shape `[nDets x 4]`.
        :param scores: The class scores of the reference detections to generate
            visual saliency maps for.
            This should be a matrix with shape `[nDets x nClasses]`.
            The order of the scores should match that returned by the input
            black-box detection algorithm.
        :param blackbox: The black-box object detector to perform arbitrary
            operations on in order to deduce visual saliency.
        :param objectness: Optional objectness score for each reference
            detection.
            This should be a vector of length `nDets`.
            If not provided, it is assumed that each detection has an
            objectness score of 1.

        :raises ValueError: The input reference image had an unexpected number
            of dimensions.

        :raises ValueError: The input bounding boxes had a width other than 4.

        :raises ValueError: The input bounding boxes, scores, and/or objectness
            scores do not match in quantity.

        :raises ShapeMismatchError: The implementation result visual saliency
            heatmap matrix did not have matching height and width components to
            the reference image.

        :raises ShapeMismatchError: The quantity of resulting heatmaps did not
            match the quantity of input reference detections.

        :return: A number of visual saliency heatmaps, one for each input
            reference detection.
            This is a single matrix of shape `[nDets x H x W]` where `H` and
            `W` are the height and width respectively of the input reference
            image.
        """

        if ref_image.ndim not in (2, 3):
            raise ValueError(f"Input image matrix has an unexpected number of dimensions: {ref_image.ndim}")

        if bboxes.shape[1] != 4:
            raise ValueError(f"Input bounding boxes matrix has width of {bboxes.shape[1]}, should have width of 4")

        # Check that reference detection inputs have matching shapes
        if objectness is None:
            if len(bboxes) != len(scores):
                raise ValueError(
                    f"Number of input bounding boxes and scores do not match: "
                    f"(bboxes) {len(bboxes)} != {len(scores)} (scores)"
                )
        else:
            if len(bboxes) != len(scores) or len(bboxes) != len(objectness):
                raise ValueError(
                    f"Number of input bounding boxes, scores, and objectness "
                    f"scores do not match: (bboxes) {len(bboxes)} != "
                    f"{len(scores)} (scores) and/or (bboxes) {len(bboxes)} != "
                    f"{len(objectness)} (objectness)"
                )

        output = self._generate(
            ref_image,
            bboxes,
            scores,
            blackbox,
            objectness,
        )

        # Check that the saliency heatmaps' shape matches the reference image.
        if output.shape[1:] != ref_image.shape[:2]:
            raise ShapeMismatchError(
                f"Output saliency heatmaps did not have matching height and "
                f"width shape components: "
                f"(ref) {ref_image.shape[:2]} != {output.shape[1:]} (output)"
            )

        # Check that the quantity of output heatmaps matches the quantity of reference detections input
        if len(output) != len(bboxes):
            raise ShapeMismatchError(
                f"Quantity of output heatmaps does not match the quantity of "
                f"input reference detections: (input) {len(bboxes)} != "
                f"{len(output)} (output)"
            )

        return output

    def __call__(
        self,
        ref_image: np.ndarray,
        bboxes: np.ndarray,
        scores: np.ndarray,
        blackbox: DetectImageObjects,
        objectness: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Alias to the :meth:`generate` method.
        See :meth:`generate` for more details.
        """
        return self.generate(
            ref_image,
            bboxes,
            scores,
            blackbox,
            objectness,
        )

    @abc.abstractmethod
    def _generate(
        self,
        ref_image: np.ndarray,
        bboxes: np.ndarray,
        scores: np.ndarray,
        blackbox: DetectImageObjects,
        objectness: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Internal method for implementing the generation logic.
        This is invoked by the above `generate` method as a template method.

        The doc-string for the `generate` method also applies here aside from
        the `ShapeMismatchError` and `ValueError` exceptions which are specific
        to `generate`.
        Given the `ValueError` checks performed in `generate`, we can assume
        that the input parameters are formatted correctly here.

        :param ref_image: Reference image that the input reference detections
            belong to.
        :param bboxes: The bounding boxes in xyxy format of the reference
            detections to generate visual saliency maps for.
            This should be a matrix with shape `[nDets x 4]`.
        :param scores: The class scores of the reference detections to generate
            visual saliency maps for.
            This should be a matrix with shape `[nDets x nClasses]`.
            The order of the scores should match that returned by the input
            black-box detection algorithm.
        :param blackbox: The black-box object detector to perform arbitrary
            operations on in order to deduce visual saliency.
        :param objectness: Optional objectness score for each reference
            detection.
            This should be a vector of length `nDets`.
            If not provided, it is assumed that each detection has an
            objectness score of 1.

        :return: A number of visual saliency heatmaps, one for each input
            reference detection.
            This is a single matrix of shape `[nDets x H x W]` where `H` and
            `W` are the height and width respectively of the input reference
            image.
        """
