import abc

import numpy as np
from smqtk_core import Plugfigurable


class GenerateDetectorProposalSaliency (Plugfigurable):
    """
    This interface proposes that implementations transform black-box image
    object detection predictions into visual saliency heatmaps.
    This should require externally-generated object detection predictions over
    some image, along with predictions for perturbed images and the permutation
    masks for those images as would be output from a
    :class:`xaitk_saliency.interfaces.perturb_image.PerturbImage`
    implementation.

    Object detection representations used here would need to encapsulate
    localization information (i.e. bounding box regions), class scores, and
    objectness scores (if applicable to the detector, such as YOLOv3).
    Object detections are converted into (4+1+nClasses) vectors (4 indices for
    bounding box locations, 1 index for objectness, and nClasses indices for
    different object classes).
    """

    @abc.abstractmethod
    def generate(
        self,
        ref_dets: np.ndarray,
        perturbed_dets: np.ndarray,
        perturb_masks: np.ndarray,
    ) -> np.ndarray:
        """
        Generate visual saliency heat-map matrices for each reference
        detection, describing what visual information contributed to the
        associated reference detection.

        We expect input detections to come from a black-box source that outputs
        our minimum requirements of a bounding-box, objectness and per-class
        scores.
        For detection black-box methods that *do not* produce objectness
        scores, we recommend using a value of 1.0 in that column.
        We assume that an input detection is coupled with a single truth class
        (or a single leaf node in a hierarchical structure).
        Detections input as references may be either ground truth or predicted
        detections.
        As for perturbed image detections input, we expect the quantity of
        detections to be decoupled from the source of reference image
        detections, which is why below we formulate the shape of perturbed
        image detects with `nProps` instead of `nDets`.

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
        object detection scores, while negative values indicate regions which
        decrease object detection scores according to the model that generated
        input object detections.

        :param ref_dets:
            Detections, objectness and class scores on a reference image as a
            float-typed array with shape `[nDets x (4+1+nClasses)]`.
        :param perturbed_dets:
            Object detections, objectness and class scores for perturbed
            variations of the reference image.
            We expect this to be a float-types array with shape
            `[nMasks x nProps x (4+1+nClasses)]`.
        :param perturb_masks:
            Perturbation masks `numpy.ndarray` over the reference image.
            This should be parallel in association to the detection
            propositions input into the `perturbed_dets` parameter.
            This should have a shape `[nMasks x H x W]`, and values in range
            [0, 1], where a value closer to 1 indicate areas of the image that
            are *unperturbed*.
        :return:
            A visual saliency heat-map matrix describing each input reference
            detection. These will be float-typed arrays with shape
            `[nDets x H x W]`.
        """

    def __call__(
        self,
        ref_dets: np.ndarray,
        perturbed_dets: np.ndarray,
        perturb_masks: np.ndarray,
    ) -> np.ndarray:
        """
        Alias for :meth:`.GenerateDetectorProposalSaliency.generate`.
        """
        return self.generate(ref_dets, perturbed_dets, perturb_masks)
