import numpy as np
from typing import Optional, Collection, Iterable, Tuple, Dict, Hashable, Callable, Any
import logging

from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects
from smqtk_detection.utils.bbox import AxisAlignedBoundingBox

from xaitk_saliency.interfaces.gen_detector_prop_sal import GenerateDetectorProposalSaliency
from xaitk_saliency.interfaces.perturb_image import PerturbImage
from xaitk_saliency.utils.masking import occlude_image_batch
from xaitk_saliency.utils.detection import format_detection

LOG = logging.getLogger(__name__)


# mute logger to suppress verbose messages
def _mute_log(_: Any) -> None: ...


def gen_detector_sal(
    ref_imgs: Collection[np.ndarray],
    detector: DetectImageObjects,
    img_perturber: PerturbImage,
    sal_generator: GenerateDetectorProposalSaliency,
    verbose: Optional[bool] = False,
) -> Tuple[Collection[np.ndarray], Collection[np.ndarray]]:
    """
    Generate saliency maps for detections in each of the given images.
    Detections generation, image perturbation, and saliency map creation is
    performed by the implemenations of ``DetectImageObjects``,
    ``PerturbImage``, and ``GenerateDetectorProposalSaliency`` that are passed,
    respectively.

    The resulting saliency maps matrix for each image has a shape of
    [nDets, imgHeight, imgWidth].

    :param ref_imgs: List of images to generate saliency maps for.
    :param detector: Implementation of ``DetectImageObjects`` to generate
        detections with.
    :param img_perturber: Implementation of ``PerturbImage`` to use.
    :param sal_generator: Implementation of
        ``GenerateDetectorProposalSaliency`` to use.
    :param verbose: Option to print out progress statements. Default is false.

    :returns: Pair of parallel lists, the first being the computed saliency
        maps, and the second being the reference detection bounding boxes for
        each image. Each saliency map matrix has shape
        [nDets, imgHeight, imgWidth] and each detections matrix [nDets, 4].
    """

    if verbose:
        _log: Callable[[str], None] = LOG.info
    else:
        _log = _mute_log

    _log("Getting reference detections.")
    ref_dets_iter = detector(ref_imgs)

    # List of saliency maps for all images
    sal_maps_list = []
    ref_dets_list = []

    for i, (ref_img, ref_dets) in enumerate(zip(ref_imgs, ref_dets_iter)):

        ref_dets = list(ref_dets)

        num_dets = len(ref_dets)

        _log(f'[{i+1}/{len(ref_imgs)}] {num_dets} reference detections found.')

        # Return empty list if no reference detections are made
        if num_dets == 0:
            sal_maps_list.append(np.asarray([]).reshape(0, ref_img.shape[0], ref_img.shape[1]))
            ref_dets_list.append(np.asarray([]).reshape(0, 4))
            continue

        _log(f'[{i+1}/{len(ref_imgs)}] Perturbing image.')

        pert_masks = img_perturber(ref_img)
        pert_imgs = occlude_image_batch(ref_img, pert_masks)

        _log(f'[{i+1}/{len(ref_imgs)}] Getting perturbation detections.')

        # Detections on all perturbed images
        pert_dets_iter = detector(pert_imgs)

        _log(f'[{i+1}/{len(ref_imgs)}] Generating saliency maps.')

        # use keys of first reference detection as keys for rest of detections
        class_dicts = [det[1] for det in ref_dets]
        class_keys = list(class_dicts[0].keys())

        ref_dets_mat = _dets_to_formatted_mat(ref_dets, class_keys)

        pert_dets_arrays = [_dets_to_formatted_mat(pert_dets, class_keys) for pert_dets in pert_dets_iter]

        # Use max detections found for a perturbed image as the number of proposals
        num_props = max([array.shape[0] for array in pert_dets_arrays])

        # Row of ones and negative objectiveness to pad proposals with
        pad_prop = np.ones(ref_dets_mat.shape[1])
        pad_prop[4] = -1

        for det_ind, dets_array in enumerate(pert_dets_arrays):
            num_pert_dets = dets_array.shape[0]

            if num_pert_dets < num_props:
                pert_dets_arrays[det_ind] = np.vstack((
                    dets_array,
                    pad_prop.repeat(num_props-num_pert_dets).reshape(num_props-num_pert_dets, -1)
                ))

        pert_dets_mat = np.asarray(pert_dets_arrays)

        sal_maps_list.append(sal_generator.generate(ref_dets_mat, pert_dets_mat, pert_masks))
        ref_dets_list.append(ref_dets_mat[:, :4])

    return (sal_maps_list, ref_dets_list)


def _dets_to_formatted_mat(
    dets: Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]],
    class_keys: list
) -> np.ndarray:
    """
    This helper function takes detections as returned by an implementation
    of `DetectImageObjects` and returns a matrix formatted for use with an
    an implementation of `GenerateDetectorProposalSaliency`.
    """

    dets = list(dets)

    bbox_mat = np.asarray([
        [
            bbox.min_vertex[0],
            bbox.min_vertex[1],
            bbox.max_vertex[0],
            bbox.max_vertex[1]
        ] for bbox in [det[0] for det in dets]
    ])

    class_dicts = [det[1] for det in dets]

    class_mat = np.asarray([
        [class_dict[key] for key in class_keys]
        for class_dict in class_dicts
    ])

    if bbox_mat.shape[0] == 0:
        return np.asarray([]).reshape(0, 4+1+len(class_keys))

    return format_detection(bbox_mat, class_mat)
