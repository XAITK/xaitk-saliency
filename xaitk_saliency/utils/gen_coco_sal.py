import numpy as np
from typing import Optional, Dict, Callable, Any, Iterable, List, Hashable, Tuple, KeysView, Union, Sequence
import logging
from PIL import Image  # type: ignore

from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects
from smqtk_detection.utils.bbox import AxisAlignedBoundingBox

from xaitk_saliency import GenerateDetectorProposalSaliency, PerturbImage
from xaitk_saliency.utils.masking import occlude_image_batch
from xaitk_saliency.utils.detection import format_detection
from xaitk_saliency.exceptions import MismatchedLabelsError

try:
    import kwcoco  # type: ignore
    is_usable = True
except ModuleNotFoundError:
    is_usable = False


LOG = logging.getLogger(__name__)


# mute logger to suppress verbose messages
def _mute_log(_: Any) -> None: ...


if not is_usable:
    LOG.warning(f"{__name__} requires additional dependencies, please install 'xaitk-saliency[tools]'")
else:

    def gen_coco_sal(
        dets_dset: kwcoco.CocoDataset,
        detector: DetectImageObjects,
        img_perturber: PerturbImage,
        sal_generator: GenerateDetectorProposalSaliency,
        fill: Optional[Union[int, Sequence[int], np.ndarray]] = None,
        verbose: Optional[bool] = False,
    ) -> Dict[int, np.ndarray]:
        """
        Generate saliency maps for every detection in a ``kwcoco`` dataset
        object. Detections generation, image perturbation, and saliency map
        creation is performed by the implemenations of ``DetectImageObjects``,
        ``PerturbImage``, and ``GenerateDetectorProposalSaliency`` that are
        passed, respectively.

        The resulting saliency maps are organized in a dictionary with
        detection id key values, taken from the kwcoco dataset. Each map has a
        shape of [imgHeight, imgWidth], reffering to the detection's
        corresponding image.

        :param dets_dset: ``kwcoco`` dataset object containing detections to
            generate saliency maps for.
        :param detector: Implementation of ``DetectImageObjects`` to generate
            detections with.
        :param img_perturber: Implementation of ``PerturbImage`` to use.
        :param sal_generator: Implementation of
            ``GenerateDetectorProposalSaliency`` to use.
        :param fill: Optional fill for alpha-blending based on the input masks
            for the occluded regions as a scalar value, a per-channel sequence
            or a shape-matched image. If a multi-channel value is used, all the
            images in ``dets_dset`` must have that number of channels.
            Similarly, if a shape-matched image is used, it must match the
            shape of each image in ``dets_dset``.
        :param verbose: Option to print out progress statements. Default is
            false.

        :returns: Dictionary of saliency maps. Keys of this dictionary are the
            ids of the corresponding detections taken from the ``kwcoco``
            dataset. Each saliency map has shape [imgHeight, imgWidth],
            reffering to the detection's corresponding image.
        """

        if verbose:
            _log: Callable[[str], None] = LOG.info
        else:
            _log = _mute_log

        # filter out images with no detections
        gid_to_aids = {key: value for (key, value) in dets_dset.gid_to_aids.items() if len(value) > 0}

        num_imgs = len(gid_to_aids)

        sal_maps = {}
        for img_i, (img_id, det_ids) in enumerate(gid_to_aids.items()):

            num_dets = len(det_ids)

            img_file = dets_dset.get_image_fpath(img_id)
            ref_img = np.asarray(Image.open(img_file))

            ref_dets_mat = _coco_dets_to_formatted_mat(dets_dset, det_ids)

            _log(f'[{img_id}] ({img_i+1}/{num_imgs}) {num_dets} reference detections found.')

            _log(f'[{img_id}] ({img_i+1}/{num_imgs}) Perturbing image.')

            pert_masks = img_perturber(ref_img)
            pert_imgs = occlude_image_batch(ref_img, pert_masks, fill=fill)

            _log(f'[{img_id}] ({img_i+1}/{num_imgs}) Getting perturbation detections.')

            pert_dets_iter = detector(pert_imgs)

            _log(f'[{img_id}] ({img_i+1}/{num_imgs}) Generating saliency maps.')

            pert_dets_mat = _dets_to_formatted_mat(pert_dets_iter, dets_dset.cats)

            img_sal_maps = sal_generator(ref_dets_mat, pert_dets_mat, pert_masks)

            for i, det_id in enumerate(det_ids):
                sal_maps[det_id] = img_sal_maps[i]

        return sal_maps

    def _name_to_cat_id(
        name: str,
        cats: dict
    ) -> int:
        """
        Converts category name to id.

        :param name: Category name.
        :param cats: Dictionary of categories, as taken from the "categories"
            section of a COCO style annotation dictionary.
        :returns: Id of matching category, or -1 if no matching category is
            found.
        """
        for cat_id, cat in cats.items():
            if name == cat['name']:
                return cat_id

        return -1

    def _map_score_dict_keys(
        score_keys: KeysView,
        coco_cats: dict
    ) -> Dict[Hashable, int]:
        """
        Matches score dict keys, as returned by an implementation of
        ``DetectImageObjects``, with categories from a COCO dataset.

        :param score_keys: List of key values from score dict, as returned by
            an implementation of ``DetectImageObjects``.
        :param coco_cats: Dict of categories, as taken from the "categories"
            section of a COCO style annotation dictionary.
        :returns: Dictionary mapping score dict key to category id.
        """

        cat_map = {}  # type: Dict[Hashable, int]
        for key in score_keys:
            try:
                # assume key is a cat id
                key_int = int(key)
                if key_int in coco_cats:
                    cat_map[key] = key_int
                else:
                    raise MismatchedLabelsError("Provided detections and detector have mismatched class labels.")
            except ValueError:  # cast to int failed
                # try and match key with category name
                cat_id = _name_to_cat_id(key, coco_cats)
                if(cat_id == -1):
                    raise MismatchedLabelsError("Provided detections and detector have mismatched class labels.")
                cat_map[key] = cat_id

        return cat_map

    def _dets_to_formatted_mat(
        dets: Iterable[Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]],
        cats: dict
    ) -> np.ndarray:
        """
        Converts detections, as returned by an implementation of
        ``DetectImageObjects``, into a detection matrix formatted for use with
        an implementation of ``GenerateDetectorProposalSaliency``. This
        involves matching the labels in the detections with categories from a
        COCO style dataset.

        :param dets: Detections, as returned by an implementation of
            ``DetectImageObjects``.
        :param cats: Dict of categories, as taken from the "categories" section
            of a COCO style annotation dictionary.
        :returns: Matrix of detections with shape
            [nImgs x nDets x (4+1+nClasses)]. If the number of detections for
            each image is not consistent, the matrix will be padded with rows
            of ones, except for the objectiveness which is set to zero.
        """

        max_cat_id = max(cats.keys())

        dets_mat_list = []
        for img_dets in dets:

            img_dets = list(img_dets)

            if len(img_dets) == 0:
                dets_mat_list.append(np.empty([0, 4+1+max_cat_id+1]))
                continue

            bbox_mat = np.asarray([
                [
                    bbox.min_vertex[0],
                    bbox.min_vertex[1],
                    bbox.max_vertex[0],
                    bbox.max_vertex[1]
                ] for bbox in [det[0] for det in img_dets]
            ])

            score_dicts = [det[1] for det in img_dets]

            cat_map = _map_score_dict_keys(score_dicts[0].keys(), cats)

            # assume class ids are arranged between 0 and the maximum id
            scores_mat = np.empty([0, max_cat_id+1])
            objectness = np.ones(len(img_dets))

            for det_i, score_dict in enumerate(score_dicts):
                score_array = np.zeros(max_cat_id+1)

                for key, score in score_dict.items():
                    score_array[cat_map[key]] = score

                # detect one-hot
                if len([score for score in score_array if score > 0]) == 1:
                    score = max(score_array)
                    objectness[det_i] = score
                    score_array /= score

                scores_mat = np.vstack((scores_mat, score_array))

            dets_mat_list.append(format_detection(bbox_mat, scores_mat, objectness))

        # pad matrices
        num_dets = [dets_mat.shape[0] for dets_mat in dets_mat_list]
        max_dets = max(num_dets)
        pad_row = np.ones(4+1+max_cat_id+1)
        pad_row[4] = 0
        for i, dets_mat in enumerate(dets_mat_list):
            size_diff = max_dets - dets_mat.shape[0]
            if size_diff > 0:
                dets_mat_list[i] = np.vstack((dets_mat, np.tile(pad_row, (size_diff, 1))))

        return np.asarray(dets_mat_list)

    def _coco_dets_to_formatted_mat(
        coco_dset: kwcoco.CocoDataset,
        aids: List[int]
    ) -> np.ndarray:
        """
        Converts detections in a COCO dataset to a matrix formatted for use
        with an implementation of ``GenerateDetectorProposalSaliency``.

        :param coco_dset: kwcoco dataset object.
        :param aids: Annotation ids of detections to include.
        :returns: Matrix of detections with shape [nDets x (4+1+nClasses)]
        """

        dets = [coco_dset.anns[det_id] for det_id in aids]

        bbox_mat = np.asarray([
            [
                det['bbox'][0],
                det['bbox'][1],
                det['bbox'][0] + det['bbox'][2],
                det['bbox'][1] + det['bbox'][3],
            ] for det in dets
        ])

        # assume class ids are arranged between 0 and the maximum id
        max_cat_id = max(coco_dset.cats.keys())
        class_mat = np.zeros((len(dets), max_cat_id+1))
        objectness = np.ones(len(dets))

        for i, det in enumerate(dets):

            # look for class probabilities
            if 'prob' in det:
                class_mat[i, :] = det['prob']

            # assume single score is present
            else:
                class_mat[i, det['category_id']] = 1
                objectness[i] = det['score']

        return format_detection(bbox_mat, class_mat, objectness)
