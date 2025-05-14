"""This module provides the `parse_coco_dset` function to load a COCO dataset for `xaitk-saliency`."""

import logging
from collections.abc import Generator

import numpy as np
from PIL import Image  # type: ignore

from xaitk_saliency.exceptions import KWCocoImportError

try:
    import kwcoco  # type: ignore

    kwcoco_available = True
except ImportError:
    kwcoco_available = False

LOG = logging.getLogger(__name__)


class KWCocoUtils:
    """Class for KWCoco Utility functions"""

    def __init__(self) -> None:
        """Initialize KWCocoUtils"""
        if not self.is_usable():
            raise KWCocoImportError

    def parse_coco_dset(
        self,
        dets_dset: "kwcoco.CocoDataset",
    ) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Generate reference image, bounding box, and class score matrices, for
        use with an implementation of `GenerateObjectDetectorBlackboxSaliency`,
        from a `kwcoco.CocoDataset` object.
        Matrices are generated for each image in the dataset that has
        annotations associated with it.

        The output bounding boxes matrices are each of shape [nDets x 4].
        The output score matrices are each of shape [nDets x nClasses].
        The value of nClasses is extrapolated from the minimum and maximum
        category id values present in the dataset.

        :param dets_dset: `kwcoco.CocoDataset` object to parse.

        :return: A generator of reference images, corresponding bounding boxes,
            and scores as would be input to an implementation of
            `GenerateObjectDetectorBlackboxSaliency`.
        """
        # filter out images with no detections
        gid_to_aids = {key: value for (key, value) in dets_dset.gid_to_aids.items() if len(value) > 0}

        cids = [cat["id"] for cat in dets_dset.cats.values()]
        min_cid = min(cids)
        num_classes = max(cids) - min_cid + 1

        for _img_i, (img_id, det_ids) in enumerate(gid_to_aids.items()):
            bboxes = np.empty((0, 4))
            scores = np.empty((0, num_classes))

            for det_id in det_ids:
                ann = dets_dset.anns[det_id]

                if "score" in ann:
                    score_array = np.zeros(num_classes)
                    score_array[ann["category_id"] - min_cid] = ann["score"]
                elif "prob" in ann:
                    score_array = ann["prob"]
                # annotation has no scores
                else:
                    # do not include any non-prediction annotations
                    continue  # pragma: no cover

                scores = np.vstack((scores, score_array))

                x, y, w, h = ann["bbox"]
                bbox = [x, y, x + w, y + h]
                bboxes = np.vstack((bboxes, bbox))

            img_file = dets_dset.get_image_fpath(img_id)
            ref_img = np.asarray(Image.open(img_file))

            yield ref_img, bboxes, scores

    @classmethod
    def is_usable(cls) -> bool:
        """
        Checks if the necessary dependencies (KWCoco) are available.

        Returns:
            bool: True if KWCoco is available; False otherwise.
        """
        return kwcoco_available
