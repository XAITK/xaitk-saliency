import numpy as np
from typing import Tuple, Generator
from PIL import Image  # type: ignore
import logging

try:
    import kwcoco  # type: ignore
    is_usable = True
except ModuleNotFoundError:
    is_usable = False

LOG = logging.getLogger(__name__)


if not is_usable:
    LOG.warning(f"{__name__} requires additional dependencies, please install 'xaitk-saliency[tools]'")
else:

    def parse_coco_dset(
        dets_dset: kwcoco.CocoDataset
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
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

        cids = [cat['id'] for cat in dets_dset.cats.values()]
        min_cid = min(cids)
        num_classes = max(cids) - min_cid + 1

        for img_i, (img_id, det_ids) in enumerate(gid_to_aids.items()):

            bboxes = np.empty((0, 4))
            scores = np.empty((0, num_classes))

            for det_id in det_ids:
                ann = dets_dset.anns[det_id]

                if 'score' in ann:
                    score_array = np.zeros(num_classes)
                    score_array[ann['category_id'] - min_cid] = ann['score']
                elif 'prob' in ann:
                    score_array = ann['prob']
                # annotation has no scores
                else:
                    # do not include any non-prediction annotations
                    continue  # pragma: no cover

                scores = np.vstack((scores, score_array))

                x, y, w, h = ann['bbox']
                bbox = [x, y, x+w, y+h]
                bboxes = np.vstack((bboxes, bbox))

            img_file = dets_dset.get_image_fpath(img_id)
            ref_img = np.asarray(Image.open(img_file))

            yield ref_img, bboxes, scores
