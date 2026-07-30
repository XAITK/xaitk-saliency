"""This module provides functionality for generating saliency maps for object detection models.

It uses blackbox saliency generation techniques and integrates with the MAITE dataset and
object detection protocols to produce visual explanations for model predictions.

Functions:
    compute_sal_maps: Generates saliency maps for a given dataset using a blackbox object detector
                      and a saliency map generator.
    sal_on_dets: Facilitates saliency map computation with a MAITE-compatible detector.

Dependencies:
    - numpy: For numerical operations.
    - smqtk_core.configuration: For configuration handling.
    - smqtk_detection.interfaces.detect_image_objects: Interface for object detection in images.
    - xaitk_saliency: For generating saliency maps for object detection.
    - maite.protocols.object_detection: For handling datasets and models in the MAITE framework.
    - xaitk_saliency.interop.object_detection.model: For interoperability with MAITE object detection models.

Usage:
    These functions enable generating saliency maps for datasets using compatible object detection
    models and saliency generators. Saliency maps provide insights into which parts of an image
    influenced the model's predictions.

Example:
    # Generate saliency maps for a dataset using a specific saliency generator and detector.
    saliency_maps, config = sal_on_dets(
        dataset=my_dataset,
        sal_generator=my_sal_generator,
        detector=my_model,
        id_to_name=my_label_mapping,
        img_batch_size=4
    )
"""

from collections.abc import Sequence

import numpy as np
from maite.protocols.object_detection import Dataset, Model
from smqtk_core.configuration import to_config_dict
from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects

from xaitk_saliency.interfaces.gen_object_detector_blackbox_sal import GenerateObjectDetectorBlackboxSaliency
from xaitk_saliency.interop.maite.object_detection.model import MAITEDetector


def compute_sal_maps(
    *,
    dataset: Dataset,
    sal_generator: GenerateObjectDetectorBlackboxSaliency,
    blackbox_detector: DetectImageObjects,
    num_classes: int,
) -> tuple[list[np.ndarray], dict]:
    """Generate saliency maps for the provided dataset.

    :param dataset: MAITE dataset
    :param sal_generator: ``GenerateObjectDetectorBlackboxSaliency`` generator
    :param blackbox_detector: ``DetectImageObjects`` detector
    :param num_classes: Number of classes potentially predicted by the detector.
    """
    img_sal_maps = list()
    for dset_idx in range(len(dataset)):
        ref_img, dets, _ = dataset[dset_idx]

        scores = np.asarray(dets.scores)
        score_matrix = np.zeros((len(scores), num_classes))
        for idx, (lbl, score) in enumerate(zip(np.asarray(dets.labels), scores, strict=False)):
            score_matrix[idx][lbl] = score

        img_sal_maps.append(
            sal_generator(
                ref_image=np.asarray(np.transpose(ref_img, axes=(1, 2, 0))),
                bboxes=np.asarray(dets.boxes),
                scores=score_matrix,
                blackbox=blackbox_detector,
            ),
        )

    return img_sal_maps, to_config_dict(sal_generator)


def sal_on_dets(
    *,
    dataset: Dataset,
    sal_generator: GenerateObjectDetectorBlackboxSaliency,
    detector: Model,
    ids: Sequence[int],
    img_batch_size: int = 1,
) -> tuple[list[np.ndarray], dict]:
    """Generate saliency maps for provided dataset.

    :param dataset: MAITE dataset
    :param sal_generator: ``GenerateObjectDetectorBlackboxSaliency`` generator
    :param detector: MAITE detector
    :param id_to_name: Mapping from label IDs to names
    :param img_batch_size: Image batch size for inference
    """
    return compute_sal_maps(
        dataset=dataset,
        sal_generator=sal_generator,
        blackbox_detector=MAITEDetector(detector=detector, ids=ids, img_batch_size=img_batch_size),
        num_classes=len(ids),
    )
