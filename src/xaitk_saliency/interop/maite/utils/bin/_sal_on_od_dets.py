"""This module provides a CLI tool for generating saliency maps for object detection models.

It uses ``datamaite`` to load the input dataset, so any object detection dataset format
``datamaite`` supports (COCO, YOLO, ...) can be used as input. It leverages saliency
generation methods and blackbox object detection models to create visual explanations for predictions.

Functions:
    sal_on_od_dets: CLI command to generate saliency maps and optionally overlay them
                      on input images with bounding boxes.

Dependencies:
    - datamaite: For loading object detection datasets.
    - matplotlib: For visualizing and saving saliency maps.
    - xaitk_saliency: For generating saliency maps.
    - smqtk_detection: For object detection interfaces.
    - PIL: For image processing.
    - numpy: For numerical operations.
"""

import json
import logging
import os
from dataclasses import replace
from typing import TextIO

import click
import datamaite
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from smqtk_core.configuration import from_config_dict
from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects

from xaitk_saliency.interfaces.gen_object_detector_blackbox_sal import GenerateObjectDetectorBlackboxSaliency
from xaitk_saliency.interop.maite.utils._sal_on_dets import compute_sal_maps

_OD_FORMATS = tuple(key.format for key in datamaite.available_loader_keys() if key.task == datamaite.Task.OD)


def _detection_score(det: datamaite.ObjectDetectionAnnotation) -> float:
    if det.score is not None:
        return det.score
    if "score" in det.attributes:
        return det.attributes["score"]
    if "prob" in det.attributes:
        return max(det.attributes["prob"])
    return 1.0


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("output_dir", type=click.Path(exists=False))
@click.argument("config_file", type=click.File(mode="r"))
@click.option(
    "-i",
    "--dataset-format",
    default="coco",
    show_default=True,
    type=click.Choice(_OD_FORMATS, case_sensitive=False),
    help="format of the input dataset",
)
@click.option(
    "-o",
    "--loader-option",
    multiple=True,
    metavar="KEY=VALUE",
    help="extra key=value option forwarded to the datamaite loader, dataset format-specific.",
)
@click.option(
    "--overlay-image",
    is_flag=True,
    help="overlay saliency map on images with bounding boxes, RGB images are converted to grayscale",
)
@click.option("--verbose", "-v", count=True, help="print progress messages")
def sal_on_od_dets(  # noqa: C901
    *,
    dataset_dir: str,
    output_dir: str,
    config_file: TextIO,
    dataset_format: datamaite.DatasetFormat,
    loader_option: tuple[str, ...],
    overlay_image: bool,
    verbose: bool,
) -> None:
    """Generate saliency maps for detections in an object detection dataset and write them to disk.

    Maps for each detection are written out in subdirectories named after their corresponding image file.

    DATASET_DIR - Root directory of dataset.
    OUTPUT_DIR - Directory to write the saliency maps to.
    CONFIG_FILE - Configuration file for the object detector and
    GenerateObjectDetectorBlackboxSaliency implementations to use.

    :param dataset_dir: Root directory of dataset.
    :param output_dir: Directory to write the saliency maps to.
    :param config_file: Config file specifying the object detector and
        ``GenerateObjectDetectorBlackboxSaliency`` implementations to use.
    :param dataset_format: Format of the input ``datamaite`` dataset.
    :param loader_option: Extra ``key=value`` options forwarded to the ``datamaite``
        loader for ``dataset_format``.
    :param overlay_image: Overlay saliency maps on images with bounding boxes.
        RGB images are converted to grayscale. Default is to write out saliency
        maps by themselves.
    :param verbose: Display progress messages. Default is false.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    extra_loader_options: dict[str, str] = {}
    for opt in loader_option:
        key, sep, value = opt.partition("=")
        if not sep:
            raise click.BadParameter(f"expected KEY=VALUE, got {opt}", param_hint="'-o' / '--loader-option'")
        extra_loader_options[key] = value

    logging.info(f"Loading {dataset_format.value} dataset from {dataset_dir}")
    input_dataset = datamaite.load_od(dataset_dir, dataset_format=dataset_format, **extra_loader_options)

    taxonomy = input_dataset.dataset_metadata.taxonomy
    if taxonomy is None or not taxonomy.entries:
        raise ValueError(
            f"No categories declared in {dataset_dir}",
        )

    num_classes = len(taxonomy.entries)

    # Category ids are not guaranteed to be contiguous or 0-based. Remap each detection's
    # category id to its dense version.
    source_to_dense_id = taxonomy.dense_ids()

    def _dense_remap(sample: datamaite.ImageObjectDetectionSample) -> datamaite.ImageObjectDetectionSample:
        keep = []
        for det in sample.detections:
            dense_id = source_to_dense_id.get(det.category_id)
            if dense_id is None:
                continue
            keep.append(replace(det, category_id=dense_id, score=_detection_score(det)))
        return replace(sample, detections=tuple(keep))

    remapped_samples = []
    num_dropped = 0
    for sample in input_dataset.samples:
        dense_sample = _dense_remap(sample)
        num_dropped += len(sample.detections) - len(dense_sample.detections)
        if dense_sample.detections:
            remapped_samples.append(dense_sample)
    input_dataset = replace(input_dataset, samples=tuple(remapped_samples))
    if num_dropped:
        logging.warning(f"Dropped {num_dropped} detections with an undeclared category id.")

    # Load config
    config = json.load(config_file)

    # Instantiate objects from config
    sal_generator = from_config_dict(
        config=config["GenerateObjectDetectorBlackboxSaliency"],
        type_iter=GenerateObjectDetectorBlackboxSaliency.get_impls(),
    )
    blackbox_detector = from_config_dict(config=config["DetectImageObjects"], type_iter=DetectImageObjects.get_impls())

    img_sal_maps, _ = compute_sal_maps(
        # datamaite.ObjectDetectionDataset does not satisfy the MAITE Dataset protocol
        # Current issue: https://gitlab.jatic.net/jatic/orchestration-interoperability/datamaite/-/work_items/124
        dataset=input_dataset,  # pyright: ignore[reportArgumentType]
        sal_generator=sal_generator,
        blackbox_detector=blackbox_detector,
        num_classes=num_classes,
    )

    # Save saliency maps
    for dset_idx in range(len(input_dataset)):
        ref_img = input_dataset.get_input(dset_idx)
        dets = input_dataset.get_target(dset_idx)
        md = input_dataset.get_metadata(dset_idx)
        sample = input_dataset.samples[dset_idx]
        img_file = md.get("file_name", str(sample.image_id))

        if overlay_image:
            # Convert to channel last
            ref_img = np.transpose(np.asarray(ref_img), axes=(1, 2, 0))
            ref_img = np.squeeze(ref_img)

        # split file from parent folder
        img_name = os.path.split(img_file)[1]
        # split off file extension
        img_name = os.path.splitext(img_name)[0]

        sub_dir = os.path.join(output_dir, img_name)

        os.makedirs(sub_dir, exist_ok=True)

        bboxes = np.asarray(dets.boxes)
        for sal_idx, bbox in enumerate(bboxes):
            sal_map = img_sal_maps[dset_idx][sal_idx]
            # ObjectDetectionAnnotation.source_annotation_id is not guaranteed to exist,
            # instead use the saliency map's index
            det_id = sal_idx

            fig = plt.figure()
            plt.axis("off")
            if overlay_image:
                gray_img = np.asarray(Image.fromarray(np.asarray(ref_img)).convert("L"))
                plt.imshow(gray_img, alpha=0.7, cmap="gray")

                plt.gca().add_patch(
                    Rectangle(
                        xy=(bbox[0], bbox[1]),
                        width=bbox[2] - bbox[0],
                        height=bbox[3] - bbox[1],
                        linewidth=1,
                        edgecolor="r",
                        facecolor="none",
                    ),
                )
                plt.imshow(sal_map, cmap="jet", alpha=0.3)
                plt.colorbar()
            else:
                plt.imshow(sal_map, cmap="jet")
                plt.colorbar()
            plt.savefig(os.path.join(sub_dir, f"det_{det_id}.jpeg"), bbox_inches="tight")
            plt.close(fig)
