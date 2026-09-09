"""This module provides the `sal_on_od_dets` CLI script.

It generates saliency maps for detections in an object detection dataset for
`xaitk-saliency`. The dataset is loaded via `datamaite`, so any object detection dataset format
``datamaite`` supports (COCO, YOLO, ...) can be used as input.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from typing import TextIO

import click
import datamaite
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from smqtk_core.configuration import from_config_dict
from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects

from xaitk_saliency import GenerateObjectDetectorBlackboxSaliency
from xaitk_saliency.utils.datamaite import DatamaiteUtils

_OD_FORMATS = tuple(key.format for key in datamaite.available_loader_keys() if key.task == datamaite.Task.OD)


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
def sal_on_od_dets(
    *,
    dataset_dir: str,
    output_dir: str,
    config_file: TextIO,
    dataset_format: datamaite.DatasetFormat,
    loader_option: tuple[str, ...],
    overlay_image: bool,
    verbose: bool,
) -> None:
    r"""Generate saliency maps for detections in an object detection dataset and write them to disk.

    Maps for each detection are written out in subdirectories named
    after their corresponding image file.

    \b
    DATASET_DIR - Root directory of the dataset with detections to compute
        saliency for.
    OUTPUT_DIR - Directory to write the saliency maps to.
    CONFIG_FILE - Configuration file for the DetectImageObjects, PerturbImage,
        and GenerateDetectorProposalSaliency implementations to use.

    \f
    :param dataset_dir: Root directory of the dataset with detections to
        compute saliency for.
    :param output_dir: Directory to write the saliency maps to.
    :param config_file: Config file specifying the ``DetectImageObjects`` and
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

    # load config
    config = json.load(config_file)

    # instantiate objects from config
    blackbox_detector = from_config_dict(config=config["DetectImageObjects"], type_iter=DetectImageObjects.get_impls())
    sal_generator = from_config_dict(
        config=config["GenerateObjectDetectorBlackboxSaliency"],
        type_iter=GenerateObjectDetectorBlackboxSaliency.get_impls(),
    )

    for ref_img, bboxes, scores, sample, kept in DatamaiteUtils().parse_datamaite_dset(input_dataset):
        sal_maps = sal_generator(ref_image=ref_img, bboxes=bboxes, scores=scores, blackbox=blackbox_detector)

        img_file = sample.file_name or str(sample.image_id)
        # split file from parent folder, then split off file extension
        img_name = os.path.splitext(os.path.split(img_file)[1])[0]

        sub_dir = os.path.join(output_dir, img_name)
        os.makedirs(sub_dir, exist_ok=True)

        _save_sal_maps(
            kept=kept,
            sal_maps=sal_maps,
            ref_img=ref_img,
            sub_dir=sub_dir,
            overlay_image=overlay_image,
        )


def _save_sal_maps(
    *,
    kept: Sequence[datamaite.ObjectDetectionAnnotation],
    sal_maps: np.ndarray,
    ref_img: np.ndarray,
    sub_dir: str,
    overlay_image: bool,
) -> None:
    for sal_idx, det in enumerate(kept):
        sal_map = sal_maps[sal_idx]
        # ObjectDetectionAnnotation.source_annotation_id is not guaranteed to exist,
        # instead use the saliency map's index
        det_id = sal_idx

        fig = plt.figure()
        plt.axis("off")
        if overlay_image:
            gray_img = np.asarray(Image.fromarray(ref_img).convert("L"))
            plt.imshow(gray_img, alpha=0.7, cmap="gray")

            x, y, w, h = det.bbox
            plt.gca().add_patch(
                Rectangle(
                    xy=(x, y),
                    width=w,
                    height=h,
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
