"""This module provides a CLI tool for generating saliency maps for object detection models.

It applies to datasets in COCO format. It leverages saliency generation methods
and blackbox object detection models to create visual explanations for predictions.

Functions:
    sal_on_coco_dets: CLI command to generate saliency maps and optionally overlay them
                      on input images with bounding boxes.

Dependencies:
    - kwcoco: For working with COCO-format datasets.
    - matplotlib: For visualizing and saving saliency maps.
    - xaitk_saliency: For generating saliency maps.
    - smqtk_detection: For object detection interfaces.
    - PIL: For image processing.
    - numpy: For numerical operations.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, TextIO

import click
import kwcoco
import numpy as np
from PIL import Image
from smqtk_core.configuration import from_config_dict, make_default_config
from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects

from xaitk_saliency.interfaces.gen_object_detector_blackbox_sal import GenerateObjectDetectorBlackboxSaliency
from xaitk_saliency.interop.maite.object_detection._dataset import COCOMAITEObjectDetectionDataset
from xaitk_saliency.interop.maite.utils._sal_on_dets import compute_sal_maps

plt = None
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:
    pass


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("dataset_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=False))
@click.argument("config_file", type=click.File(mode="r"))
@click.option(
    "-g",
    "--generate-config-file",
    help="write default config to specified file",
    type=click.File(mode="w"),
)
@click.option(
    "--overlay-image",
    is_flag=True,
    help="overlay saliency map on images with bounding boxes, RGB images are converted to grayscale",
)
@click.option("--verbose", "-v", count=True, help="print progress messages")
def sal_on_coco_dets(  # noqa: C901
    *,
    dataset_dir: str,
    output_dir: str,
    config_file: TextIO,
    overlay_image: bool,
    generate_config_file: TextIO,
    verbose: bool,
) -> None:
    """Generate saliency maps for detections in a COCO format file and write them to disk.

    Maps for each detection are written out in subdirectories named after their corresponding image file.

    DATASET_DIR - Root directory of dataset.
    OUTPUT_DIR - Directory to write the saliency maps to.
    CONFIG_FILE - Configuration file for the object detector and
    GenerateObjectDetectorBlackboxSaliency implementations to use.

    :param dataset_dir: Root directory of dataset.
    :param output_dir: Directory to write the saliency maps to.
    :param config_file: Config file specifying the object detector and
        ``GenerateObjectDetectorBlackboxSaliency`` implementations to use.
    :param overlay_image: Overlay saliency maps on images with bounding boxes.
        RGB images are converted to grayscale. Default is to write out saliency
        maps by themselves.
    :param generate_config_file: File to write default config file, only written
        if specified. Only one of "DetectImageObjects" and "ObjectDetector" should
        be kept.
        This skips the normal operation of this tool and only outputs the file.
    :param verbose: Display progress messages. Default is false.
    """
    if generate_config_file:
        config: dict[str, Any] = {}

        config["DetectImageObjects"] = make_default_config(DetectImageObjects.get_impls())
        config["GenerateObjectDetectorBlackboxSaliency"] = make_default_config(
            GenerateObjectDetectorBlackboxSaliency.get_impls(),
        )

        json.dump(config, generate_config_file, indent=4)

        exit()

    if verbose:
        logging.basicConfig(level=logging.INFO)

    # Load COCO dataset
    coco_file = Path(dataset_dir) / "annotations.json"
    if not coco_file.is_file():
        raise ValueError("Could not identify annotations file. Expected at '[dataset_dir]/annotations.json'.")
    logging.info(f"Loading kwcoco annotations from {coco_file}.")
    # kwcoco's __init__.py uses lazy import loading, so pyright
    # infers `kwcoco.CocoDataset` as `ModuleType` instead of a class.
    kwcoco_dataset = kwcoco.CocoDataset(coco_file)  # pyright: ignore[reportCallIssue]

    # Load metadata, if it exists
    metadata_file = Path(dataset_dir) / "image_metadata.json"
    if not metadata_file.is_file():
        logging.info(
            "Could not identify metadata file, assuming no metadata. Expected at '[dataset_dir]/image_metadata.json'",
        )
        metadata = [{"id": img_id} for img_id in kwcoco_dataset.imgs]
    else:
        logging.info(f"Loading metadata from {metadata_file}")
        with open(metadata_file) as f:
            metadata = json.load(f)

    # Initialize dataset object
    input_dataset = COCOMAITEObjectDetectionDataset(
        kwcoco_dataset=kwcoco_dataset,
        image_metadata=metadata,
        skip_no_anns=True,
    )

    cids = [cat["id"] for cat in kwcoco_dataset.cats.values()]
    min_cid = min(cids)
    num_classes = max(cids) - min_cid + 1

    # Load config
    config = json.load(config_file)

    # Instantiate objects from config
    sal_generator = from_config_dict(
        config["GenerateObjectDetectorBlackboxSaliency"],
        GenerateObjectDetectorBlackboxSaliency.get_impls(),
    )
    blackbox_detector = from_config_dict(config["DetectImageObjects"], DetectImageObjects.get_impls())

    img_sal_maps, _ = compute_sal_maps(
        dataset=input_dataset,
        sal_generator=sal_generator,
        blackbox_detector=blackbox_detector,
        num_classes=num_classes,
    )

    # Save saliency maps
    for dset_idx in range(len(input_dataset)):
        ref_img, dets, md = input_dataset[dset_idx]
        img_file = md["image_info"]["file_name"]

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

        ann_ids = md["ann_ids"]
        bboxes = np.asarray(dets.boxes)
        for sal_idx, bbox in enumerate(bboxes):
            sal_map = img_sal_maps[dset_idx][sal_idx]
            det_id = ann_ids[sal_idx]

            if plt is not None:
                fig = plt.figure()
                plt.axis("off")
                if overlay_image:
                    gray_img = np.asarray(Image.fromarray(np.asarray(ref_img)).convert("L"))
                    plt.imshow(gray_img, alpha=0.7, cmap="gray")

                    plt.gca().add_patch(
                        # Non-None plt confirms Rectangle will be non-None.
                        Rectangle(  # pyright: ignore[reportPossiblyUnboundVariable]
                            (bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1],
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
