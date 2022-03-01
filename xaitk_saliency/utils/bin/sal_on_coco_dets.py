import click  # type: ignore
import os
from PIL import Image  # type: ignore
import json
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.patches import Rectangle  # type: ignore
import numpy as np
from typing import TextIO, Optional, Tuple
import logging

from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects
from smqtk_core.configuration import from_config_dict, make_default_config

from xaitk_saliency import PerturbImage, GenerateDetectorProposalSaliency

try:
    import kwcoco  # type: ignore
    from xaitk_saliency.utils.gen_coco_sal import gen_coco_sal
    is_usable = True
except (ModuleNotFoundError, NameError):
    is_usable = False


@click.command(context_settings={"help_option_names": ['-h', '--help']})
@click.argument('coco_file', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=False))
@click.argument('config_file', type=click.File(mode='r'))
@click.option(
    '--fill',
    '-f',
    help="""channel value (0-255) of fill to use when occluding images, number
            of values passed should match the number of channels in each
            image""",
    multiple=True,
    type=int,
)
@click.option(
    '--overlay-image',
    is_flag=True,
    help='overlay saliency map on images with bounding boxes, RGB images are converted to grayscale'
)
@click.option('-g', '--generate-config-file', help='write default config to specified file', type=click.File(mode='w'))
@click.option('--verbose', '-v', count=True, help='print progress messages')
def sal_on_coco_dets(
    coco_file: str,
    output_dir: str,
    config_file: TextIO,
    fill: Optional[Tuple[int]],
    overlay_image: bool,
    generate_config_file: TextIO,
    verbose: bool
) -> None:
    """
    Generate saliency maps for detections in a COCO format file and write them
    to disk. Maps for each detection are written out in subdirectories named
    after their corresponding image file.

    \b
    COCO_FILE - COCO style annotation file with detections to compute saliency
        for.
    OUTPUT_DIR - Directory to write the saliency maps to.
    CONFIG_FILE - Configuration file for the DetectImageObjects, PerturbImage,
        and GenerateDetectorProposalSaliency implementations to use.

    \f
    :param coco_file: COCO style annotation file with detections to compute
        saliency for.
    :param output_dir: Directory to write the saliency maps to.
    :param config_file: Config file specifying the ``DetectImageObjects``,
        ``PerturbImage``, and ``GenerateDetectorProposalSaliency``
        implementations to use.
    :param fill: Channel values for fill to use when occluding images. The
        number of values provided should match the number of channels in each
        of your images.
    :param overlay_image: Overlay saliency maps on images with bounding boxes.
        RGB images are converted to grayscale. Default is to write out saliency
        maps by themselves.
    :param verbose: Display progress messages. Default is false.
    """

    if generate_config_file:

        config = {}

        config["DetectImageObjects"] = make_default_config(DetectImageObjects.get_impls())
        config["PerturbImage"] = make_default_config(PerturbImage.get_impls())
        config["GenerateDetectorProposalSaliency"] = make_default_config(GenerateDetectorProposalSaliency.get_impls())

        json.dump(config, generate_config_file, indent=4)

        exit()

    if not is_usable:
        print("This tool requires additional dependencies, please install 'xaitk-saliency[tools]'")
        exit(-1)

    # load dets
    dets_dset = kwcoco.CocoDataset(coco_file)

    # load config
    config = json.load(config_file)

    # instantiate objects from config
    blackbox_detector = from_config_dict(config["DetectImageObjects"], DetectImageObjects.get_impls())
    img_perturber = from_config_dict(config["PerturbImage"], PerturbImage.get_impls())
    sal_generator = from_config_dict(
        config["GenerateDetectorProposalSaliency"],
        GenerateDetectorProposalSaliency.get_impls()
    )

    if verbose:
        logging.basicConfig(level=logging.INFO)

    if fill == ():
        fill = None

    sal_maps = gen_coco_sal(
        dets_dset,
        blackbox_detector,
        img_perturber,
        sal_generator,
        fill=fill,
        verbose=verbose
    )

    for img_id, det_ids in dets_dset.gid_to_aids.items():
        # continue if there are no dets for this image
        if len(det_ids) == 0:
            continue

        img_file = dets_dset.get_image_fpath(img_id)
        ref_img = np.asarray(Image.open(img_file))

        img_file = dets_dset.imgs[img_id]['file_name']

        # split file from parent folder
        img_name = os.path.split(img_file)[1]
        # split off file extension
        img_name = os.path.splitext(img_name)[0]

        sub_dir = os.path.join(output_dir, img_name)

        os.makedirs(sub_dir, exist_ok=True)

        for det_id in det_ids:

            sal_map = sal_maps[det_id]

            fig = plt.figure()
            plt.axis('off')
            if overlay_image:
                gray_img = np.asarray(Image.fromarray(ref_img).convert("L"))
                plt.imshow(gray_img, alpha=0.7, cmap='gray')

                bbox = dets_dset.anns[det_id]['bbox']
                plt.gca().add_patch(Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2],
                    bbox[3],
                    linewidth=1,
                    edgecolor='r',
                    facecolor='none'
                ))
                plt.imshow(sal_map, cmap='jet', alpha=0.3)
                plt.colorbar()
            else:
                plt.imshow(sal_map, cmap='jet')
                plt.colorbar()
            plt.savefig(os.path.join(sub_dir, f"det_{det_id}.jpeg"), bbox_inches='tight')
            plt.close(fig)
