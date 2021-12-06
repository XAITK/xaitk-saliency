import click  # type: ignore
import os
from PIL import Image  # type: ignore
import json
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.patches import Rectangle  # type: ignore
import numpy as np
from typing import TextIO
import logging

from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects
from smqtk_core.configuration import from_config_dict
from smqtk_core.configuration import make_default_config

from xaitk_saliency.interfaces.gen_detector_prop_sal import GenerateDetectorProposalSaliency
from xaitk_saliency.interfaces.perturb_image import PerturbImage
from xaitk_saliency.utils.tools.gen_detector_sal import gen_detector_sal


@click.command(context_settings={"help_option_names": ['-h', '--help']})
@click.argument('img_dir', type=click.Path(exists=True, file_okay=False))
@click.argument('output_dir', type=click.Path(exists=False))
@click.argument('config_file', type=click.File(mode='r'))
@click.option(
    '--overlay-image',
    is_flag=True,
    help='overlay saliency map on images with bounding boxes, RGB images are converted to grayscale'
)
@click.option('-g', '--generate-config-file', help='write default config to specified file', type=click.File(mode='w'))
@click.option('--verbose', '-v', count=True, help='print progress messages')
def det_sal_on_img_dir(
    img_dir: str,
    output_dir: str,
    config_file: TextIO,
    overlay_image: bool,
    generate_config_file: TextIO,
    verbose: bool
) -> None:
    """
    Generate saliency maps for each image in a directory.

    \b
    IMG_DIR - Directory containg the imgaes.
    OUTPUT_DIR - Directory to write the saliency maps to.
    CONFIG_FILE - Configuration file for the DetectImageObjects, PerturbImage,
        and GenerateDetectorProposalSaliency implementations to use.

    \f
    :param img_dir: Directory containing the images.
    :param output_dir: Directory to write the saliency maps to.
    :param config_file: Config file specifying the ``DetectImageObjects``,
        ``PerturbImage``, and ``GenerateDetectorProposalSaliency``
        implementations to use.
    :param overlay_image: Overlay saliency maps on images with bounding boxes.
        RGB images are converted to grayscaleDefault is to output write out
        saliency maps by themselves.
    :param verbose: Display progress messages. Default is false.
    """

    if generate_config_file:

        config = {}

        config["DetectImageObjects"] = make_default_config(DetectImageObjects.get_impls())
        config["PerturbImage"] = make_default_config(PerturbImage.get_impls())
        config["GenerateDetectorProposalSaliency"] = make_default_config(GenerateDetectorProposalSaliency.get_impls())

        json.dump(config, generate_config_file, indent=4)

        exit()

    files = os.listdir(img_dir)

    img_files = []
    imgs = []

    for file_name in files:
        try:
            img = np.asarray(Image.open(os.path.join(img_dir, file_name)))
            imgs.append(img)
            img_files.append(file_name)
        except IOError:
            print(f"File {file_name} is not a valid image file.")

    config = json.load(config_file)

    blackbox_detector = from_config_dict(config["DetectImageObjects"], DetectImageObjects.get_impls())
    img_perturber = from_config_dict(config["PerturbImage"], PerturbImage.get_impls())
    sal_generator = from_config_dict(
        config["GenerateDetectorProposalSaliency"],
        GenerateDetectorProposalSaliency.get_impls()
    )

    if verbose:
        logging.basicConfig(level=logging.INFO)

    sal_maps_list, ref_dets_list = gen_detector_sal(
        imgs,
        blackbox_detector,
        img_perturber,
        sal_generator,
        verbose=verbose
    )

    for img_file, ref_img, img_sal_maps, img_dets in zip(img_files, imgs, sal_maps_list, ref_dets_list):

        dot_ind = img_file.rfind('.')

        if dot_ind == -1:
            img_name = img_file
        else:
            img_name = img_file[:dot_ind]

        sub_dir = os.path.join(output_dir, img_name)

        os.makedirs(sub_dir, exist_ok=True)

        for det_i, (sal_map, ref_det) in enumerate(zip(img_sal_maps, img_dets)):
            fig = plt.figure()
            plt.axis('off')
            if overlay_image:
                gray_img = np.asarray(Image.fromarray(ref_img).convert("L"))
                plt.imshow(gray_img, alpha=0.7, cmap='gray')
                plt.gca().add_patch(Rectangle(
                    (ref_det[0], ref_det[1]),
                    ref_det[2] - ref_det[0],
                    ref_det[3] - ref_det[1],
                    linewidth=1,
                    edgecolor='r',
                    facecolor='none'
                ))
                plt.imshow(sal_map, cmap='jet', alpha=0.3)
                plt.colorbar()
            else:
                plt.imshow(sal_map, cmap='jet')
                plt.colorbar()
            plt.savefig(os.path.join(sub_dir, f"det_{det_i}.jpeg"), bbox_inches='tight')
            plt.close(fig)
