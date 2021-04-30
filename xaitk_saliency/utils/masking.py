from typing import List

import numpy as np
import PIL.Image


def generate_masked_images(
    masks: np.ndarray,
    img: PIL.Image.Image
) -> List[PIL.Image.Image]:
    """
    Apply some binary masks onto one common image, generating a number of new
    images with the masked regions maintained.

    We expect the "mask" matrices and the image to be the same height and
    width, and be valued in the [0, 1] floating-point range.
    In the mask matrix, higher values correspond to regions of the image that
    preserved. E.g. a 0 in the mask will translate to blacking out the
    corresponding location in the source image.

    :param masks: Mask images in the [N, Height, Weight, 1] shape format.
    :param img: Original base image

    :return: List of masked images in PIL Image form.
    """
    masked_imgs = []
    for mask in masks:
        masked_img = np.multiply(mask, img, casting='unsafe')
        masked_imgs.append(PIL.Image.fromarray(np.uint8(masked_img)))
    return masked_imgs
