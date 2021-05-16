from typing import List, Tuple

import numpy as np
import PIL.Image
from sklearn.preprocessing import minmax_scale

def generate_block_masks(
    window_size: Tuple[int, int],
    stride: Tuple[int, int],
    image_size: Tuple[int, int],
) -> np.ndarray:
    """
    Generates a number of mask images with each masking a distinct square area
    within an input image shape.

    An amount of masks generated will be equal to the number of full
    `window_size x window_size` blocks we can fit into the given shape with the
    given stride.
    If the input shape is not evenly divisible by window size and stride then
    there may be pixels on teh right and bottom that are never masked.

    :param window_size: the block window size as a tuple with format
        `(height, width)`.
    :param stride: the sliding step as a tuple with format
        `(height_step, width_step`.
    :param image_size: The mask size to be output, which should be the same to
        the image size we are creating masks for.

    :return: the sliding window style masks
    """
    win_h, win_w = window_size
    st_h, st_w = stride
    rows = np.arange(0 + st_h - win_h, image_size[0], st_h)
    cols = np.arange(0 + st_w - win_w, image_size[1], st_w)

    mask_num = len(rows) * len(cols)
    masks = np.ones((mask_num, image_size[0], image_size[1]), dtype=bool)
    i = 0
    for r in rows:
        for c in cols:
            if r < 0:
                r1 = 0
            else:
                r1 = r
            if r + win_h > image_size[0]:
                r2 = image_size[0]
            else:
                r2 = r + win_h
            if c < 0:
                c1 = 0
            else:
                c1 = c
            if c + win_w > image_size[1]:
                c2 = image_size[1]
            else:
                c2 = c + win_w
            masks[i, r1:r2, c1:c2] = 0
            i += 1
    return masks


def generate_masked_images(
    masks: np.ndarray,
    img: np.ndarray
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

def weight_regions_by_scalar(
    scalar_vec: np.ndarray,
    masks: np.ndarray) -> np.ndarray:
    """
    Weight some binary masks reguib with its respective vector in scalar_vec.

    We expect the "masks" matrices and the image to be the same height and
    width, and be valued in the [0, 1] floating-point range. The length
    and order of scalar_vec and masks are assumed to be the same.
    In the mask matrix, higher values correspond to regions of the image that
    preserved. E.g. a 0 in the mask will translate to blacking out the
    corresponding location in the source image.

    :param scalar_vec: Weights for image regions in masks.
    :param masks: Mask array in the [nMasks, Height, Weight] shape format.

    :return: A numpy array representing the weighted heatmap.
    """

    # Creating an empty heatmap for aggregating weighted matrices
    heatmap = np.zeros((masks.shape[-2:]))

    # Iterating through perturbation mask and respective weight
    for weight, prtb_region in zip(scalar_vec, masks):
        heatmap += ((1 - prtb_region) * weight)

    normalized_heatmap = heatmap / len(scalar_vec)
    final_heatmap = minmax_scale(normalized_heatmap.ravel(), \
                                 feature_range=(0,1)).reshape(heatmap.shape)
    return final_heatmap
