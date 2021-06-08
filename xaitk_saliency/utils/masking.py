from typing import List

import numpy as np
import PIL.Image


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
    masks: np.ndarray
) -> np.ndarray:
    """
    Weight some binary masks region with its respective vector in scalar_vec.

    We expect the "masks" matrices and the image to be the same height and
    width, and be valued in the [0, 1] floating-point range. The length
    and order of scalar_vec and masks are assumed to be the same.
    In the mask matrix, higher values correspond to regions of the image that
    preserved. E.g. a 0 in the mask will translate to blacking out the
    corresponding location in the source image.

    :param scalar_vec: Weights for image regions for nClasses and shape
                       [nMasks, nClasses]
    :param masks: Mask array in the [nMasks, Height, Weight] shape format.

    :return: A numpy array representing the weighted heatmap.

    Note masks can have pixel regions that are never masked raising a
    warning for divide by zero. A 'NaN' value is present in all such positions
    of the final_heatmap.
    """
    # Weighting each perturbed region with its respective score in vector.
    heatmap = (np.expand_dims(np.transpose(1 - masks), axis=3) * scalar_vec)

    # Aggregate scores across all perturbed regions.
    sal_across_masks = np.transpose(heatmap.sum(axis=2))

    # Compute final saliency map by normalizing with sampling factor.
    final_heatmap = sal_across_masks/(1 - masks).sum(axis=0)

    return final_heatmap
