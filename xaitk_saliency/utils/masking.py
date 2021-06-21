from typing import Generator, Iterable, Optional, Tuple, Union

import numpy as np
# import PIL.Image
# from smqtk_descriptors.utils import parallel_map


# def occlude_image(
#     img: PIL.Image.Image,
#     masks: Iterable[np.ndarray],
#     color: Union[int, Tuple[int, ...]],
#     threads: Optional[int] = None,
# ) -> Generator[PIL.Image.Image, None, None]:
#     """
#     Apply some binary masks onto one common image, generating a number of new
#     images with the masked regions maintained.
#
#     We expect the "mask" matrices and the image to be the same height and
#     width, and be valued in the [0, 1] floating-point range.
#     In the mask matrix, higher values correspond to regions of the image that
#     preserved. E.g. a 0 in the mask will translate to blacking out the
#     corresponding location in the source image.
#
#     :param masks: Mask images in the [N, Height, Weight, 1] shape format.
#     :param img: Original base image
#     :param threads: Optional number of threads to use for parallelism when set
#         to a positive integer. If 0, a negative value, or `None`, work will be
#         performed on the main-thread in-line.
#
#     :return: List of masked images in PIL Image form.
#     """
#     image_from_array = PIL.Image.fromarray
#     ref_mat = np.asarray(img)
#     ref_mode = img.mode
#     s: Tuple = (...,)
#     if ref_mat.ndim > 2:
#         s = (..., None)  # add channel axis for multiplication
#
#     def work_func(m: np.ndarray) -> PIL.Image.Image:
#         # !!! This is the majority cost of the perturbation-masking pipeline.
#         img_m = (m[s] * ref_mat).astype(ref_mat.dtype)
#         img_p = image_from_array(img_m, mode=ref_mode)
#         return img_p
#
#     if threads is None or threads < 1:
#         for mask in masks:
#             yield work_func(mask)
#     else:
#         for mask in parallel_map(
#             work_func, masks,
#             cores=threads,
#             use_multiprocessing=False,
#         ):
#             yield mask


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
    """
    # Weighting each perturbed region with its respective score in vector.
    heatmap = (np.expand_dims(np.transpose(1 - masks), axis=3) * scalar_vec)

    # Aggregate scores across all perturbed regions.
    sal_across_masks = np.transpose(heatmap.sum(axis=2))

    # Removing regions that are never masked to avoid a dividebyzero warning
    mask_sum = (1 - masks).sum(axis=0)
    mask_sum[mask_sum == 0] = 1.

    # Compute final saliency map by normalizing with sampling factor.
    final_heatmap = sal_across_masks/mask_sum
    return final_heatmap
