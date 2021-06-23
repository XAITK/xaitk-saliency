import itertools
from typing import Generator, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import PIL.Image
from smqtk_descriptors.utils import parallel_map


# Lowest volume to convert `1` value.
UINT8_ONE = np.uint8(1)


def occlude_image_batch(
    ref_image: PIL.Image.Image,
    masks: np.ndarray,
    fill: Optional[Union[int, Sequence[int]]] = None,
) -> List[PIL.Image.Image]:
    """
    Apply a number of input occlusion masks to the given reference image,
    producing a list of images equivalent in length, and parallel in order, to
    the input masks.

    We expect the "mask" matrices and the image to be the same height and
    width, and for the mask matrix values to be in the [0, 1] range.
    In the mask matrix, values closer to 1 correspond to regions of the image
    that should *NOT* be occluded.
    E.g. a 0 in the mask will translate to "blacking out" the corresponding
    location in the source image.

    We optionally take in a fill-color value as input. This may be either a
    scalar or per-channel sequence. When no fill is passed, black is used
    (default absence of color).

    Images output will mirror the input image format. As such, the `fill` value
    passed must be compatible with the input image channels for broadcasting.
    For example, a single channel input will not be able to be broadcast
    against a multi-channel `fill` input. A ValueError will be raised by the
    underlying numpy call in such cases.

    NOTE: Due to the batch nature of this function, utilizing a fill color will
    consistently utilize more RAM due to the use of alpha blending

    Assumptions:
      * Mask input is per-pixel. Does not accept per-channel masks.
      * Fill value input is in an applicable value range supported by the input
        image format, which is mirrored in output images.

    :param ref_image: Reference image to generate perturbations from.
    :param masks: Mask matrix input of shape `[N x H x W]` where height and
        width dimensions are the same size as the input `ref_image`.
    :param fill: Optional fill color for the occluded regions as a single uint8
        value or a per-channel sequence.

    :raises ValueError: The input mask matrix was not 3-dimensional, its last
        two dimensions did not match the shape of the input imagery, or the
        input fill value could not be broadcast against the input image.

    :return: List of masked images in PIL Image form.
    """
    if masks.ndim != 3:
        raise ValueError(
            f"Expected a 3-dimension mask input [N x H x W]. "
            f"Got masks.ndim = {masks.ndim}"
        )
    if ref_image.size[::-1] != masks.shape[1:]:
        raise ValueError(
            f"Input image shape and mask image shape did not match: "
            f"{ref_image.size[::-1]} != {masks.shape[1:]}"
        )
    ref_mode = ref_image.mode
    ref_mat = np.asarray(ref_image)
    s: Tuple = (...,)
    if ref_mat.ndim > 2:
        s = (..., None)  # add channel axis for multiplication
    # The Batch Operation -> Bulk apply mask matrices to input image matrix to
    # generate occluded images.
    occ_img_mats = np.ndarray([len(masks), *ref_mat.shape], dtype=ref_mat.dtype)
    if fill is not None:
        masks_sview = masks[s]
        np.add(
            (masks_sview * ref_mat),
            ((UINT8_ONE - masks_sview) * fill),
            out=occ_img_mats, casting="unsafe"
        )
    else:
        np.multiply(masks[s], ref_mat, out=occ_img_mats, casting="unsafe")
    return [
        PIL.Image.fromarray(img_mat, mode=ref_mode)
        for img_mat in occ_img_mats
    ]


def occlude_image_streaming(
    ref_image: PIL.Image.Image,
    masks: Iterable[np.ndarray],
    fill: Optional[Union[int, Sequence[int]]] = None,
    threads: Optional[int] = None,
) -> Generator[PIL.Image.Image, None, None]:
    """
    Apply a number of input occlusion masks to the given reference image,
    producing a list of images equivalent in length, and parallel in order, to
    the input masks.

    We expect the "mask" matrices and the image to be the same height and
    width, and for the mask matrix values to be in the [0, 1] range.
    In the mask matrix, values closer to 1 correspond to regions of the image
    that should *NOT* be occluded.
    E.g. a 0 in the mask will translate to "blacking out" the corresponding
    location in the source image.

    We optionally take in a fill-color value as input. This may be either a
    scalar or per-channel sequence. When no fill is passed, black is used
    (default absence of color).

    Images output will mirror the input image format. As such, the `fill` value
    passed must be compatible with the input image channels for broadcasting.
    For example, a single channel input will not be able to be broadcast
    against a multi-channel `fill` input. A ValueError will be raised by the
    underlying numpy call in such cases.

    Assumptions:
      * Mask input is per-pixel. Does not accept per-channel masks.
      * Fill value input is in an applicable value range supported by the input
        image format, which is mirrored in output images.

    :param ref_image: Original base image
    :param masks: Mask images in the [N, Height, Weight] shape format.
    :param fill: Optional fill color for the occluded regions as a single uint8
        value or a per-channel sequence.
    :param threads: Optional number of threads to use for parallelism when set
        to a positive integer. If 0, a negative value, or `None`, work will be
        performed on the main-thread in-line.

    :raises ValueError: One or more input masks in the input iterable did not
        match shape of the input reference image.

    :return: List of masked images in PIL Image form.
    """
    img_shape = (ref_image.height, ref_image.width)
    image_from_array = PIL.Image.fromarray
    ref_mat = np.asarray(ref_image)
    ref_mode = ref_image.mode
    ref_dtype = ref_mat.dtype
    s: Tuple = (...,)
    if ref_mat.ndim > 2:
        s = (..., None)  # add channel axis for multiplication

    def work_func(i_: int, m: np.ndarray) -> PIL.Image.Image:
        m_shape = m.shape
        if m_shape != img_shape:
            raise ValueError(
                f"Input mask (position {i_}) did not the shape of the input "
                f"image: {m_shape} != {img_shape}"
            )
        img_m = np.ndarray(ref_mat.shape, dtype=ref_dtype)
        if fill is not None:
            np.add(
                (m[s] * ref_mat),
                ((UINT8_ONE - m[s]) * fill),
                out=img_m, casting="unsafe"
            )
        else:
            np.multiply(m[s], ref_mat, out=img_m, casting="unsafe")
        img_p = image_from_array(img_m, mode=ref_mode)
        return img_p

    if threads is None or threads < 1:
        for i, mask in enumerate(masks):
            yield work_func(i, mask)
    else:
        for img in parallel_map(
            work_func, itertools.count(), masks,
            cores=threads,
            use_multiprocessing=False,
        ):
            yield img


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


def weight_regions_by_scalar_rise(
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
    # NOTE: Here we use masks instead of 1 - masks as in the original method
    heatmap = (np.expand_dims(np.transpose(masks), axis=3) * scalar_vec)

    # Aggregate scores across all perturbed regions.
    sal_across_masks = np.transpose(heatmap.sum(axis=2))
    # NOTE: RISE does not need an explicit mask normalization step
    return sal_across_masks
