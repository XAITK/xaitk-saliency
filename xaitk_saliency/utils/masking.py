import itertools
import time
from typing import Generator, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from smqtk_descriptors.utils import parallel_map


# Lowest volume to convert `1` value.
UINT8_ONE = np.uint8(1)


def occlude_image_batch(
    ref_image: np.ndarray,
    masks: np.ndarray,
    fill: Optional[Union[int, Sequence[int], np.ndarray]] = None,
    threads: Optional[int] = None,
) -> np.ndarray:
    """
    Apply a number of input occlusion masks to the given reference image,
    producing a list of images equivalent in length, and parallel in order, to
    the input masks.
    This batch version will compute all occluded images and returns them all in
    one large matrix.

    We expect the "mask" matrices and the image to be the same height and
    width, and for the mask matrix values to be in the [0, 1] range.
    In the mask matrix, values closer to 1 correspond to regions of the image
    that should *NOT* be occluded.
    E.g. a 0 in the mask will translate to *fully* occluding the corresponding
    location in the source image.

    We optionally take in a "fill" that alpha-blend into masked regions of the
    input `ref_image`.
    `fill` may be either a scalar, sequence of scalars, or another image matrix
    congruent in shape to the `ref_image`.
    When `fill` is a scalar or a sequence of scalars, the scalars should be in
    the same data-type and value range as the input image.
    A sequence of scalars should be the same length as there are channels in
    the `ref_image`.
    When `fill` is an image matrix it should follow the format of `[H x W]` or
    `[H x W x C]`, should be in the same dtype and value range as `ref_image`
    and should match the same number of channels if channels are provided.
    When no fill is passed, black is used (default absence of color).

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
    :param fill: Optional fill for alpha-blending based on the input masks for
        the occluded regions as a scalar value, a per-channel sequence or a
        shape-matched image.
    :param threads: Optional number of threads to use for parallelism when set
        to a positive integer. If 0, a negative value, or `None`, work will be
        performed on the main-thread in-line.

    :raises ValueError: The input mask matrix was not three-dimensional, its last
        two dimensions did not match the shape of the input imagery, or the
        input fill value could not be broadcast against the input image.

    :return: A numpy array of masked images.
    """
    # Full shape of input image, including channel dim if present.
    ref_image_shape = ref_image.shape
    if masks.ndim != 3:
        raise ValueError(
            f"Expected a 3-dimension mask input [N x H x W]. "
            f"Got masks.ndim = {masks.ndim}"
        )
    if ref_image_shape[:2] != masks.shape[1:]:
        raise ValueError(
            f"Input image shape and mask image shape did not match: "
            f"{ref_image_shape[:2]} != {masks.shape[1:]}"
        )
    s: Tuple = (...,)
    if ref_image.ndim > 2:
        s = (..., None)  # add channel axis for multiplication
    # Basically `np.empty_like` but tacking on the num-masks-dim to the front
    # of the shape.
    occ_img_mats = np.empty(
        (len(masks), *ref_image_shape),
        dtype=ref_image.dtype,
    )
    # The Batch Operation -> Bulk apply mask matrices to input image matrix to
    # generate occluded images.

    masks_sview = masks[s]

    def work_func(i_: int) -> np.ndarray:
        occ_mat = masks_sview[i_] * ref_image

        if fill is not None:
            occ_mat += (UINT8_ONE - masks_sview[i_]) * np.array(fill, dtype=ref_image.dtype)

        return occ_mat.astype(ref_image.dtype)

    if threads is None or threads < 1:
        for i in range(len(masks)):
            occ_img_mats[i] = work_func(i)
    else:
        for i, m in enumerate(parallel_map(
            work_func, range(len(masks)),
            cores=threads,
            use_multiprocessing=False,
        )):
            occ_img_mats[i] = m

    return occ_img_mats


def occlude_image_streaming(
    ref_image: np.ndarray,
    masks: Iterable[np.ndarray],
    fill: Optional[Union[int, Sequence[int], np.ndarray]] = None,
    threads: Optional[int] = None,
) -> Generator[np.ndarray, None, None]:
    """
    Apply a number of input occlusion masks to the given reference image,
    producing a list of images equivalent in length, and parallel in order, to
    the input masks.
    This streaming version will return an iterator that yields occluded image
    matrices.

    We expect the "mask" matrices and the image to be the same height and
    width, and for the mask matrix values to be in the [0, 1] range.
    In the mask matrix, values closer to 1 correspond to regions of the image
    that should *NOT* be occluded.
    E.g. a 0 in the mask will translate to *fully* occluding the corresponding
    location in the source image.

    We optionally take in a "fill" that alpha-blend into masked regions of the
    input `ref_image`.
    `fill` may be either a scalar, sequence of scalars, or another image matrix
    congruent in shape to the `ref_image`.
    When `fill` is a scalar or a sequence of scalars, the scalars should be in
    the same data-type and value range as the input image.
    A sequence of scalars should be the same length as there are channels in
    the `ref_image`.
    When `fill` is an image matrix it should follow the format of `[H x W]` or
    `[H x W x C]`, should be in the same dtype and value range as `ref_image`
    and should match the same number of channels if channels are provided.
    When no fill is passed, black is used (default absence of color).

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
    :param fill: Optional fill for alpha-blending based on the input masks for
        the occluded regions as a scalar value, a per-channel sequence or a
        shape-matched image.
    :param threads: Optional number of threads to use for parallelism when set
        to a positive integer. If 0, a negative value, or `None`, work will be
        performed on the main-thread in-line.

    :raises ValueError: One or more input masks in the input iterable did not
        match shape of the input reference image.

    :return: A generator of numpy array masked images.
    """
    # Just the [H x W] component.
    img_shape = ref_image.shape[:2]
    s: Tuple = (...,)
    if ref_image.ndim > 2:
        s = (..., None)  # add channel axis for multiplication

    def work_func(i_: int, m: np.ndarray) -> np.ndarray:
        m_shape = m.shape
        if m_shape != img_shape:
            raise ValueError(
                f"Input mask (position {i_}) did not the shape of the input "
                f"image: {m_shape} != {img_shape}"
            )
        img_m = np.empty_like(ref_image)
        if fill is not None:
            np.add(
                (m[s] * ref_image),
                ((UINT8_ONE - m[s]) * fill),
                out=img_m, casting="unsafe"
            )
        else:
            np.multiply(m[s], ref_image, out=img_m, casting="unsafe")
        return img_m

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


def benchmark_occlude_image(
    img_shape: Tuple[int, int] = (224, 224),
    img_channels: int = 3,
    num_masks: int = 1000,
    threading_tests: Sequence[int] = (0, 1, 2),
) -> None:
    """
    Simple benchmark for the two above `occlude_image_*` functions above w.r.t.
    the given reference image matrix, which should be of the shape
    `[H x W [x C]]`.
    """
    img_mat = np.ones((*img_shape, img_channels), dtype=np.uint8)
    masks = (np.random.rand(num_masks, *img_shape[:2]) < 0.5)
    fill_1c: int = 0
    fill_mc = [0] * img_channels
    perf_counter = time.perf_counter
    print(f"Image shape={img_mat.shape}, masks={masks.shape}, fill_1c={fill_1c}, fill_{img_channels}c={fill_mc}")

    s = perf_counter()
    occlude_image_batch(img_mat, masks)
    e = perf_counter()
    print(f"Batch - no-fill - {e-s} s")
    for threads in threading_tests:
        s = perf_counter()
        occlude_image_batch(img_mat, masks, threads=threads)
        e = perf_counter()
        print(f"Batch - threads={threads:2d} - no-fill - {e-s} s")
    for threads in threading_tests:
        s = perf_counter()
        np.asarray(list(occlude_image_streaming(img_mat, masks, threads=threads)))
        e = perf_counter()
        print(f"Streaming - threads={threads:2d} - no-fill - {e-s} s")

    s = perf_counter()
    occlude_image_batch(img_mat, masks, fill=fill_1c)
    e = perf_counter()
    print(f"Batch - fill-1c - {e-s} s")
    for threads in threading_tests:
        s = perf_counter()
        occlude_image_batch(img_mat, masks, fill=fill_1c, threads=threads)
        e = perf_counter()
        print(f"Batch - threads={threads:2d} - fill-1c - {e-s} s")
    for threads in threading_tests:
        s = perf_counter()
        np.asarray(list(occlude_image_streaming(img_mat, masks, fill=fill_1c, threads=threads)))
        e = perf_counter()
        print(f"Streaming - threads={threads:2d} - fill-1c - {e-s} s")

    s = perf_counter()
    occlude_image_batch(img_mat, masks, fill=fill_mc)
    e = perf_counter()
    print(f"Batch - fill-{img_channels}c - {e-s} s")
    for threads in threading_tests:
        s = perf_counter()
        occlude_image_batch(img_mat, masks, fill=fill_mc, threads=threads)
        e = perf_counter()
        print(f"Batch - threads={threads:2d} - fill-{img_channels}c - {e-s} s")
    for threads in threading_tests:
        s = perf_counter()
        np.asarray(list(occlude_image_streaming(img_mat, masks, fill=fill_mc, threads=threads)))
        e = perf_counter()
        print(f"Streaming - threads={threads:2d} - fill-{img_channels}c - {e-s} s")


def weight_regions_by_scalar(
    scalar_vec: np.ndarray,
    masks: np.ndarray,
    inv_masks: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Weight some binary masks region with its respective vector in scalar_vec.

    We expect the "masks" matrices and the image to be the same height and
    width, and be valued in the [0, 1] floating-point range. The length
    and order of scalar_vec and masks are assumed to be the same.
    In the mask matrix, higher values correspond to regions of the image that
    preserved. E.g. a 0 in the mask will translate to blacking out the
    corresponding location in the source image.

    We can optionally per-pixel normalize the weighted sum by the sum of masks.
    E.g. if some region is covered by more masks than others, that region's
    sum is down-weighted.

    :param scalar_vec: Weights for image regions for nClasses and shape
                       [nMasks, nClasses]
    :param masks: Mask array in the [nMasks, Height, Weight] shape format.
    :param inv_masks: Boolean flag to apply the `scalar_vec` to the inverse of
        the masks, i.e. `(1 - masks)`. If False, we more simply apply to just
        `masks`.
    :param normalize: If the output heatmap should be per-pixel normalized by
        mask coverage. E.g. if some region is covered by more masks than
        others, then the weighted sum

    :return: A numpy array representing the weighted heatmap.
    """
    # upcast to common type
    if scalar_vec.dtype < masks.dtype:
        scalar_vec = scalar_vec.astype(max(scalar_vec.dtype, masks.dtype))
    elif masks.dtype < scalar_vec.dtype:
        masks = masks.astype(max(scalar_vec.dtype, masks.dtype))

    if inv_masks:
        masks = (UINT8_ONE - masks)

    # initialize final saliency maps
    sal_across_masks = np.zeros(
        (scalar_vec.shape[1], masks.shape[1], masks.shape[2]),
        dtype=max(scalar_vec.dtype, masks.dtype)
    )

    # split weights per class
    for i, class_scales in enumerate(np.transpose(scalar_vec)):
        # aggregate scaled masks
        for (mask, scalar) in zip(masks, class_scales):
            sal_across_masks[i] += mask * scalar

    if normalize:
        # Removing regions that are never masked to avoid a dividebyzero warning
        mask_sum = masks.sum(axis=0)
        mask_sum[mask_sum == 0] = 1.

        # Compute final saliency map by normalizing with sampling factor.
        sal_across_masks /= mask_sum

    return sal_across_masks
