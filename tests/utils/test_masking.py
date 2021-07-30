import functools
from typing import Callable

import numpy as np
import numpy.typing as npt
import pytest

from xaitk_saliency.utils.masking import (
    occlude_image_batch,
    occlude_image_streaming,
    benchmark_occlude_image,
)


@pytest.fixture(
    scope='module',
    ids=["batch", "streaming-serial", "streaming-parallel"],
    params=[
        occlude_image_batch,
        functools.partial(occlude_image_streaming, threads=None),
        functools.partial(occlude_image_streaming, threads=4),
    ]
)
def occ_func(request: pytest.FixtureRequest) -> Callable:
    """
    Module-local fixture for parameterizing tests across occlusion
    function variants.
    """
    # param is "optional" in FixtureRequest, so it's not resolving as a "valid"
    # attribute (SubRequest is the "real" type here but is seemingly a private
    # class).
    return request.param  # type: ignore


def as_uint8(v: npt.ArrayLike) -> np.ndarray:
    """
    Convert input value to np.uint8 type. This only exists to make
    type-checking happy by using `np.asarray(..., dtype=np.uint8)` instead
    of `np.uint8(v)`, in a wrapped func to reduce typing.
    """
    return np.asarray(v, dtype=np.uint8)


class TestOccludeImageCommon:
    """
    Common tests for both batch and streaming occlusion methods since each
    should output the same results for these.
    """

    def test_gray_bool_no_fill(self, occ_func: Callable) -> None:
        """
        Exercise the combo of gray image and boolean masks.
        """
        # Simple mult against bool masks is expect image in our white input case.
        expected_images = TEST_MASKS_BOOL * as_uint8(255)
        res_images = list(occ_func(TEST_IMAGE_GRAY, TEST_MASKS_BOOL))
        assert np.allclose(res_images, expected_images)

    def test_rgb_bool_no_fill(self, occ_func: Callable) -> None:
        """
        Exercise the combo of RGB image and boolean masks.
        """
        # Simple mult against bool masks is expect image in our white input case.
        expected_images = TEST_MASKS_BOOL[..., None] * as_uint8([255, 255, 255])
        res_images = list(occ_func(TEST_IMAGE_RGB, TEST_MASKS_BOOL))
        assert np.allclose(res_images, expected_images)

    def test_gray_float_no_fill(self, occ_func: Callable) -> None:
        """
        Exercise the combo of gray image and float masks.
        """
        # Simple mult against bool masks is expect image in our white input case.
        expected_images = as_uint8(TEST_MASKS_FLOAT * 255)
        res_images = list(occ_func(TEST_IMAGE_GRAY, TEST_MASKS_FLOAT))
        assert np.allclose(res_images, expected_images)

    def test_rgb_float_no_fill(self, occ_func: Callable) -> None:
        """
        Exercise the combo of RGB image and float masks.
        """
        # Simple mult against bool masks is expect image in our white input case.
        expected_images = as_uint8(TEST_MASKS_FLOAT[..., None] * [255, 255, 255])
        res_images = list(occ_func(TEST_IMAGE_RGB, TEST_MASKS_FLOAT))
        assert np.allclose(res_images, expected_images)

    def test_gray_bool_fill_scalar(self, occ_func: Callable) -> None:
        """
        Test using a custom fill color for the masked regions.
        """
        res_images = list(occ_func(TEST_IMAGE_GRAY, TEST_MASKS_BOOL, 44))
        # Known alpha blending
        expected_images = as_uint8([
            [
                [255, 255, 44, 44, 44],
                [255, 255, 44, 44, 44],
                [255, 255, 44, 44, 44],
                [44, 44, 255, 255, 44],
                [255, 44, 44, 44, 255],
            ],
            [
                [44, 44, 255, 255, 44],
                [44, 44, 255, 255, 44],
                [255, 255, 44, 44, 44],
                [255, 255, 44, 44, 44],
                [44, 44, 44, 255, 255]
            ],
        ])
        assert np.allclose(res_images, expected_images)

    def test_gray_bool_fill_list_error(self, occ_func: Callable) -> None:
        """
        Test using a custom fill color for masked regions as a 3-channel list.
        """
        with pytest.raises(ValueError, match=r"operands could not be broadcast together"):
            # This should error because the input is single channel while the
            # fill value is 3-channel
            list(occ_func(TEST_IMAGE_GRAY, TEST_MASKS_BOOL, [255, 0, 255]))

    def test_gray_float_fill_scalar(self, occ_func: Callable) -> None:
        # Using 5 because it exposes a detail about float-level accumulation.
        res_images = list(occ_func(TEST_IMAGE_GRAY, TEST_MASKS_FLOAT, 5))
        expected_images = as_uint8([
            [
                [255, 192, 130,  67,   5],
                [192, 130,  67,   5,  67],
                [130,  67,   5,  67, 130],
                [67,    5,  67, 130, 192],
                [5,    67, 130, 192, 255]
            ],
            [
                [5,    67, 130, 192, 255],
                [67,  130, 192, 255, 192],
                [130, 192, 255, 192, 130],
                [192, 255, 192, 130,  67],
                [255, 192, 130,  67,   5]
            ],
        ])
        assert np.allclose(res_images, expected_images)

    def test_gray_float_fill_list_error(self, occ_func: Callable) -> None:
        with pytest.raises(ValueError, match=r"operands could not be broadcast together"):
            # This should error because the input is single channel while the
            # fill value is 3-channel
            list(occ_func(TEST_IMAGE_GRAY, TEST_MASKS_FLOAT, [1, 2, 3]))

    def test_rgb_bool_fill_scalar(self, occ_func: Callable) -> None:
        res_images = list(occ_func(TEST_IMAGE_RGB, TEST_MASKS_BOOL, 44))
        expected_images = as_uint8([
            [
                [[255, 255, 255], [255, 255, 255], [44, 44, 44], [44, 44, 44], [44, 44, 44]],
                [[255, 255, 255], [255, 255, 255], [44, 44, 44], [44, 44, 44], [44, 44, 44]],
                [[255, 255, 255], [255, 255, 255], [44, 44, 44], [44, 44, 44], [44, 44, 44]],
                [[44, 44, 44], [44, 44, 44], [255, 255, 255], [255, 255, 255], [44, 44, 44]],
                [[255, 255, 255], [44, 44, 44], [44, 44, 44], [44, 44, 44], [255, 255, 255]],
            ],
            [
                [[44, 44, 44], [44, 44, 44], [255, 255, 255], [255, 255, 255], [44, 44, 44]],
                [[44, 44, 44], [44, 44, 44], [255, 255, 255], [255, 255, 255], [44, 44, 44]],
                [[255, 255, 255], [255, 255, 255], [44, 44, 44], [44, 44, 44], [44, 44, 44]],
                [[255, 255, 255], [255, 255, 255], [44, 44, 44], [44, 44, 44], [44, 44, 44]],
                [[44, 44, 44], [44, 44, 44], [44, 44, 44], [255, 255, 255], [255, 255, 255]],
            ]
        ])
        assert np.allclose(res_images, expected_images)

    def test_rgb_bool_fill_list(self, occ_func: Callable) -> None:
        # Let's use half-magenta because why not
        res_images = list(occ_func(TEST_IMAGE_RGB, TEST_MASKS_BOOL, [128, 0, 128]))
        expected_images = as_uint8([
            [
                [[255, 255, 255], [255, 255, 255], [128, 0, 128], [128, 0, 128], [128, 0, 128]],
                [[255, 255, 255], [255, 255, 255], [128, 0, 128], [128, 0, 128], [128, 0, 128]],
                [[255, 255, 255], [255, 255, 255], [128, 0, 128], [128, 0, 128], [128, 0, 128]],
                [[128, 0, 128], [128, 0, 128], [255, 255, 255], [255, 255, 255], [128, 0, 128]],
                [[255, 255, 255], [128, 0, 128], [128, 0, 128], [128, 0, 128], [255, 255, 255]],
            ],
            [
                [[128, 0, 128], [128, 0, 128], [255, 255, 255], [255, 255, 255], [128, 0, 128]],
                [[128, 0, 128], [128, 0, 128], [255, 255, 255], [255, 255, 255], [128, 0, 128]],
                [[255, 255, 255], [255, 255, 255], [128, 0, 128], [128, 0, 128], [128, 0, 128]],
                [[255, 255, 255], [255, 255, 255], [128, 0, 128], [128, 0, 128], [128, 0, 128]],
                [[128, 0, 128], [128, 0, 128], [128, 0, 128], [255, 255, 255], [255, 255, 255]],
            ]
        ])
        assert np.allclose(res_images, expected_images)

    def test_rgb_float_fill_scalar(self, occ_func: Callable) -> None:
        # Using 5 because it exposes a detail about float-level accumulation.
        res_images = list(occ_func(TEST_IMAGE_RGB, TEST_MASKS_FLOAT, 5))
        expected_images = as_uint8([
            [
                [[255, 255, 255], [192, 192, 192], [130, 130, 130], [67, 67, 67], [5, 5, 5]],
                [[192, 192, 192], [130, 130, 130], [67, 67, 67], [5, 5, 5], [67, 67, 67]],
                [[130, 130, 130], [67, 67, 67], [5, 5, 5], [67, 67, 67], [130, 130, 130]],
                [[67, 67, 67], [5, 5, 5], [67, 67, 67], [130, 130, 130], [192, 192, 192]],
                [[5, 5, 5], [67, 67, 67], [130, 130, 130], [192, 192, 192], [255, 255, 255]]
            ],
            [
                [[5, 5, 5], [67, 67, 67], [130, 130, 130], [192, 192, 192], [255, 255, 255]],
                [[67, 67, 67], [130, 130, 130], [192, 192, 192], [255, 255, 255], [192, 192, 192]],
                [[130, 130, 130], [192, 192, 192], [255, 255, 255], [192, 192, 192], [130, 130, 130]],
                [[192, 192, 192], [255, 255, 255], [192, 192, 192], [130, 130, 130], [67, 67, 67]],
                [[255, 255, 255], [192, 192, 192], [130, 130, 130], [67, 67, 67], [5, 5, 5]]
            ]
        ])
        assert np.allclose(res_images, expected_images)

    def test_rgb_float_fill_list(self, occ_func: Callable) -> None:
        # Using [5,7,9] here because it exposes a detail about float-level
        # accumulation.
        res_images = list(occ_func(TEST_IMAGE_RGB, TEST_MASKS_FLOAT, [5, 7, 9]))
        expected_images = as_uint8([
            [
                [[255, 255, 255], [192, 193, 193], [130, 131, 132], [67, 69, 70], [5, 7, 9]],
                [[192, 193, 193], [130, 131, 132], [67, 69, 70], [5, 7, 9], [67, 69, 70]],
                [[130, 131, 132], [67, 69, 70], [5, 7, 9], [67, 69, 70], [130, 131, 132]],
                [[67, 69, 70], [5, 7, 9], [67, 69, 70], [130, 131, 132], [192, 193, 193]],
                [[5, 7, 9], [67, 69, 70], [130, 131, 132], [192, 193, 193], [255, 255, 255]]
            ],
            [
                [[5, 7, 9], [67, 69, 70], [130, 131, 132], [192, 193, 193], [255, 255, 255]],
                [[67, 69, 70], [130, 131, 132], [192, 193, 193], [255, 255, 255], [192, 193, 193]],
                [[130, 131, 132], [192, 193, 193], [255, 255, 255], [192, 193, 193], [130, 131, 132]],
                [[192, 193, 193], [255, 255, 255], [192, 193, 193], [130, 131, 132], [67, 69, 70]],
                [[255, 255, 255], [192, 193, 193], [130, 131, 132], [67, 69, 70], [5, 7, 9]]
            ]
        ])
        assert np.allclose(res_images, expected_images)


class TestOccludeImageBatch:

    def test_catch_bad_masks_dim(self) -> None:
        """
        Test the expectation that input mask matrices need to be 3 dimensional
        for the [N x H x W] shape.
        """
        with pytest.raises(ValueError,  match="Expected a 3-dimension mask input"):
            # WHAT IF WE PUT IN ONE MASK GUYS!
            occlude_image_batch(TEST_IMAGE_GRAY, np.ones((5, 5)))

    def test_catch_bad_mask_shape(self) -> None:
        """
        Test catching input masks that do not have the same shape as the input
        ref image.
        """
        with pytest.raises(ValueError, match="Input image shape and mask image shape did not match"):
            occlude_image_batch(TEST_IMAGE_GRAY, np.ones((3, 4, 2)))


class TestOccludeImageStreaming:

    def test_catch_bad_mask_shape(self) -> None:
        """
        Test catching input masks that do not have the same shape as the input
        ref image.
        """
        with pytest.raises(ValueError,  match=r"Input mask \(position 0\) did not the shape of the input image"):
            # Giving just make will cause the first dim to seem to be the
            # iteration axis, so 1D vectors will be input as "masks"
            # incorrectly.
            list(occlude_image_streaming(TEST_IMAGE_GRAY, np.ones((5, 5))))

        with pytest.raises(ValueError, match=r"Input mask \(position 0\) did not the shape of the input image"):
            list(occlude_image_streaming(TEST_IMAGE_GRAY, np.ones((3, 4, 2))))

        # List with non-zero position?
        with pytest.raises(ValueError, match=r"Input mask \(position 2\) did not the shape of the input image"):
            list(occlude_image_streaming(
                TEST_IMAGE_GRAY,
                [
                    np.ones((5, 5)),
                    np.ones((5, 5)),
                    np.ones((2, 8)),
                    np.ones((5, 5)),
                    np.ones((5, 5)),
                ]
            ))


def test_benchmark() -> None:
    """
    Simple run test of the benchmark function.
    """
    benchmark_occlude_image(threading_tests=[0, 1, 2])


TEST_IMAGE_GRAY = np.full(
    (5, 5), 255, dtype=np.uint8
)
TEST_IMAGE_RGB = np.full(
    (5, 5, 3), 255, dtype=np.uint8
)
TEST_MASKS_BOOL = np.asarray([
    [[1, 1, 0, 0, 0],
     [1, 1, 0, 0, 0],
     [1, 1, 0, 0, 0],
     [0, 0, 1, 1, 0],
     [1, 0, 0, 0, 1]],
    [[0, 0, 1, 1, 0],
     [0, 0, 1, 1, 0],
     [1, 1, 0, 0, 0],
     [1, 1, 0, 0, 0],
     [0, 0, 0, 1, 1]]
], dtype=bool)
TEST_MASKS_FLOAT = np.asarray([
    [[1.00, 0.75, 0.50, 0.25, 0.00],
     [0.75, 0.50, 0.25, 0.00, 0.25],
     [0.50, 0.25, 0.00, 0.25, 0.50],
     [0.25, 0.00, 0.25, 0.50, 0.75],
     [0.00, 0.25, 0.50, 0.75, 1.00]],
    [[0.00, 0.25, 0.50, 0.75, 1.00],
     [0.25, 0.50, 0.75, 1.00, 0.75],
     [0.50, 0.75, 1.00, 0.75, 0.50],
     [0.75, 1.00, 0.75, 0.50, 0.25],
     [1.00, 0.75, 0.50, 0.25, 0.00]],
], dtype=np.float32)
