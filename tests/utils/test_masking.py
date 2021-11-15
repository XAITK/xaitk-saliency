import functools
from typing import Callable

import numpy as np
import numpy.typing as npt
import pytest

from xaitk_saliency.utils.masking import (
    occlude_image_batch,
    occlude_image_streaming,
    benchmark_occlude_image,
    weight_regions_by_scalar,
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

    def test_gray_bool_fill_list_error(self, occ_func: Callable) -> None:
        """
        Test using a custom fill color for masked regions as a 3-channel list.
        For single-channel input this is an error.
        """
        with pytest.raises(ValueError, match=r"operands could not be broadcast together"):
            # This should error because the input is single channel while the
            # fill value is 3-channel
            list(occ_func(TEST_IMAGE_GRAY, TEST_MASKS_BOOL, [255, 0, 255]))

    def test_gray_float_fill_list_error(self, occ_func: Callable) -> None:
        """
        Test using a custom fill color for masked regions as a 3-channel list.
        For single-channel input this is an error.
        """
        with pytest.raises(ValueError, match=r"operands could not be broadcast together"):
            # This should error because the input is single channel while the
            # fill value is 3-channel
            list(occ_func(TEST_IMAGE_GRAY, TEST_MASKS_FLOAT, [1, 2, 3]))

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

    # fill_image permute with
    # image: gray, rgb
    # masks: bool, float
    def test_gray_bool_fill_img_gray(self, occ_func: Callable) -> None:
        # combination of TEST_MASKS_BOOL with TEST_FILL_IMG_GRAY
        expected_images = as_uint8([
            [[255, 255,  32,  48,  64],
             [255, 255,  48,  64,  80],
             [255, 255,  64,  80,  96],
             [48,   64, 255, 255, 112],
             [255,  80,  96, 112, 255]],
            [[0,    16, 255, 255,  64],
             [16,   32, 255, 255,  80],
             [255, 255,  64,  80,  96],
             [255, 255,  80,  96, 112],
             [64,   80,  96, 255, 255]]
        ])
        res_images = list(occ_func(
            TEST_IMAGE_GRAY,
            TEST_MASKS_BOOL,
            fill=TEST_FILL_IMG_GRAY
        ))
        assert np.allclose(res_images, expected_images)

    def test_gray_bool_fill_img_rgb_error(self, occ_func: Callable) -> None:
        """
        Filling with a multi-channel image when the ref-image is single-channel
        should be an error.
        """
        with pytest.raises(
            ValueError,
            match=r"operands could not be broadcast together with shapes "
                  r"\((2,)?5,5\) \(5,5,3\)"
        ):
            list(occ_func(
                TEST_IMAGE_GRAY,
                TEST_MASKS_BOOL,
                fill=TEST_FILL_IMG_RGB
            ))

    def test_rgb_bool_fill_img_gray_error(self, occ_func: Callable) -> None:
        """
        Filling with a single-channel image when the ref-image is multi-channel
        should be an error.
        """
        with pytest.raises(
            ValueError,
            match=r"operands could not be broadcast together with shapes "
                  r"\((2,)?5,5,3\) \((2,)?5,5,5\) \((2,)?5,5,3\)"
        ):
            list(occ_func(
                TEST_IMAGE_RGB,
                TEST_MASKS_BOOL,
                fill=TEST_FILL_IMG_GRAY
            ))

    def test_rgb_bool_fill_img_rgb(self, occ_func: Callable) -> None:
        res_images = list(occ_func(
            TEST_IMAGE_RGB,
            TEST_MASKS_BOOL,
            fill=TEST_FILL_IMG_RGB,
        ))
        expected_images = as_uint8([
            [[[255, 255, 255], [255, 255, 255], [32, 32, 32], [48, 48, 48], [64, 64, 64]],
             [[255, 255, 255], [255, 255, 255], [48, 48, 48], [64, 64, 64], [80, 80, 80]],
             [[255, 255, 255], [255, 255, 255], [64, 64, 64], [80, 80, 80], [96, 96, 96]],
             [[48, 48, 48], [64, 64, 64], [255, 255, 255], [255, 255, 255], [112, 112, 112]],
             [[255, 255, 255], [80, 80, 80], [96, 96, 96], [112, 112, 112], [255, 255, 255]]],
            [[[0, 0, 0], [16, 16, 16], [255, 255, 255], [255, 255, 255], [64, 64, 64]],
             [[16, 16, 16], [32, 32, 32], [255, 255, 255], [255, 255, 255], [80, 80, 80]],
             [[255, 255, 255], [255, 255, 255], [64, 64, 64], [80, 80, 80], [96, 96, 96]],
             [[255, 255, 255], [255, 255, 255], [80, 80, 80], [96, 96, 96], [112, 112, 112]],
             [[64, 64, 64], [80, 80, 80], [96, 96, 96], [255, 255, 255], [255, 255, 255]]]
        ])
        assert np.allclose(res_images, expected_images)

    def test_gray_float_fill_img(self, occ_func: Callable) -> None:
        res_images = list(occ_func(
            TEST_IMAGE_GRAY,
            TEST_MASKS_FLOAT,
            fill=TEST_FILL_IMG_GRAY
        ))
        expected_images = as_uint8([
            [[255, 195, 143, 99, 64],
             [195, 143, 99, 64, 123],
             [143, 99, 64, 123, 175],
             [99, 64, 123, 175, 219],
             [64, 123, 175, 219, 255]],
            [[0, 75, 143, 203, 255],
             [75, 143, 203, 255, 211],
             [143, 203, 255, 211, 175],
             [203, 255, 211, 175, 147],
             [255, 211, 175, 147, 128]]
        ])
        assert np.allclose(res_images, expected_images)

    def test_rgb_float_fill_img(self, occ_func: Callable) -> None:
        res_images = list(occ_func(
            TEST_IMAGE_RGB,
            TEST_MASKS_FLOAT,
            fill=TEST_FILL_IMG_RGB
        ))
        expected_images = as_uint8([
            [[[255, 255, 255], [195, 195, 195], [143, 143, 143], [99, 99, 99], [64, 64, 64]],
             [[195, 195, 195], [143, 143, 143], [99, 99, 99], [64, 64, 64], [123, 123, 123]],
             [[143, 143, 143], [99, 99, 99], [64, 64, 64], [123, 123, 123], [175, 175, 175]],
             [[99, 99, 99], [64, 64, 64], [123, 123, 123], [175, 175, 175], [219, 219, 219]],
             [[64, 64, 64], [123, 123, 123], [175, 175, 175], [219, 219, 219], [255, 255, 255]]],
            [[[0, 0, 0], [75, 75, 75], [143, 143, 143], [203, 203, 203], [255, 255, 255]],
             [[75, 75, 75], [143, 143, 143], [203, 203, 203], [255, 255, 255], [211, 211, 211]],
             [[143, 143, 143], [203, 203, 203], [255, 255, 255], [211, 211, 211], [175, 175, 175]],
             [[203, 203, 203], [255, 255, 255], [211, 211, 211], [175, 175, 175], [147, 147, 147]],
             [[255, 255, 255], [211, 211, 211], [175, 175, 175], [147, 147, 147], [128, 128, 128]]]
        ])
        assert np.allclose(res_images, expected_images)

    def test_fill_img_bad_height_width(self, occ_func: Callable) -> None:
        """
        Test that inputting a fill image with inconsistent height, width or
        both with respect to the ref image is a ValueError.
        """
        fill_img = np.full(
            (3, 4), 255, dtype=np.uint8
        )
        with pytest.raises(
            ValueError,
            # ()? in the match because in streaming individual images are
            # broadcast.
            match=r"operands could not be broadcast together with shapes "
                  r"\((2,)?5,5\) \(3,4\)"
        ):
            list(occ_func(TEST_IMAGE_GRAY, TEST_MASKS_BOOL, fill=fill_img))

    def test_fill_img_bad_channels(self, occ_func: Callable) -> None:
        fill_img = np.full(
            (5, 5, 4), 255, dtype=np.uint8
        )
        with pytest.raises(
            ValueError,
            # ()? in the match because in streaming individual images are
            # broadcast.
            match=r"operands could not be broadcast together with shapes "
                  r"\((2,)?5,5,3\) \((2,)?5,5,4\) \((2,)?5,5,3\)"
        ):
            list(occ_func(TEST_IMAGE_RGB, TEST_MASKS_BOOL, fill=fill_img))


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


class TestWeightRegionsByScalar:

    @pytest.mark.parametrize("inv_masks", [True, False])
    @pytest.mark.parametrize("normalize", [True, False])
    @pytest.mark.parametrize(
        "scalar_type,mask_type,expected_output_type",
        [
            (np.float32, np.bool_, np.float32),
            (np.float32, np.int16, np.float32),
            (np.float32, np.float32, np.float32),
            (np.float64, np.float32, np.float64),
            (np.float32, np.float64, np.float64),
        ]
    )
    def test_against_type_upcasting(
        self, scalar_type: np.dtype, mask_type: np.dtype,
        expected_output_type: np.dtype, inv_masks: bool, normalize: bool
    ) -> None:
        """
        Test that the output is not of a type that is larger than is input.
        In other words, the output should follow numpy's type promotion rules
        based on the input data types.
        E.g. when input is float32, output is *not* float64, but still float32.
        """
        scalar_vec = np.random.randn(100, 10).astype(scalar_type)
        masks = np.ones((100, 224, 224)).astype(mask_type)
        output = weight_regions_by_scalar(scalar_vec, masks, inv_masks, normalize)
        assert output.dtype == expected_output_type


# Test input images
TEST_IMAGE_GRAY = np.full(
    (5, 5), 255, dtype=np.uint8
)
TEST_IMAGE_RGB = np.full(
    (5, 5, 3), 255, dtype=np.uint8
)
# Test mask inputs
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
# Test "fill" images for alpha-blending
TEST_FILL_IMG_GRAY = np.array([
    [0,  16, 32,  48,  64],
    [16, 32, 48,  64,  80],
    [32, 48, 64,  80,  96],
    [48, 64, 80,  96, 112],
    [64, 80, 96, 112, 128],
], dtype=np.uint8)
TEST_FILL_IMG_RGB = np.stack([TEST_FILL_IMG_GRAY]*3).transpose(1, 2, 0)
