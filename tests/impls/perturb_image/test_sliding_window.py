import numpy as np
from smqtk_core.configuration import configuration_test_helper

from tests import EXPECTED_MASKS_4x6
from xaitk_saliency import PerturbImage
from xaitk_saliency.impls.perturb_image.sliding_window import SlidingWindow


class TestOcclusionBasedPerturb:
    def test_init_default(self) -> None:
        """Test empty construction since we provide defaults."""
        impl = SlidingWindow()
        assert impl.window_size == (50, 50)
        assert impl.stride == (20, 20)

    def test_init_valued(self) -> None:
        """Test that constructor values pass."""
        ex_w = (777, 776)
        ex_s = (444, 445)
        impl = SlidingWindow(window_size=ex_w, stride=ex_s)
        assert impl.window_size == ex_w
        assert impl.stride == ex_s

    def test_plugin_find(self) -> None:
        """
        This implementation has no optional plugins so it should be found and
        exposed by the super-type's impl getter.
        """
        assert SlidingWindow in PerturbImage.get_impls()

    def test_standard_config(self) -> None:
        ex_w = (777, 776)
        ex_s = (444, 445)
        impl = SlidingWindow(window_size=ex_w, stride=ex_s)
        for inst in configuration_test_helper(impl):
            assert inst.window_size == ex_w
            assert inst.stride == ex_s

    def test_perturb_1channel(self) -> None:
        """
        Test basic perturbation on a known image with even windowing + stride.
        Input image mode should not impact the masks output.
        """
        # Image is slightly wide, should be occluded 6-ways.
        # Simulate grayscale image (single-channel)
        white_image = np.full((4, 6), fill_value=255, dtype=np.uint8)

        impl = SlidingWindow(window_size=(2, 2), stride=(2, 2))
        actual_masks = impl.perturb(white_image)

        assert np.allclose(actual_masks, EXPECTED_MASKS_4x6)

    def test_perturb_3channel(self) -> None:
        """
        Test basic perturbation on a known image with even windowing + stride.
        Input image mode should not impact the masks output.
        """
        # Image is slightly wide, should be occluded 6-ways.
        # Simulate RGB or BGR image (3-channel)
        white_image = np.full((4, 6, 3), fill_value=255, dtype=np.uint8)

        impl = SlidingWindow(window_size=(2, 2), stride=(2, 2))
        actual_masks = impl.perturb(white_image)

        assert np.allclose(actual_masks, EXPECTED_MASKS_4x6)

    def test_perturb_3channel_nonsquare(self) -> None:
        """
        Test basic perturbation on a known image with non-square window +
        stride.
        Input image mode should not impact the masks output.
        """
        # Square image for uneven masking 6-ways.
        # Simulate RGB or BGR image (3-channel)
        white_image = np.full((6, 6, 3), fill_value=255, dtype=np.uint8)

        impl = SlidingWindow(window_size=(3, 2), stride=(3, 2))
        actual_masks = impl.perturb(white_image)

        assert np.allclose(actual_masks, EXPECTED_MASKS_6x6_rect)

    def test_perturb_4channel(self) -> None:
        """
        Test basic perturbation on a known image with even windowing + stride.
        Input image mode should not impact the masks output.
        """
        # Image is slightly wide, should be occluded 6-ways.
        # Simulate RGBA format image (4-channel)
        white_image = np.full((4, 6, 4), fill_value=255, dtype=np.uint8)

        impl = SlidingWindow(window_size=(2, 2), stride=(2, 2))
        actual_masks = impl.perturb(white_image)

        assert np.allclose(actual_masks, EXPECTED_MASKS_4x6)

    def test_window_size_agnostic(self) -> None:
        """
        Test that the number of masks remains the same, with  a given stride,
        as window size changes.
        """
        img = np.empty((21, 21))

        impl_2x2 = SlidingWindow(window_size=(3, 2), stride=(2, 2))
        masks_2x2 = impl_2x2.perturb(img)

        impl_3x3 = SlidingWindow(window_size=(244, 72), stride=(2, 2))
        masks_3x3 = impl_3x3.perturb(img)

        assert len(masks_2x2) == len(masks_3x3)


# Common expected masks for 6x6 tests
EXPECTED_MASKS_6x6_rect = np.array(
    [
        [
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        [
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
        ],
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
        ],
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
        ],
    ],
    dtype=bool,
)
