from unittest import TestCase

import PIL.Image
import numpy as np
from smqtk_core.configuration import configuration_test_helper

from xaitk_saliency import PerturbImage
from xaitk_saliency.impls.perturb_image.sliding_window import SlidingWindowPerturb


class TestOcclusionBasedPerturb (TestCase):

    def test_init_default(self) -> None:
        """
        Test empty construction since we provide defaults.
        """
        impl = SlidingWindowPerturb()
        assert impl.window_size == (50, 50)
        assert impl.stride == (20, 20)

    def test_init_valued(self) -> None:
        """
        Test that constructor values pass.
        """
        ex_w = (777, 776)
        ex_s = (444, 445)
        impl = SlidingWindowPerturb(window_size=ex_w, stride=ex_s)
        assert impl.window_size == ex_w
        assert impl.stride == ex_s

    def test_plugin_find(self) -> None:
        """
        This implementation has no optional plugins so it should be found and
        exposed by the super-type's impl getter.
        """
        assert SlidingWindowPerturb in PerturbImage.get_impls()

    def test_standard_config(self) -> None:
        ex_w = (777, 776)
        ex_s = (444, 445)
        ex_threads = 86
        impl = SlidingWindowPerturb(window_size=ex_w, stride=ex_s, threads=ex_threads)
        for inst in configuration_test_helper(impl):
            assert inst.window_size == ex_w
            assert inst.stride == ex_s
            assert inst.threads == ex_threads

    def test_perturb_1channel(self) -> None:
        """
        Test basic perturbation on a known image with even windowing + stride.
        """
        impl = SlidingWindowPerturb(window_size=(2, 2), stride=(2, 2))
        # Image is slightly wide, should be occluded 6-ways.
        white_image = PIL.Image.fromarray(
            np.full((4, 6), fill_value=255, dtype=np.uint8)
        )
        assert white_image.mode == "L"
        expected_perturbed_images = list(map(
            PIL.Image.fromarray,
            (EXPECTED_MASKS_4x6 * 255).astype(np.uint8)
        ))
        total_perts = []
        for i, (img, mask) in enumerate(impl.perturb(white_image)):
            assert img.mode == "L"
            assert img == expected_perturbed_images[i]
            assert np.allclose(mask, EXPECTED_MASKS_4x6[i])
            total_perts.append(img)
        assert len(total_perts) == 6

    def test_perturb_3channel(self) -> None:
        """
        Test basic perturbation on a known image with even windowing + stride.
        """
        impl = SlidingWindowPerturb(window_size=(2, 2), stride=(2, 2))
        # Image is slightly wide, should be occluded 6-ways.
        white_image = PIL.Image.fromarray(
            np.full((4, 6, 3), fill_value=255, dtype=np.uint8)
        )
        assert white_image.mode == "RGB"
        expected_perturbed_images = list(map(
            PIL.Image.fromarray,
            np.repeat(np.uint8(EXPECTED_MASKS_4x6 * 255), 3).reshape((6, 4, 6, 3))
        ))
        total_perts = []
        for i, (img, mask) in enumerate(impl.perturb(white_image)):
            assert img.mode == "RGB"
            assert img == expected_perturbed_images[i]
            assert np.allclose(mask, EXPECTED_MASKS_4x6[i])
            total_perts.append(img)
        assert len(total_perts) == 6

    def test_perturb_3channel_nonsquare(self) -> None:
        """
        Test basic perturbation on a known image with non-square window +
        stride.
        """
        impl = SlidingWindowPerturb(window_size=(3, 2), stride=(3, 2))
        # Image is slightly wide, should be occluded 6-ways.
        white_image = PIL.Image.fromarray(
            np.full((6, 6, 3), fill_value=255, dtype=np.uint8)
        )
        assert white_image.mode == "RGB"
        expected_perturbed_images = list(map(
            PIL.Image.fromarray,
            np.repeat(np.uint8(EXPECTED_MASKS_6x6_rect * 255), 3).reshape((6, 6, 6, 3))
        ))
        total_perts = []
        for i, (img, mask) in enumerate(impl.perturb(white_image)):
            assert img.mode == "RGB"
            assert img == expected_perturbed_images[i]
            assert np.allclose(mask, EXPECTED_MASKS_6x6_rect[i])
            total_perts.append(img)
        assert len(total_perts) == 6

    def test_perturb_4channel(self) -> None:
        """
        Test basic perturbation on a known image with even windowing + stride.
        """
        impl = SlidingWindowPerturb(window_size=(2, 2), stride=(2, 2))
        # Image is slightly wide, should be occluded 6-ways.
        white_image = PIL.Image.fromarray(
            np.full((4, 6, 4), fill_value=255, dtype=np.uint8)
        )
        assert white_image.mode == "RGBA"
        expected_perturbed_images = list(map(
            PIL.Image.fromarray,
            np.repeat(np.uint8(EXPECTED_MASKS_4x6 * 255), 4).reshape((6, 4, 6, 4))
        ))
        total_perts = []
        for i, (img, mask) in enumerate(impl.perturb(white_image)):
            assert img.mode == "RGBA"
            assert img == expected_perturbed_images[i]
            assert np.allclose(mask, EXPECTED_MASKS_4x6[i])
            total_perts.append(img)
        assert len(total_perts) == 6


# Common expected masks for 4x6 tests
EXPECTED_MASKS_4x6 = np.array([
    [[0, 0, 1, 1, 1, 1],
     [0, 0, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1]],
    [[1, 1, 0, 0, 1, 1],
     [1, 1, 0, 0, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1]],
    [[1, 1, 1, 1, 0, 0],
     [1, 1, 1, 1, 0, 0],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1]],
    [[1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [0, 0, 1, 1, 1, 1],
     [0, 0, 1, 1, 1, 1]],
    [[1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 0, 0, 1, 1],
     [1, 1, 0, 0, 1, 1]],
    [[1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 0, 0],
     [1, 1, 1, 1, 0, 0]],
], dtype=bool)
EXPECTED_MASKS_6x6_rect = np.array([
    [[0, 0, 1, 1, 1, 1],
     [0, 0, 1, 1, 1, 1],
     [0, 0, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1]],
    [[1, 1, 0, 0, 1, 1],
     [1, 1, 0, 0, 1, 1],
     [1, 1, 0, 0, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1]],
    [[1, 1, 1, 1, 0, 0],
     [1, 1, 1, 1, 0, 0],
     [1, 1, 1, 1, 0, 0],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1]],
    [[1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [0, 0, 1, 1, 1, 1],
     [0, 0, 1, 1, 1, 1],
     [0, 0, 1, 1, 1, 1]],
    [[1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 0, 0, 1, 1],
     [1, 1, 0, 0, 1, 1],
     [1, 1, 0, 0, 1, 1]],
    [[1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 0, 0],
     [1, 1, 1, 1, 0, 0],
     [1, 1, 1, 1, 0, 0]],
], dtype=bool)
