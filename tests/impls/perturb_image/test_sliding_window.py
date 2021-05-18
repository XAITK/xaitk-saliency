from unittest import TestCase

import PIL.Image
import numpy as np
from smqtk_core.configuration import configuration_test_helper

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

    def test_standard_config(self) -> None:
        ex_w = (777, 776)
        ex_s = (444, 445)
        impl = SlidingWindowPerturb(window_size=ex_w, stride=ex_s)
        for inst in configuration_test_helper(impl):
            assert inst.window_size == ex_w
            assert inst.stride == ex_s

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
        pert_imgs, pert_masks = impl.perturb(white_image)
        assert len(pert_imgs) == 6
        assert len(pert_masks) == 6
        assert np.allclose(pert_masks, EXPECTED_MASKS_4x6)
        # Output image modes should match input
        for i, img in enumerate(pert_imgs):
            assert img.mode == "L"
        # Test for expected output perturbed image content
        assert np.allclose(
            np.asarray(pert_imgs[0]),
            np.vstack([
                np.hstack([BLACK_2x2_L, WHITE_2x2_L, WHITE_2x2_L]),
                np.hstack([WHITE_2x2_L, WHITE_2x2_L, WHITE_2x2_L]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[1]),
            np.vstack([
                np.hstack([WHITE_2x2_L, BLACK_2x2_L, WHITE_2x2_L]),
                np.hstack([WHITE_2x2_L, WHITE_2x2_L, WHITE_2x2_L]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[2]),
            np.vstack([
                np.hstack([WHITE_2x2_L, WHITE_2x2_L, BLACK_2x2_L]),
                np.hstack([WHITE_2x2_L, WHITE_2x2_L, WHITE_2x2_L]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[3]),
            np.vstack([
                np.hstack([WHITE_2x2_L, WHITE_2x2_L, WHITE_2x2_L]),
                np.hstack([BLACK_2x2_L, WHITE_2x2_L, WHITE_2x2_L]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[4]),
            np.vstack([
                np.hstack([WHITE_2x2_L, WHITE_2x2_L, WHITE_2x2_L]),
                np.hstack([WHITE_2x2_L, BLACK_2x2_L, WHITE_2x2_L]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[5]),
            np.vstack([
                np.hstack([WHITE_2x2_L, WHITE_2x2_L, WHITE_2x2_L]),
                np.hstack([WHITE_2x2_L, WHITE_2x2_L, BLACK_2x2_L]),
            ])
        )

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
        pert_imgs, pert_masks = impl.perturb(white_image)
        assert len(pert_imgs) == 6
        assert len(pert_masks) == 6
        assert np.allclose(pert_masks, EXPECTED_MASKS_4x6)
        # Output image modes should match input
        for i, img in enumerate(pert_imgs):
            assert img.mode == "RGB"
        # Test for expected output perturbed image content
        assert np.allclose(
            np.asarray(pert_imgs[0]),
            np.vstack([
                np.hstack([BLACK_2x2_RGB, WHITE_2x2_RGB, WHITE_2x2_RGB]),
                np.hstack([WHITE_2x2_RGB, WHITE_2x2_RGB, WHITE_2x2_RGB]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[1]),
            np.vstack([
                np.hstack([WHITE_2x2_RGB, BLACK_2x2_RGB, WHITE_2x2_RGB]),
                np.hstack([WHITE_2x2_RGB, WHITE_2x2_RGB, WHITE_2x2_RGB]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[2]),
            np.vstack([
                np.hstack([WHITE_2x2_RGB, WHITE_2x2_RGB, BLACK_2x2_RGB]),
                np.hstack([WHITE_2x2_RGB, WHITE_2x2_RGB, WHITE_2x2_RGB]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[3]),
            np.vstack([
                np.hstack([WHITE_2x2_RGB, WHITE_2x2_RGB, WHITE_2x2_RGB]),
                np.hstack([BLACK_2x2_RGB, WHITE_2x2_RGB, WHITE_2x2_RGB]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[4]),
            np.vstack([
                np.hstack([WHITE_2x2_RGB, WHITE_2x2_RGB, WHITE_2x2_RGB]),
                np.hstack([WHITE_2x2_RGB, BLACK_2x2_RGB, WHITE_2x2_RGB]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[5]),
            np.vstack([
                np.hstack([WHITE_2x2_RGB, WHITE_2x2_RGB, WHITE_2x2_RGB]),
                np.hstack([WHITE_2x2_RGB, WHITE_2x2_RGB, BLACK_2x2_RGB]),
            ])
        )

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
        pert_imgs, pert_masks = impl.perturb(white_image)
        assert len(pert_imgs) == 6
        assert len(pert_masks) == 6
        assert np.allclose(pert_masks, EXPECTED_MASKS_6x6_rect)
        # Output image modes should match input
        for i, img in enumerate(pert_imgs):
            assert img.mode == "RGB"
        # Test for expected output perturbed image content
        assert np.allclose(
            np.asarray(pert_imgs[0]),
            np.vstack([
                np.hstack([BLACK_3x2_RGB, WHITE_3x2_RGB, WHITE_3x2_RGB]),
                np.hstack([WHITE_3x2_RGB, WHITE_3x2_RGB, WHITE_3x2_RGB]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[1]),
            np.vstack([
                np.hstack([WHITE_3x2_RGB, BLACK_3x2_RGB, WHITE_3x2_RGB]),
                np.hstack([WHITE_3x2_RGB, WHITE_3x2_RGB, WHITE_3x2_RGB]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[2]),
            np.vstack([
                np.hstack([WHITE_3x2_RGB, WHITE_3x2_RGB, BLACK_3x2_RGB]),
                np.hstack([WHITE_3x2_RGB, WHITE_3x2_RGB, WHITE_3x2_RGB]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[3]),
            np.vstack([
                np.hstack([WHITE_3x2_RGB, WHITE_3x2_RGB, WHITE_3x2_RGB]),
                np.hstack([BLACK_3x2_RGB, WHITE_3x2_RGB, WHITE_3x2_RGB]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[4]),
            np.vstack([
                np.hstack([WHITE_3x2_RGB, WHITE_3x2_RGB, WHITE_3x2_RGB]),
                np.hstack([WHITE_3x2_RGB, BLACK_3x2_RGB, WHITE_3x2_RGB]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[5]),
            np.vstack([
                np.hstack([WHITE_3x2_RGB, WHITE_3x2_RGB, WHITE_3x2_RGB]),
                np.hstack([WHITE_3x2_RGB, WHITE_3x2_RGB, BLACK_3x2_RGB]),
            ])
        )

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
        pert_imgs, pert_masks = impl.perturb(white_image)
        assert len(pert_imgs) == 6
        assert len(pert_masks) == 6
        assert np.allclose(pert_masks, EXPECTED_MASKS_4x6)
        # Output image modes should match input
        for i, img in enumerate(pert_imgs):
            assert img.mode == "RGBA"
        # Test for expected output perturbed image content
        assert np.allclose(
            np.asarray(pert_imgs[0]),
            np.vstack([
                np.hstack([BLACK_2x2_RGBA, WHITE_2x2_RGBA, WHITE_2x2_RGBA]),
                np.hstack([WHITE_2x2_RGBA, WHITE_2x2_RGBA, WHITE_2x2_RGBA]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[1]),
            np.vstack([
                np.hstack([WHITE_2x2_RGBA, BLACK_2x2_RGBA, WHITE_2x2_RGBA]),
                np.hstack([WHITE_2x2_RGBA, WHITE_2x2_RGBA, WHITE_2x2_RGBA]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[2]),
            np.vstack([
                np.hstack([WHITE_2x2_RGBA, WHITE_2x2_RGBA, BLACK_2x2_RGBA]),
                np.hstack([WHITE_2x2_RGBA, WHITE_2x2_RGBA, WHITE_2x2_RGBA]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[3]),
            np.vstack([
                np.hstack([WHITE_2x2_RGBA, WHITE_2x2_RGBA, WHITE_2x2_RGBA]),
                np.hstack([BLACK_2x2_RGBA, WHITE_2x2_RGBA, WHITE_2x2_RGBA]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[4]),
            np.vstack([
                np.hstack([WHITE_2x2_RGBA, WHITE_2x2_RGBA, WHITE_2x2_RGBA]),
                np.hstack([WHITE_2x2_RGBA, BLACK_2x2_RGBA, WHITE_2x2_RGBA]),
            ])
        )
        assert np.allclose(
            np.asarray(pert_imgs[5]),
            np.vstack([
                np.hstack([WHITE_2x2_RGBA, WHITE_2x2_RGBA, WHITE_2x2_RGBA]),
                np.hstack([WHITE_2x2_RGBA, WHITE_2x2_RGBA, BLACK_2x2_RGBA]),
            ])
        )


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

WHITE_2x2_L = np.array([
    [255, 255],
    [255, 255],
])
WHITE_2x2_RGB = np.array([
    [[255, 255, 255], [255, 255, 255]],
    [[255, 255, 255], [255, 255, 255]],
])
WHITE_3x2_RGB = np.array([
    [[255, 255, 255], [255, 255, 255]],
    [[255, 255, 255], [255, 255, 255]],
    [[255, 255, 255], [255, 255, 255]],
])
WHITE_2x2_RGBA = np.array([
    [[255, 255, 255, 255], [255, 255, 255, 255]],
    [[255, 255, 255, 255], [255, 255, 255, 255]],
])

BLACK_2x2_L = np.array([
    [0, 0],
    [0, 0]
])
BLACK_2x2_RGB = np.array([
    [[0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0]]
])
BLACK_3x2_RGB = np.array([
    [[0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0]]
])
BLACK_2x2_RGBA = np.array([
    [[0, 0, 0, 0], [0, 0, 0, 0]],
    [[0, 0, 0, 0], [0, 0, 0, 0]]
])
