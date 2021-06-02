
from unittest import TestCase

import PIL.Image
import numpy as np
from smqtk_core.configuration import configuration_test_helper

from xaitk_saliency.impls.perturb_image.rise import RISEPertubation


class TestRISEPerturbation (TestCase):
    def test_init_valued(self) -> None:
        """
        Test that constructor values pass.
        """
        ex_N = 1000
        ex_s = 8
        ex_p1 = 0.5
        impl = RISEPertubation(N=ex_N, s=ex_s, p1=ex_p1)
        assert impl.N == ex_N
        assert impl.s == ex_s
        assert impl.p1 == ex_p1

    def test_standard_config(self) -> None:
        ex_N = 1000
        ex_s = 8
        ex_p1 = 0.5
        impl = RISEPertubation(N=ex_N, s=ex_s, p1=ex_p1)
        for inst in configuration_test_helper(impl):
            assert inst.N == ex_N
            assert inst.s == ex_s
            assert inst.p1 == ex_p1

    def test_if_random(self) -> None:
        """
        Test that the perturbations are randomized
        """
        impl1 = RISEPertubation(N=1000, s=8, p1=0.5)
        impl2 = RISEPertubation(N=1000, s=8, p1=0.5)
        assert not np.array_equal(impl1.grid, impl2.grid)

    def test_seed(self) -> None:
        """
        Test that passing a seed generates equivalent masks
        """
        impl1 = RISEPertubation(N=1000, s=8, p1=0.5, seed=42)
        impl2 = RISEPertubation(N=1000, s=8, p1=0.5, seed=42)
        assert np.array_equal(impl1.grid, impl2.grid)

    def test_perturb_1channel(self) -> None:
        """
        Test basic perturbation on a known image with even windowing + stride.
        """
        impl = RISEPertubation(N=2, s=2, p1=0.5, seed=42)
        # Image is slightly wide, should be occluded 6-ways.
        white_image = PIL.Image.fromarray(
            np.full((4, 6), fill_value=255, dtype=np.uint8)
        )
        pert_imgs, pert_masks = impl.perturb(white_image)
        assert len(pert_imgs) == 2
        assert len(pert_masks) == 2
        assert np.allclose(pert_masks, EXPECTED_MASKS_4x6)
        # Output image modes should match input
        for i, img in enumerate(pert_imgs):
            assert img.mode == "L"
            # Test for expected output perturbed image content
            assert np.allclose(img, EXPECTED_IMAGES_4x6_L[i])

    def test_perturb_3channel(self) -> None:
        """
        Test basic perturbation on a known image with even windowing + stride.
        """
        impl = RISEPertubation(N=2, s=2, p1=0.5, seed=42)
        # Image is slightly wide, should be occluded 6-ways.
        white_image = PIL.Image.fromarray(
            np.full((4, 6, 3), fill_value=255, dtype=np.uint8)
        )
        pert_imgs, pert_masks = impl.perturb(white_image)
        assert len(pert_imgs) == 2
        assert len(pert_masks) == 2
        assert np.allclose(pert_masks, EXPECTED_MASKS_4x6)
        # Output image modes should match input
        for i, img in enumerate(pert_imgs):
            assert img.mode == "RGB"
            # Test for expected output perturbed image content
            expected_img_ch = EXPECTED_IMAGES_4x6_L[i]
            expected_img = np.stack(
                (expected_img_ch, expected_img_ch, expected_img_ch),
                -1
            )
            assert np.allclose(img, expected_img)

    def test_perturb_4channel(self) -> None:
        pass

    def test_multiple_image_sizes(self) -> None:
        """
        Test that once we initialize a RISEPerturbation we can call it on
        images of varying sizes
        """
        impl = RISEPertubation(N=2, s=2, p1=0.5, seed=42)
        white_image_small = PIL.Image.fromarray(
            np.full((4, 6), fill_value=255, dtype=np.uint8)
        )
        white_image_large = PIL.Image.fromarray(
            np.full((41, 26), fill_value=255, dtype=np.uint8)
        )
        _, _ = impl.perturb(white_image_small)
        _, _ = impl.perturb(white_image_large)


# Common expected masks for 4x6 tests
EXPECTED_MASKS_4x6 = np.array([
    [
        [0.38888890, 0.16666669, 0.05555555, 0.27777779, 0.50000000, 0.72222227],
        [0.25925925, 0.11111112, 0.03703703, 0.18518519, 0.33333331, 0.48148149],
        [0.12962964, 0.05555557, 0.01851852, 0.09259260, 0.16666669, 0.24074079],
        [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000]
    ],
    [
        [0.83333331, 0.94444442, 0.72222221, 0.50000000, 0.27777773, 0.05555552],
        [0.55555552, 0.62962961, 0.48148146, 0.33333331, 0.18518515, 0.03703701],
        [0.27777779, 0.31481487, 0.24074076, 0.16666669, 0.09259259, 0.01851851],
        [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000]
    ]
])

EXPECTED_IMAGES_4x6_L = [
    np.array([
        [99,  42,  14,  70, 127, 184],
        [66,  28,   9,  47,  84, 122],
        [33,  14,   4,  23,  42,  61],
        [0,    0,   0,   0,   0,   0]],
        dtype='uint8'),

    np.array([
        [212, 240, 184, 127,  70,  14],
        [141, 160, 122,  84,  47,   9],
        [70,   80,  61,  42,  23,   4],
        [0,     0,   0,   0,   0,   0]],
        dtype='uint8')
]
