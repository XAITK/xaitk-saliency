
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
        ex_n = 1000
        ex_s = 8
        ex_p1 = 0.5
        impl = RISEPertubation(n=ex_n, s=ex_s, p1=ex_p1)
        assert impl.n == ex_n
        assert impl.s == ex_s
        assert impl.p1 == ex_p1

    def test_standard_config(self) -> None:
        ex_n = 1000
        ex_s = 8
        ex_p1 = 0.5
        impl = RISEPertubation(n=ex_n, s=ex_s, p1=ex_p1)
        for inst in configuration_test_helper(impl):
            assert inst.n == ex_n
            assert inst.s == ex_s
            assert inst.p1 == ex_p1

    def test_if_random(self) -> None:
        """
        Test that the perturbations are randomized
        """
        impl1 = RISEPertubation(n=1000, s=8, p1=0.5)
        impl2 = RISEPertubation(n=1000, s=8, p1=0.5)
        assert not np.array_equal(impl1.grid, impl2.grid)

    def test_seed(self) -> None:
        """
        Test that passing a seed generates equivalent masks
        """
        impl1 = RISEPertubation(n=1000, s=8, p1=0.5, seed=42)
        impl2 = RISEPertubation(n=1000, s=8, p1=0.5, seed=42)
        assert np.array_equal(impl1.grid, impl2.grid)

    def test_perturb_1channel(self) -> None:
        """
        Test basic perturbation on a known image with even windowing + stride.
        """
        # Image is slightly wide
        white_image = PIL.Image.fromarray(
            np.full((4, 6), fill_value=255, dtype=np.uint8)
        )

        # Setting threads=0 for serialized processing for deterministic
        # results.
        impl = RISEPertubation(n=2, s=2, p1=0.5, seed=42, threads=0)

        expected_perturbed_images = list(map(
            PIL.Image.fromarray,
            (EXPECTED_MASKS_4x6 * 255).astype(np.uint8)
        ))

        total_perts = []
        for i, (img, mask) in enumerate(impl.perturb(white_image)):
            assert img.mode == "L"
            # Test for expected output perturbed image content
            assert np.allclose(img, expected_perturbed_images[i])
            assert np.allclose(mask, EXPECTED_MASKS_4x6[i])
            total_perts.append(img)
        assert len(total_perts) == 2

    def test_call_idempotency(self) -> None:
        """
        Test that, at least when seeded and single-threaded, perturbation
        generation is idempotent.
        """
        # Image is slightly wide
        white_image = PIL.Image.fromarray(
            np.full((4, 6), fill_value=255, dtype=np.uint8)
        )
        # Setting threads=0 for serialized processing for deterministic
        # results. When greater than 1 idempotency cannot be guaranteed due to
        # thread interleaving.
        # Also of course seeding otherwise random will do its random things.
        impl = RISEPertubation(n=2, s=2, p1=0.5, seed=42, threads=0)

        imgs1, masks1 = zip(*impl.perturb(white_image))
        imgs2, masks2 = zip(*impl.perturb(white_image))

        for i, (img1, img2) in enumerate(zip(imgs1, imgs2)):
            assert img1 == img2
        assert np.allclose(np.asarray(masks1), np.asarray(masks2))

    def test_perturb_3channel(self) -> None:
        """
        Test basic perturbation on a known image with even windowing + stride.
        """
        # Image is slightly wide
        white_image = PIL.Image.fromarray(
            np.full((4, 6, 3), fill_value=255, dtype=np.uint8)
        )

        # Setting threads=0 for serialized processing for deterministic
        # results.
        impl = RISEPertubation(n=2, s=2, p1=0.5, seed=42, threads=0)

        expected_perturbed_images = list(map(
            PIL.Image.fromarray,
            np.repeat(np.uint8(EXPECTED_MASKS_4x6 * 255), 3).reshape((2, 4, 6, 3))
        ))

        total_perts = []
        for i, (img, mask) in enumerate(impl.perturb(white_image)):
            assert img.mode == "RGB"
            # Test for expected output perturbed image content
            assert np.allclose(img, expected_perturbed_images[i])
            assert np.allclose(mask, EXPECTED_MASKS_4x6[i])
            total_perts.append(img)
        assert len(total_perts) == 2

    def test_multiple_image_sizes(self) -> None:
        """
        Test that once we initialize a RISEPerturbation we can call it on
        images of varying sizes
        """
        impl = RISEPertubation(n=2, s=2, p1=0.5, seed=42)
        white_image_small = PIL.Image.fromarray(
            np.full((4, 6), fill_value=255, dtype=np.uint8)
        )
        white_image_large = PIL.Image.fromarray(
            np.full((41, 26), fill_value=255, dtype=np.uint8)
        )
        pairs_small = list(impl.perturb(white_image_small))
        assert len(pairs_small) == 2
        for i, m in pairs_small:
            assert i.size == white_image_small.size
            assert m.shape[::-1] == white_image_small.size

        pairs_large = list(impl.perturb(white_image_large))
        assert len(pairs_large) == 2
        for i, m in pairs_large:
            assert i.size == white_image_large.size
            assert m.shape[::-1] == white_image_large.size


# Common expected masks for 4x6 tests
# These assume seed=42, serial generation with threads=0
EXPECTED_MASKS_4x6 = np.array([
    [
        [0.03703703, 0.18518518, 0.33333330, 0.48148150, 0.62962960, 0.55555546],
        [0.05555555, 0.27777780, 0.50000000, 0.72222227, 0.94444450, 0.83333325],
        [0.03703703, 0.18518520, 0.33333330, 0.48148150, 0.62962970, 0.55555550],
        [0.01851852, 0.09259260, 0.16666669, 0.24074079, 0.31481487, 0.27777780]
    ],
    [
        [0.83333330, 0.94444440, 0.72222220, 0.50000000, 0.27777773, 0.05555552],
        [0.55555550, 0.62962960, 0.48148146, 0.33333330, 0.18518515, 0.03703701],
        [0.27777780, 0.31481487, 0.24074076, 0.16666669, 0.09259259, 0.01851851],
        [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000]
    ]
], dtype=np.float32)
