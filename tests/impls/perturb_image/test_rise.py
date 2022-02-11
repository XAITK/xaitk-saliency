import pytest
import numpy as np
from smqtk_core.configuration import configuration_test_helper

from xaitk_saliency.impls.perturb_image.rise import RISEGrid


class TestRISEPerturbation:
    def test_init_valued(self) -> None:
        """
        Test that constructor values pass.
        """
        ex_n = 1000
        ex_s = 8
        ex_p1 = 0.5
        impl = RISEGrid(n=ex_n, s=ex_s, p1=ex_p1)
        assert impl.n == ex_n
        assert impl.s == ex_s
        assert impl.p1 == ex_p1

    def test_init_outofrange_p1(self) -> None:
        """
        Test catching an out of range p1 value.
        """
        with pytest.raises(
            ValueError,
            match=r"Input p1 value of -0\.3 is not within the expected \[0,1\] range\."
        ):
            RISEGrid(10, 8, p1=-0.3)

        with pytest.raises(
            ValueError,
            match=r"Input p1 value of 5 is not within the expected \[0,1\] range\."
        ):
            RISEGrid(10, 8, p1=5)

    def test_standard_config(self) -> None:
        ex_n = 1000
        ex_s = 8
        ex_p1 = 0.5
        impl = RISEGrid(n=ex_n, s=ex_s, p1=ex_p1)
        for inst in configuration_test_helper(impl):
            assert inst.n == ex_n
            assert inst.s == ex_s
            assert inst.p1 == ex_p1

    def test_if_random(self) -> None:
        """
        Test that the perturbations are randomized
        """
        impl1 = RISEGrid(n=1000, s=8, p1=0.5)
        impl2 = RISEGrid(n=1000, s=8, p1=0.5)
        assert not np.array_equal(impl1.grid, impl2.grid)

    def test_seed(self) -> None:
        """
        Test that passing a seed generates equivalent masks
        """
        impl1 = RISEGrid(n=1000, s=8, p1=0.5, seed=42)
        impl2 = RISEGrid(n=1000, s=8, p1=0.5, seed=42)
        assert np.array_equal(impl1.grid, impl2.grid)

    def test_perturb_1channel(self) -> None:
        """
        Test basic perturbation on a known image with even windowing + stride.
        Input image mode should not impact the masks output.
        """
        # Image is slightly wide
        white_image = np.full((4, 6), fill_value=255, dtype=np.uint8)

        # Setting threads=0 for serialized processing for deterministic
        # results.
        impl = RISEGrid(n=2, s=2, p1=0.5, seed=42, threads=0)
        actual_masks = impl.perturb(white_image)

        assert np.allclose(
            actual_masks,
            EXPECTED_MASKS_4x6
        )

    def test_call_idempotency(self) -> None:
        """
        Test that, at least when seeded and single-threaded, perturbation
        generation is idempotent.
        """
        # Image is slightly wide
        white_image = np.full((4, 6), fill_value=255, dtype=np.uint8)
        # Setting threads=0 for serialized processing for deterministic
        # results. When greater than 1 idempotency cannot be guaranteed due to
        # thread interleaving.
        # Also of course seeding otherwise random will do its random things.
        impl = RISEGrid(n=2, s=2, p1=0.5, seed=42, threads=0)

        actual_masks1 = impl.perturb(white_image)
        actual_masks2 = impl.perturb(white_image)

        assert np.allclose(
            actual_masks1,
            actual_masks2,
        )

    def test_perturb_3channel(self) -> None:
        """
        Test basic perturbation on a known image with even windowing + stride.
        Input image mode should not impact the masks output.
        """
        # Image is slightly wide
        white_image = np.full((4, 6, 3), fill_value=255, dtype=np.uint8)

        # Setting threads=0 for serialized processing for deterministic
        # results.
        impl = RISEGrid(n=2, s=2, p1=0.5, seed=42, threads=0)
        actual_masks = impl.perturb(white_image)

        assert np.allclose(
            actual_masks,
            EXPECTED_MASKS_4x6
        )

    def test_multiple_image_sizes(self) -> None:
        """
        Test that once we initialize a RISEPerturbation we can call it on
        images of varying sizes
        """
        impl = RISEGrid(n=2, s=2, p1=0.5, seed=42)
        white_image_small = np.full((4, 6), fill_value=255, dtype=np.uint8)
        white_image_large = np.full((41, 26), fill_value=255, dtype=np.uint8)
        masks_small = impl.perturb(white_image_small)
        assert len(masks_small) == 2
        assert masks_small.shape[1:] == white_image_small.shape

        masks_large = impl.perturb(white_image_large)
        assert len(masks_large) == 2
        assert masks_large.shape[1:] == white_image_large.shape

    def test_multi_axis_grid(self) -> None:
        """
        Test using tuple s-value to split horizontal and vertical axis
        unevenly.
        """
        img = np.full((4, 6), fill_value=255, dtype=np.uint8)
        impl1 = RISEGrid(n=2, s=(2, 3), p1=0.5, seed=42, threads=0)
        impl2 = RISEGrid(n=2, s=(2, None), p1=0.5, seed=42, threads=0)
        impl3 = RISEGrid(n=2, s=(None, 3), p1=0.5, seed=42, threads=0)

        masks1 = impl1(img)
        masks2 = impl2(img)
        masks3 = impl3(img)

        assert np.allclose(
            masks1,
            masks2,
            masks3,
            EXPECTED_MASKS_4X6_SQUARE,
        )


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

EXPECTED_MASKS_4X6_SQUARE = np.array([
    [
        [0.06250000, 0.43750000, 0.81250000, 0.81250000, 0.43750000, 0.06250000],
        [0.06250000, 0.43750000, 0.81250000, 0.81250000, 0.43750000, 0.06250000],
        [0.06250000, 0.43750000, 0.81250000, 0.81250000, 0.43750000, 0.06250000],
        [0.06250000, 0.43750000, 0.81250000, 0.81250000, 0.43750000, 0.06250000],
    ],

    [
        [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.18750000, 0.56250000],
        [0.33333334, 0.33333334, 0.33333334, 0.33333334, 0.39583334, 0.52083330],
        [0.66666660, 0.66666660, 0.66666660, 0.66666660, 0.60416660, 0.47916670],
        [1.00000000, 1.00000000, 1.00000000, 1.00000000, 0.81250000, 0.43750000],
    ]
], dtype=np.float32)
