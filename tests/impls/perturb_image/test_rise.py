import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper
from syrupy.assertion import SnapshotAssertion

from tests.test_utils import CustomFloatSnapshotExtension
from xaitk_saliency.impls.perturb_image.rise import RISEGrid


@pytest.fixture
def snapshot_custom(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(CustomFloatSnapshotExtension)


class TestRISEPerturbation:
    def test_init_valued(self) -> None:
        """Test that constructor values pass."""
        ex_n = 1000
        ex_s = 8
        ex_p1 = 0.5
        impl = RISEGrid(n=ex_n, s=ex_s, p1=ex_p1)
        assert impl.n == ex_n
        assert impl.s == ex_s
        assert np.allclose(impl.p1, ex_p1)

    def test_init_outofrange_p1(self) -> None:
        """Test catching an out of range p1 value."""
        with pytest.raises(ValueError, match=r"Input p1 value of -0\.3 is not within the expected \[0,1\] range\."):
            RISEGrid(10, 8, p1=-0.3)

        with pytest.raises(ValueError, match=r"Input p1 value of 5 is not within the expected \[0,1\] range\."):
            RISEGrid(10, 8, p1=5)

    def test_standard_config(self) -> None:
        ex_n = 1000
        ex_s = 8
        ex_p1 = 0.5
        impl = RISEGrid(n=ex_n, s=ex_s, p1=ex_p1)
        for inst in configuration_test_helper(impl):
            assert inst.n == ex_n
            assert inst.s == ex_s
            assert np.allclose(inst.p1, ex_p1)

    def test_if_random(self) -> None:
        """Test that the perturbations are randomized"""
        impl1 = RISEGrid(n=1000, s=8, p1=0.5)
        impl2 = RISEGrid(n=1000, s=8, p1=0.5)
        assert not np.array_equal(impl1.grid, impl2.grid)

    def test_seed(self) -> None:
        """Test that passing a seed generates equivalent masks"""
        impl1 = RISEGrid(n=1000, s=8, p1=0.5, seed=42)
        impl2 = RISEGrid(n=1000, s=8, p1=0.5, seed=42)
        assert np.array_equal(impl1.grid, impl2.grid)

    def test_perturb_1channel(self, snapshot_custom: SnapshotAssertion) -> None:
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

        snapshot_custom.assert_match(actual_masks)

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

    def test_perturb_3channel(self, snapshot_custom: SnapshotAssertion) -> None:
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

        snapshot_custom.assert_match(actual_masks)

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
