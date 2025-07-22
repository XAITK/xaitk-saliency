import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper
from syrupy.assertion import SnapshotAssertion

from tests.test_utils import CustomFloatSnapshotExtension
from xaitk_saliency.impls.perturb_image.random_grid import RandomGrid


@pytest.fixture
def snapshot_custom(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(CustomFloatSnapshotExtension)


class TestRandomGrid:
    def test_init_valued(self) -> None:
        """Test that constructor values pass."""

        test_n = 837
        test_s = (34, 12)
        test_p1 = 0.82
        test_seed = 938
        test_threads = 106

        impl = RandomGrid(n=test_n, s=test_s, p1=test_p1, seed=test_seed, threads=test_threads)

        assert impl.n == test_n
        assert impl.s == test_s
        assert np.allclose(impl.p1, test_p1)
        assert impl.seed == test_seed
        assert impl.threads == test_threads

    def test_standard_config(self) -> None:
        """Test values in implementation config."""

        test_n = 123
        test_s = (55, 44)
        test_p1 = 0.2
        test_seed = 777
        test_threads = 999

        impl = RandomGrid(n=test_n, s=test_s, p1=test_p1, seed=test_seed, threads=test_threads)

        for inst in configuration_test_helper(impl):
            assert inst.n == test_n
            assert inst.s == test_s
            assert np.allclose(impl.p1, test_p1)
            assert inst.seed == test_seed
            assert inst.threads == test_threads

    def test_if_random(self) -> None:
        """Test that the perturbations are randomized."""

        impl1 = RandomGrid(n=3, s=(5, 4), p1=0.5)
        impl2 = RandomGrid(n=3, s=(5, 4), p1=0.5)

        rng = np.random.default_rng(seed=0)
        img = rng.integers(0, 255, size=(20, 20), dtype=np.uint8)

        masks1 = impl1(img)
        masks2 = impl2(img)

        assert not np.array_equal(masks1, masks2)

    def test_seed(self) -> None:
        """Test that using a seed generates the same masks."""

        impl1 = RandomGrid(n=3, s=(2, 1), p1=0.6, seed=5)
        impl2 = RandomGrid(n=3, s=(2, 1), p1=0.6, seed=5)

        rng = np.random.default_rng(seed=0)
        img = rng.integers(0, 255, size=(10, 10), dtype=np.uint8)

        masks1 = impl1(img)
        masks2 = impl2(img)

        assert np.array_equal(masks1, masks2)

    def test_call_idempotency(self) -> None:
        """
        Test that perturbation generation is idempotent, at least when seeded
        and single-threaded.
        """
        rng = np.random.default_rng(seed=0)
        img = rng.integers(0, 255, size=(10, 10), dtype=np.uint8)

        impl = RandomGrid(n=2, s=(1, 2), p1=0.4, seed=1, threads=0)

        masks1 = impl(img)
        masks2 = impl(img)

        assert np.array_equal(masks1, masks2)

    def test_perturb_1_channel(self, snapshot_custom: SnapshotAssertion) -> None:
        """
        Test mask generation on a one-channel image of a known size. Number
        of channels should not affect output masks.
        """
        rng = np.random.default_rng(seed=0)
        img = rng.integers(0, 255, size=(4, 6), dtype=np.uint8)

        impl = RandomGrid(n=2, s=(2, 2), p1=0.5, seed=123, threads=0)
        actual_masks = impl(img)

        snapshot_custom.assert_match(actual_masks)

    def test_perturb_3_channel(self, snapshot_custom: SnapshotAssertion) -> None:
        """
        Test mask generation on a three-channel image of a known size. Number
        of channels should not affect output masks.
        """
        rng = np.random.default_rng(seed=0)
        img = rng.integers(0, 255, size=(4, 6, 3), dtype=np.uint8)

        impl = RandomGrid(n=2, s=(2, 2), p1=0.5, seed=123, threads=0)
        actual_masks = impl.perturb(img)

        snapshot_custom.assert_match(actual_masks)

    def test_multiple_image_size(self) -> None:
        """
        Test that a single implementation can be used for images of varying
        sizes.
        """
        impl = RandomGrid(n=5, s=(3, 4), p1=0.2, seed=42, threads=0)

        rng = np.random.default_rng(seed=0)
        img_small = rng.integers(0, 255, size=(5, 7), dtype=np.uint8)
        img_large = rng.integers(0, 255, size=(55, 77), dtype=np.uint8)

        masks_small = impl(img_small)
        masks_large = impl(img_large)

        assert len(masks_small) == 5
        assert masks_small.shape[1:] == img_small.shape

        assert len(masks_large) == 5
        assert masks_large.shape[1:] == img_large.shape

    def test_threading(self, snapshot_custom: SnapshotAssertion) -> None:
        """Test that using threading does not affect results."""
        rng = np.random.default_rng(seed=0)
        img = rng.integers(0, 255, size=(4, 6), dtype=np.uint8)

        impl = RandomGrid(n=2, s=(2, 2), p1=0.5, seed=123, threads=1)

        actual_masks = impl.perturb(img)

        snapshot_custom.assert_match(actual_masks)
