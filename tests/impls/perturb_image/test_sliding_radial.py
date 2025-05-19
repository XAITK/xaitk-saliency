import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper
from syrupy.assertion import SnapshotAssertion

from tests.test_utils import CustomFloatSnapshotExtension
from xaitk_saliency import PerturbImage
from xaitk_saliency.impls.perturb_image.sliding_radial import SlidingRadial


@pytest.fixture
def snapshot_custom(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(CustomFloatSnapshotExtension)


class TestRadialOcclusionBasedPerturb:
    def test_init_default(self) -> None:
        """Test empty construction since we provide defaults."""
        impl = SlidingRadial()
        assert impl.radius == (50, 50)
        assert impl.stride == (20, 20)
        assert impl.sigma is None

    def test_init_valued(self) -> None:
        """Test that constructor values pass."""
        ex_r = (123.5, 456)
        ex_stride = (444, 445)
        ex_sig = (63, 73.2)
        impl = SlidingRadial(radius=ex_r, stride=ex_stride, sigma=ex_sig)
        assert impl.radius == ex_r
        assert impl.stride == ex_stride
        assert impl.sigma == ex_sig

    def test_plugin_find(self) -> None:
        """
        This implementation has no optional plugins so it should be found and
        exposed by the super-type's impl getter.
        """
        assert SlidingRadial in PerturbImage.get_impls()

    def test_standard_config(self) -> None:
        ex_r = (123.5, 456)
        ex_stride = (444, 445)
        ex_sig = (63, 73.2)
        impl = SlidingRadial(radius=ex_r, stride=ex_stride, sigma=ex_sig)
        for inst in configuration_test_helper(impl):
            assert inst.radius == ex_r
            assert inst.stride == ex_stride
            assert inst.sigma == ex_sig

    def test_perturb_1channel(self, snapshot_custom: SnapshotAssertion) -> None:
        """
        Test basic perturbation on a known 1-channel image.
        Input image mode should not impact the masks output.
        """
        # Image is slightly wide, should be occluded 6-ways.
        # Simulate grayscale image (single-channel)
        white_image = np.full((4, 6), fill_value=255, dtype=np.uint8)

        impl = SlidingRadial(radius=(2, 4), stride=(2, 2))
        actual_masks = impl.perturb(white_image)

        snapshot_custom.assert_match(actual_masks)

    def test_perturb_3channel(self, snapshot_custom: SnapshotAssertion) -> None:
        """
        Test basic perturbation on a known 3-channel image.
        Input image mode should not impact the masks output.
        """
        # Image is slightly wide, should be occluded 6-ways.
        # Simulate RGB or BGR image (3-channel)
        white_image = np.full((4, 6, 3), fill_value=255, dtype=np.uint8)

        impl = SlidingRadial(radius=(2, 4), stride=(2, 2))
        actual_masks = impl.perturb(white_image)

        snapshot_custom.assert_match(actual_masks)

    def test_perturb_3channel_with_blurring(self, snapshot_custom: SnapshotAssertion) -> None:
        """
        Test basic perturbation on a known 3-channel image with blurring.
        Input image mode should not impact the masks output.
        """
        # Square image for uneven masking 6-ways.
        # Simulate RGB or BGR image (3-channel)
        white_image = np.full((4, 6, 3), fill_value=255, dtype=np.uint8)

        impl = SlidingRadial(radius=(2, 4), stride=(2, 2), sigma=(2, 2))
        actual_masks = impl.perturb(white_image)

        snapshot_custom.assert_match(actual_masks)

    def test_perturb_4channel(self, snapshot_custom: SnapshotAssertion) -> None:
        """
        Test basic perturbation on a known 4-channel image.
        Input image mode should not impact the masks output.
        """
        # Image is slightly wide, should be occluded 6-ways.
        # Simulate RGBA format image (4-channel)
        white_image = np.full((4, 6, 4), fill_value=255, dtype=np.uint8)

        impl = SlidingRadial(radius=(2, 4), stride=(2, 2))
        actual_masks = impl.perturb(white_image)

        snapshot_custom.assert_match(actual_masks)
