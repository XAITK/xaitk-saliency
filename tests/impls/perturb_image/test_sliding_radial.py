import numpy as np
from smqtk_core.configuration import configuration_test_helper

from xaitk_saliency import PerturbImage
from xaitk_saliency.impls.perturb_image.sliding_radial import SlidingRadial


class TestRadialOcclusionBasedPerturb:

    def test_init_default(self) -> None:
        """
        Test empty construction since we provide defaults.
        """
        impl = SlidingRadial()
        assert impl.radius == 50
        assert impl.sigma == 0
        assert impl.stride == (20, 20)

    def test_init_valued(self) -> None:
        """
        Test that constructor values pass.
        """
        ex_r = 456
        ex_sig = 73
        ex_stride = (444, 445)
        impl = SlidingRadial(radius=ex_r, sigma=ex_sig, stride=ex_stride)
        assert impl.radius == ex_r
        assert impl.sigma == ex_sig
        assert impl.stride == ex_stride

    def test_plugin_find(self) -> None:
        """
        This implementation has no optional plugins so it should be found and
        exposed by the super-type's impl getter.
        """
        assert SlidingRadial in PerturbImage.get_impls()

    def test_standard_config(self) -> None:
        ex_r = 456
        ex_sig = 73
        ex_stride = (444, 445)
        impl = SlidingRadial(radius=ex_r, sigma=ex_sig, stride=ex_stride)
        for inst in configuration_test_helper(impl):
            assert inst.radius == ex_r
            assert inst.sigma == ex_sig
            assert inst.stride == ex_stride

    def test_perturb_1channel(self) -> None:
        """
        Test basic perturbation on a known 1-channel image.
        Input image mode should not impact the masks output.
        """
        # Image is slightly wide, should be occluded 6-ways.
        # Simulate grayscale image (single-channel)
        white_image = np.full((4, 6), fill_value=255, dtype=np.uint8)

        impl = SlidingRadial(radius=2, stride=(2, 2))
        actual_masks = impl.perturb(white_image)

        assert np.allclose(
            actual_masks,
            EXPECTED_MASKS_4x6
        )

    def test_perturb_3channel(self) -> None:
        """
        Test basic perturbation on a known 3-channel image.
        Input image mode should not impact the masks output.
        """
        # Image is slightly wide, should be occluded 6-ways.
        # Simulate RGB or BGR image (3-channel)
        white_image = np.full((4, 6, 3), fill_value=255, dtype=np.uint8)

        impl = SlidingRadial(radius=2, stride=(2, 2))
        actual_masks = impl.perturb(white_image)

        assert np.allclose(
            actual_masks,
            EXPECTED_MASKS_4x6
        )

    def test_perturb_3channel_with_blurring(self) -> None:
        """
        Test basic perturbation on a known 3-channel image with blurring.
        Input image mode should not impact the masks output.
        """
        # Square image for uneven masking 6-ways.
        # Simulate RGB or BGR image (3-channel)
        white_image = np.full((4, 6, 3), fill_value=255, dtype=np.uint8)

        impl = SlidingRadial(radius=2, sigma=2, stride=(2, 2))
        actual_masks = impl.perturb(white_image)

        assert np.allclose(
            actual_masks,
            EXPECTED_BLURRED_MASKS_4x6
        )

    def test_perturb_4channel(self) -> None:
        """
        Test basic perturbation on a known 4-channel image.
        Input image mode should not impact the masks output.
        """
        # Image is slightly wide, should be occluded 6-ways.
        # Simulate RGBA format image (4-channel)
        white_image = np.full((4, 6, 4), fill_value=255, dtype=np.uint8)

        impl = SlidingRadial(radius=2, stride=(2, 2))
        actual_masks = impl.perturb(white_image)

        assert np.allclose(
            actual_masks,
            EXPECTED_MASKS_4x6
        )


# Common expected masks for 4x6 tests
EXPECTED_MASKS_4x6 = np.array([
    [[0., 0., 0., 1., 1., 1.],
     [0., 0., 0., 1., 1., 1.],
     [0., 0., 1., 1., 1., 1.],
     [1., 1., 1., 1., 1., 1.]],
    [[1., 0., 0., 0., 0., 1.],
     [1., 0., 0., 0., 0., 1.],
     [1., 1., 0., 0., 1., 1.],
     [1., 1., 1., 1., 1., 1.]],
    [[1., 1., 1., 0., 0., 0.],
     [1., 1., 1., 0., 0., 0.],
     [1., 1., 1., 1., 0., 0.],
     [1., 1., 1., 1., 1., 1.]],
    [[1., 1., 1., 1., 1., 1.],
     [0., 0., 1., 1., 1., 1.],
     [0., 0., 0., 1., 1., 1.],
     [0., 0., 0., 1., 1., 1.]],
    [[1., 1., 1., 1., 1., 1.],
     [1., 1., 0., 0., 1., 1.],
     [1., 0., 0., 0., 0., 1.],
     [1., 0., 0., 0., 0., 1.]],
    [[1., 1., 1., 1., 1., 1.],
     [1., 1., 1., 1., 0., 0.],
     [1., 1., 1., 0., 0., 0.],
     [1., 1., 1., 0., 0., 0.]],
])

# Common expected masks for 4x6 tests with blurring.
EXPECTED_BLURRED_MASKS_4x6 = np.array([
    [[0.28770529, 0.37306646, 0.51972448, 0.68434688, 0.82001644, 0.89441662],
     [0.35337468, 0.43282055, 0.56811979, 0.71829136, 0.84071776, 0.90732297],
     [0.44965848, 0.52017537, 0.63853724, 0.767426,   0.87053172, 0.92583059],
     [0.52016574, 0.58396686, 0.68972971, 0.80296752, 0.89199163, 0.93909587]],
    [[0.53260386, 0.4838977,  0.43525844, 0.43525844, 0.4838977,  0.53260386],
     [0.58612609, 0.54129053, 0.49650279, 0.49650279, 0.54129053, 0.58612609],
     [0.66323343, 0.62415411, 0.58509631, 0.58509631, 0.62415411, 0.66323343],
     [0.71875235, 0.68394444, 0.64913995, 0.64913995, 0.68394444, 0.71875235]],
    [[0.89441662, 0.82001644, 0.68434688, 0.51972448, 0.37306646, 0.28770529],
     [0.90732297, 0.84071776, 0.71829136, 0.56811979, 0.43282055, 0.35337468],
     [0.92583059, 0.87053172, 0.767426,   0.63853724, 0.52017537, 0.44965848],
     [0.93909587, 0.89199163, 0.80296752, 0.68972971, 0.58396686, 0.52016574]],
    [[0.52016574, 0.58396686, 0.68972971, 0.80296752, 0.89199163, 0.93909587],
     [0.44965848, 0.52017537, 0.63853724, 0.767426,   0.87053172, 0.92583059],
     [0.35337468, 0.43282055, 0.56811979, 0.71829136, 0.84071776, 0.90732297],
     [0.28770529, 0.37306646, 0.51972448, 0.68434688, 0.82001644, 0.89441662]],
    [[0.71875235, 0.68394444, 0.64913995, 0.64913995, 0.68394444, 0.71875235],
     [0.66323343, 0.62415411, 0.58509631, 0.58509631, 0.62415411, 0.66323343],
     [0.58612609, 0.54129053, 0.49650279, 0.49650279, 0.54129053, 0.58612609],
     [0.53260386, 0.4838977, 0.43525844, 0.43525844, 0.4838977,  0.53260386]],
    [[0.93909587, 0.89199163, 0.80296752, 0.68972971, 0.58396686, 0.52016574],
     [0.92583059, 0.87053172, 0.767426,   0.63853724, 0.52017537, 0.44965848],
     [0.90732297, 0.84071776, 0.71829136, 0.56811979, 0.43282055, 0.35337468],
     [0.89441662, 0.82001644, 0.68434688, 0.51972448, 0.37306646, 0.28770529]],
])
