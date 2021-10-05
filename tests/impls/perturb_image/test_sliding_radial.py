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
     [0., 0., 1., 1., 1., 1.],
     [0., 1., 1., 1., 1., 1.],
     [1., 1., 1., 1., 1., 1.]],
    [[0., 0., 0., 0., 0., 1.],
     [1., 0., 0., 0., 1., 1.],
     [1., 1., 0., 1., 1., 1.],
     [1., 1., 1., 1., 1., 1.]],
    [[1., 1., 0., 0., 0., 0.],
     [1., 1., 1., 0., 0., 0.],
     [1., 1., 1., 1., 0., 1.],
     [1., 1., 1., 1., 1., 1.]],
    [[0., 1., 1., 1., 1., 1.],
     [0., 0., 1., 1., 1., 1.],
     [0., 0., 0., 1., 1., 1.],
     [0., 0., 1., 1., 1., 1.]],
    [[1., 1., 0., 1., 1., 1.],
     [1., 0., 0., 0., 1., 1.],
     [0., 0., 0., 0., 0., 1.],
     [1., 0., 0., 0., 1., 1.]],
    [[1., 1., 1., 1., 0., 1.],
     [1., 1., 1., 0., 0., 0.],
     [1., 1., 0., 0., 0., 0.],
     [1., 1., 1., 0., 0., 0.]]
])

# Common expected masks for 4x6 tests with blurring.
EXPECTED_BLURRED_MASKS_4x6 = np.array([
    [[0., 0.14060639, 0.3684608, 0.6055491, 0.7870983, 0.8814234],
     [0.11828791, 0.2463867, 0.45155752, 0.66162825, 0.81982994, 0.9009696],
     [0.28718147, 0.3967774, 0.5688757, 0.7401796, 0.8653188, 0.9279456],
     [0.40775135, 0.50368714, 0.6516967, 0.7951928, 0.89692104, 0.94655156]],
    [[0.00304978, 0., 0.03676931, 0.144639, 0.29384592, 0.40327683],
     [0.14811273, 0.1421291, 0.17103323, 0.2648694, 0.3970375, 0.4943345],
     [0.35128394, 0.3417027, 0.35997337, 0.43389973, 0.54148114, 0.62119234],
     [0.4935506, 0.48181382, 0.49291036, 0.55271417, 0.6425658, 0.70954275]],
    [[0.74328494, 0.6296464, 0.44116887, 0.23976363, 0.08246047, 0.],
     [0.7842362, 0.6852854, 0.5199054, 0.3414799, 0.20092557, 0.12683398],
     [0.8410181, 0.76295763, 0.63066375, 0.48550427, 0.36948237, 0.30776653],
     [0.8803714, 0.8171652, 0.7085613, 0.5874625, 0.48938036, 0.43679395]],
    [[0.13464972, 0.27303606, 0.48789603, 0.6978908, 0.84783155, 0.92130893],
     [0.09269758, 0.23386501, 0.455219, 0.67466205, 0.83381104, 0.9128079],
     [0.03692145, 0.18190618, 0.4120327, 0.6440924, 0.81544554, 0.90172434],
     [0., 0.14760496, 0.38364652, 0.6241008, 0.8035026, 0.894558]],
    [[0.22468002, 0.20661545, 0.22349325, 0.3151755, 0.45326152, 0.55636704],
     [0.15831259, 0.14193895, 0.16266125, 0.26062152, 0.4061635, 0.51459485],
     [0.07063648, 0.05634791, 0.08203695, 0.188367, 0.34397882, 0.45962927],
     [0.01303541, 0., 0.02886414, 0.14075261, 0.30315253, 0.4236895]],
    [[0.8297765, 0.73840344, 0.58104193, 0.40472543, 0.26079497, 0.18294823],
     [0.8121847, 0.714369, 0.5468448, 0.36023647, 0.20847073, 0.12647676],
     [0.78911746, 0.68268037, 0.50148207, 0.30091003, 0.13841493, 0.05070167],
     [0.77409834, 0.6619108, 0.4715341, 0.26150006, 0.09165931, 0.]]
])
