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
        assert impl.radius == (50, 50)
        assert impl.stride == (20, 20)
        assert impl.sigma is None

    def test_init_valued(self) -> None:
        """
        Test that constructor values pass.
        """
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

    def test_perturb_1channel(self) -> None:
        """
        Test basic perturbation on a known 1-channel image.
        Input image mode should not impact the masks output.
        """
        # Image is slightly wide, should be occluded 6-ways.
        # Simulate grayscale image (single-channel)
        white_image = np.full((4, 6), fill_value=255, dtype=np.uint8)

        impl = SlidingRadial(radius=(2, 4), stride=(2, 2))
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

        impl = SlidingRadial(radius=(2, 4), stride=(2, 2))
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

        impl = SlidingRadial(radius=(2, 4), stride=(2, 2), sigma=(2, 2))
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

        impl = SlidingRadial(radius=(2, 4), stride=(2, 2))
        actual_masks = impl.perturb(white_image)

        assert np.allclose(
            actual_masks,
            EXPECTED_MASKS_4x6
        )


# Common expected masks for 4x6 tests
EXPECTED_MASKS_4x6 = np.array([
    [[0, 0, 0, 0, 1, 1],
     [0, 0, 0, 0, 1, 1],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1]],
    [[0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1]],
    [[1, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0],
     [1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1]],
    [[1, 1, 1, 1, 1, 1],
     [0, 0, 0, 0, 1, 1],
     [0, 0, 0, 0, 1, 1],
     [0, 0, 0, 0, 1, 1]],
    [[1, 1, 1, 1, 1, 1],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0]],
    [[1, 1, 1, 1, 1, 1],
     [1, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0]]
], dtype=bool)

# Common expected masks for 4x6 tests with blurring.
EXPECTED_BLURRED_MASKS_4x6 = np.array([
    [[0., 0.0584771, 0.18115371, 0.35675043, 0.5382167, 0.65533626],
     [0.15234476, 0.20191318, 0.30590063, 0.45474613, 0.6085669, 0.707844],
     [0.36783165, 0.40479892, 0.48235124, 0.5933579, 0.70807517, 0.7821145],
     [0.5201763, 0.54823494, 0.6070981, 0.69135356, 0.7784254, 0.83462214]],
    [[0., 0., 0., 0., 0., 0.],
     [0.1523447, 0.1523447, 0.1523447, 0.1523447, 0.1523447, 0.1523447],
     [0.3678316, 0.3678316, 0.3678316, 0.3678316, 0.3678316, 0.3678316],
     [0.5201763, 0.5201763, 0.5201763, 0.5201763, 0.5201763, 0.5201763]],
    [[0.36687934, 0.28730744, 0.17449367, 0.07927084, 0.02294374, 0.],
     [0.46333194, 0.39588237, 0.30025518, 0.2195391, 0.1717931, 0.1523447],
     [0.5997611, 0.54945827, 0.47814095, 0.41794413, 0.3823359, 0.3678316],
     [0.69621366, 0.65803313, 0.60390246, 0.5582123, 0.5311852, 0.5201763]],
    [[0.28489286, 0.32671022, 0.41443717, 0.5400076, 0.6697754, 0.75352854],
     [0.19732106, 0.2442593, 0.34272927, 0.4836771, 0.62933624, 0.7233457],
     [0.07932705, 0.13316524, 0.24611032, 0.40777743, 0.57484853, 0.6826774],
     [0., 0.05847704, 0.18115371, 0.35675037, 0.5382166, 0.65533626]],
    [[0.28489286, 0.28489286, 0.28489286, 0.28489286, 0.28489286, 0.28489286],
     [0.19732106, 0.19732106, 0.19732106, 0.19732106, 0.19732106, 0.19732106],
     [0.07932705, 0.07932705, 0.07932705, 0.07932705, 0.07932705, 0.07932705],
     [0., 0., 0., 0., 0., 0.]],
    [[0.54725087, 0.49034846, 0.40967453, 0.34158003, 0.30130005, 0.28489286],
     [0.4918074, 0.42793667, 0.3373834, 0.26095015, 0.21573746, 0.19732106],
     [0.41710287, 0.34384316, 0.23997855, 0.1523096, 0.10045069, 0.07932699],
     [0.36687934, 0.28730738, 0.17449361, 0.0792709, 0.02294368, 0.]]
], dtype='float32')
