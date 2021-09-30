import numpy as np
import pytest

from xaitk_saliency.utils.detection import format_detection


class TestFormatDetection:

    def test_default_objectness_fill(self) -> None:
        """
        Test that when objectness scores are not provided the expected default
        is filled into the appropriate column.
        """
        test_bbox_mat = np.random.random_integers(0, 255, 16 * 4).reshape(16, 4)
        test_class_mat = np.random.rand(16, 8)
        combined_mat = format_detection(test_bbox_mat, test_class_mat)
        # The objectness scores should be in the index 4 column and they should
        # all be 1's.
        assert set(combined_mat[:, 4]) == {1}

    def test_explicit_objectness(self) -> None:
        """
        Test that input objectness vector is represented in the output.
        """
        test_bbox_mat = np.random.random_integers(0, 255, 16 * 4).reshape(16, 4)
        test_class_mat = np.random.rand(16, 8)
        test_obj_v = np.tile([.1, .2, .3, .4], 4)
        combined_mat = format_detection(test_bbox_mat, test_class_mat, test_obj_v)
        # that the output objectness column is equivalent to the input.
        assert np.allclose(combined_mat[:, 4], test_obj_v)

    def test_explicit_objectness_2d(self) -> None:
        """
        Test that an input objectness vector that is of shape `[nDets x 1]` is
        treated fine.
        """
        test_bbox_mat = np.random.random_integers(0, 255, 16 * 4).reshape(16, 4)
        test_class_mat = np.random.rand(16, 8)
        test_obj_v = np.tile([.1, .2, .3, .4], 4).reshape(16, 1)
        combined_mat = format_detection(test_bbox_mat, test_class_mat, test_obj_v)
        # Test that the output objectness column is equivalent to the input.
        # NOTE: "close" is sensitive to shape, so (16,) is not equal to (16,1)
        #       even if the content bytes laid out flat are equal, thus the
        #       reshape here.
        assert np.allclose(combined_mat[:, 4], test_obj_v.reshape(16))

    def test_input_nonimpact(self) -> None:
        """
        Test that the invocation of the function does not change the input.
        """
        def gen_fresh_bbox() -> np.ndarray:
            return np.tile([1, 2, 3, 4], 16).reshape(16, 4)

        def gen_fresh_clss() -> np.ndarray:
            return np.tile([.5, .6, .7, .8, .9, .0], 16).reshape(16, 6)

        def gen_fresh_objness() -> np.ndarray:
            return np.tile([.1, .3, .5, .7], 4)

        test_bbox_mat = gen_fresh_bbox()
        test_class_mat = gen_fresh_clss()
        test_obj_v = gen_fresh_objness()

        format_detection(test_bbox_mat, test_class_mat, test_obj_v)

        assert np.allclose(test_bbox_mat, gen_fresh_bbox())
        assert np.allclose(test_class_mat, gen_fresh_clss())
        assert np.allclose(test_obj_v, gen_fresh_objness())

    def test_bbox_class_shape_mismatch(self) -> None:
        """
        Test that an error is raised when there is a `nDets` dimension shape
        mismatch in input bbox and classification matrices.
        """
        # 16 boxes, 14 classifications.
        test_bbox_mat = np.random.random_integers(0, 255, 16 * 4).reshape(16, 4)
        test_class_mat = np.random.rand(14, 8)
        with pytest.raises(
            ValueError,
            match=r"along dimension 0, the array at index 0 has size 16 and "
                  r"the array at index 2 has size 14"
        ):
            format_detection(test_bbox_mat, test_class_mat)

    def test_objectness_shape_mismatch(self) -> None:
        """
        Test that an error is raised when the explicitly input objectness array
        is not a matching size.
        """
        test_bbox_mat = np.random.random_integers(0, 255, 16 * 4).reshape(16, 4)
        test_class_mat = np.random.rand(16, 8)
        test_objnes_v = np.random.rand(11)
        with pytest.raises(
            ValueError,
            match=r"along dimension 0, the array at index 0 has size 16 and "
                  r"the array at index 1 has size 11"
        ):
            format_detection(test_bbox_mat, test_class_mat, test_objnes_v)
