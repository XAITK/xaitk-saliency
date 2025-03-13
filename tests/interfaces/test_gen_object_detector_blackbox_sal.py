import unittest.mock as mock

import numpy as np
import pytest
from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects

from xaitk_saliency.exceptions import ShapeMismatchError
from xaitk_saliency.interfaces.gen_object_detector_blackbox_sal import GenerateObjectDetectorBlackboxSaliency


def test_generate_checks_success() -> None:
    """Tests successful passage though the wrapper method."""
    m_impl = mock.Mock(spec=GenerateObjectDetectorBlackboxSaliency)
    # mock implementation result, number of maps should match number of input detections
    m_impl._generate.return_value = np.empty((5, 256, 256))

    m_detector = mock.Mock(spec=DetectImageObjects)

    # test reference detections inputs with matching lengths
    test_bboxes = np.ones((5, 4), dtype=float)
    test_scores = np.ones((5, 3), dtype=float)
    test_objectness = np.ones((5,), dtype=float)

    # 2-channel image as just HxW should work
    test_image = np.ones((256, 256), dtype=np.uint8)
    GenerateObjectDetectorBlackboxSaliency.generate(
        m_impl,
        test_image,
        test_bboxes,
        test_scores,
        m_detector,
    )

    m_impl._generate.assert_called_with(test_image, test_bboxes, test_scores, m_detector, None)  # no objectness passed

    # multi-channel image shoudl work with whatever channel dim size
    test_image = np.ones((256, 256, 7), dtype=np.uint8)
    GenerateObjectDetectorBlackboxSaliency.generate(
        m_impl,
        test_image,
        test_bboxes,
        test_scores,
        m_detector,
    )

    m_impl._generate.assert_called_with(test_image, test_bboxes, test_scores, m_detector, None)  # no objectness passed

    # With objectness
    GenerateObjectDetectorBlackboxSaliency.generate(
        m_impl,
        test_image,
        test_bboxes,
        test_scores,
        m_detector,
        test_objectness,
    )

    m_impl._generate.assert_called_with(test_image, test_bboxes, test_scores, m_detector, test_objectness)


def test_generate_checks_image_shape() -> None:
    """Test that the input image shape conforms to our assumption."""
    m_impl = mock.Mock(spec=GenerateObjectDetectorBlackboxSaliency)

    m_bboxes = mock.Mock(spec=np.ndarray)
    m_scores = mock.Mock(spec=np.ndarray)

    # a single vector is not considered an image
    test_image = np.ones((256,), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"^Input image matrix has an unexpected number of dimensions: 1$"):
        GenerateObjectDetectorBlackboxSaliency._verify_generate_inputs(
            m_impl,
            test_image,
            m_bboxes,
            m_scores,
        )

    # image with more than 3 dimenstions
    test_image = np.ones((256, 256, 3, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"^Input image matrix has an unexpected number of dimensions: 4$"):
        GenerateObjectDetectorBlackboxSaliency._verify_generate_inputs(
            m_impl,
            test_image,
            m_bboxes,
            m_scores,
        )


def test_generate_checks_detection_inputs_length() -> None:
    """Test that the reference detection inputs must all have the same length."""
    m_impl = mock.Mock(spec=GenerateObjectDetectorBlackboxSaliency)

    test_image = np.ones((64, 64))

    # Mismatched number of bboxes and scores, without objectness
    test_bboxes = np.ones((4, 4), dtype=float)
    test_scores = np.ones((5, 3), dtype=float)
    with pytest.raises(
        ValueError,
        match=r"^Number of input bounding boxes and scores do not match: \(bboxes\) 4 != 5 \(scores\)$",
    ):
        GenerateObjectDetectorBlackboxSaliency._verify_generate_inputs(m_impl, test_image, test_bboxes, test_scores)

    # Mismatched number of bboxes and scores, with objectness
    test_bboxes = np.ones((5, 4), dtype=float)
    test_scores = np.ones((4, 3), dtype=float)
    test_objectness = np.ones((5,), dtype=float)
    with pytest.raises(
        ValueError,
        match=r"^Number of input bounding boxes, scores, and objectness "
        r"scores do not match: \(bboxes\) 5 != 4 \(scores\) and/or "
        r"\(bboxes\) 5 != 5 \(objectness\)$",
    ):
        GenerateObjectDetectorBlackboxSaliency._verify_generate_inputs(
            m_impl,
            test_image,
            test_bboxes,
            test_scores,
            test_objectness,
        )

    # Different number of objectness scores
    test_bboxes = np.ones((5, 4), dtype=float)
    test_scores = np.ones((5, 3), dtype=float)
    test_objectness = np.ones((4,), dtype=float)
    with pytest.raises(
        ValueError,
        match=r"^Number of input bounding boxes, scores, and objectness "
        r"scores do not match: \(bboxes\) 5 != 5 \(scores\) and/or "
        r"\(bboxes\) 5 != 4 \(objectness\)$",
    ):
        GenerateObjectDetectorBlackboxSaliency._verify_generate_inputs(
            m_impl,
            test_image,
            test_bboxes,
            test_scores,
            test_objectness,
        )


def test_generate_checks_bboxes_width() -> None:
    """Test that the input bounding boxes must have a width of 4."""
    m_impl = mock.Mock(spec=GenerateObjectDetectorBlackboxSaliency)

    test_bboxes = np.ones((2, 3), dtype=float)
    test_scores = np.ones((2, 3), dtype=float)

    test_image = np.ones((256, 256), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"^Input bounding boxes matrix has width of 3, should have width of 4$"):
        GenerateObjectDetectorBlackboxSaliency._verify_generate_inputs(
            m_impl,
            test_image,
            test_bboxes,
            test_scores,
        )


def test_generate_checks_output_shape_mismatch() -> None:
    """
    Test that the appropriate error is raised when the output of `_generate`
    has a different shape than the reference image.
    """
    m_impl = mock.Mock(spec=GenerateObjectDetectorBlackboxSaliency)

    m_detector = mock.Mock(spec=DetectImageObjects)

    test_bboxes = np.ones((2, 4), dtype=float)
    test_scores = np.ones((2, 3), dtype=float)

    # Mismatch the simulated impl output from the input image.
    test_image = np.ones((256, 256), dtype=np.uint8)
    m_impl._generate.return_value = np.empty((2, 128, 128))

    with pytest.raises(
        ShapeMismatchError,
        match=r"^Output saliency heatmaps did not have matching height and "
        r"width shape components: \(ref\) \(256, 256\) != \(128, 128\) "
        r"\(output\)$",
    ):
        GenerateObjectDetectorBlackboxSaliency.generate(
            m_impl,
            test_image,
            test_bboxes,
            test_scores,
            m_detector,
        )


def test_generate_checks_output_quantity_mismatch() -> None:
    """
    Test that the appropriate error is raised when the quantity of heatmaps
    output from `_generate` does not match the quantity of input reference
    detections.
    """
    m_impl = mock.Mock(spec=GenerateObjectDetectorBlackboxSaliency)

    m_detector = mock.Mock(spec=DetectImageObjects)

    # three input reference detections
    test_bboxes = np.ones((3, 4), dtype=float)
    test_scores = np.ones((3, 2), dtype=float)

    test_image = np.ones((164, 164), dtype=np.uint8)

    # two output heatmaps
    m_impl._generate.return_value = np.empty((2, 164, 164))

    with pytest.raises(
        ShapeMismatchError,
        match=r"^Quantity of output heatmaps does not match the quantity of "
        r"input reference detections: \(input\) 3 != 2 \(output\)$",
    ):
        GenerateObjectDetectorBlackboxSaliency.generate(
            m_impl,
            test_image,
            test_bboxes,
            test_scores,
            m_detector,
        )


def test_call_alias() -> None:
    """Test that __call__ is just an alias to the generate method."""
    m_impl = mock.Mock(spec=GenerateObjectDetectorBlackboxSaliency)
    m_img = mock.Mock(spec=np.ndarray)
    m_detector = mock.Mock(spec=DetectImageObjects)

    test_bboxes = np.ones((3, 4), dtype=float)
    test_scores = np.ones((3, 3), dtype=float)

    expected_return = "test return"
    m_impl.generate.return_value = expected_return

    test_ret = GenerateObjectDetectorBlackboxSaliency.__call__(
        m_impl,
        m_img,
        test_bboxes,
        test_scores,
        m_detector,
    )

    m_impl.generate.assert_called_once_with(m_img, test_bboxes, test_scores, m_detector, None)  # no objectness passed
    assert test_ret == expected_return


def test_return_empty_map() -> None:
    """Test that an empty array of maps is returned properly"""
    m_impl = mock.Mock(spec=GenerateObjectDetectorBlackboxSaliency)
    m_detector = mock.Mock(spec=DetectImageObjects)

    # test reference detections inputs with matching lengths
    test_bboxes = np.ones((5, 4), dtype=float)
    test_scores = np.ones((5, 3), dtype=float)

    # 2-channel image as just HxW should work
    test_image = np.ones((256, 256), dtype=np.uint8)

    expected_return = np.array([])
    m_impl._generate.return_value = expected_return

    test_ret = GenerateObjectDetectorBlackboxSaliency.generate(
        m_impl,
        test_image,
        test_bboxes,
        test_scores,
        m_detector,
    )

    m_impl._generate.assert_called_with(
        test_image,
        test_bboxes,
        test_scores,
        m_detector,
        None,  # no objectness passed
    )
    assert len(test_ret) == 0
