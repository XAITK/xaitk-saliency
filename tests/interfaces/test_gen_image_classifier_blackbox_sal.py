import unittest.mock as mock

import numpy as np
import pytest
from smqtk_classifier.interfaces.classify_image import ClassifyImage

from xaitk_saliency.interfaces.gen_image_classifier_blackbox_sal import (
    GenerateImageClassifierBlackboxSaliency,
    ShapeMismatchError,
)


def test_generate_checks_success() -> None:
    """Test successful passage through the wrapper method."""
    m_impl = mock.Mock(spec=GenerateImageClassifierBlackboxSaliency)
    # mock implementation result, first dim value doesn't matter here.
    m_impl._generate.return_value = np.empty((3, 256, 256))

    m_clfier = mock.Mock(spec=ClassifyImage)

    # 2-channel image as just HxW should work
    test_image = np.ones((256, 256), dtype=np.uint8)
    GenerateImageClassifierBlackboxSaliency.generate(m_impl, test_image, m_clfier)

    # multi-channel image should work with whatever channel dim size.
    test_image = np.ones((256, 256, 7), dtype=np.uint8)
    GenerateImageClassifierBlackboxSaliency.generate(m_impl, test_image, m_clfier)


def test_generate_checks_image_shape() -> None:
    """Test that the input image shape conforms to our assumption."""
    m_impl = mock.Mock(spec=GenerateImageClassifierBlackboxSaliency)
    m_clfier = mock.Mock(spec=ClassifyImage)

    # A single vector is not being considered an image.
    test_image = np.ones((256,), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"^Input image matrix has an unexpected number of dimensions: 1$"):
        GenerateImageClassifierBlackboxSaliency.generate(m_impl, test_image, m_clfier)

    # Way too many dimensions.
    test_image = np.ones(
        (
            2,
            2,
            2,
            2,
            2,
        ),
        dtype=np.uint8,
    )
    with pytest.raises(ValueError, match=r"^Input image matrix has an unexpected number of dimensions: 5$"):
        GenerateImageClassifierBlackboxSaliency.generate(m_impl, test_image, m_clfier)


def test_generate_checks_output_mismatch() -> None:
    """Test that the `_generate` output shape appropriately checks against the
    input image shape and errors when not matching."""
    m_impl = mock.Mock(spec=GenerateImageClassifierBlackboxSaliency)
    m_clfier = mock.Mock(spec=ClassifyImage)

    # Mismatch the simulated impl output from the input image.
    test_image = np.ones((256, 256), dtype=np.uint8)
    m_impl._generate.return_value = np.empty((3, 128, 128))

    with pytest.raises(
        ShapeMismatchError,
        match=r"^Output saliency heatmaps did not have matching height and "
        r"width shape components: \(ref\) \(256, 256\) != \(128, 128\) "
        r"\(output\)$",
    ):
        GenerateImageClassifierBlackboxSaliency.generate(m_impl, test_image, m_clfier)


def test_call_alias() -> None:
    """Test that __call__ is just an alias to the generate method."""
    m_impl = mock.Mock(spec=GenerateImageClassifierBlackboxSaliency)
    m_img = mock.Mock(spec=np.ndarray)
    m_bbox = mock.Mock(spec=ClassifyImage)

    expected_return = "expected return"
    m_impl.generate.return_value = expected_return

    test_ret = GenerateImageClassifierBlackboxSaliency.__call__(m_impl, m_img, m_bbox)
    m_impl.generate.assert_called_once_with(m_img, m_bbox)
    assert test_ret == expected_return
