import unittest.mock as mock
import pytest
import numpy as np

from smqtk_descriptors.interfaces.image_descriptor_generator import ImageDescriptorGenerator

from xaitk_saliency import GenerateImageSimilarityBlackboxSaliency
from xaitk_saliency.exceptions import ShapeMismatchError, UnexpectedDimensionsError


def test_generate_checks_success() -> None:
    """
    Test successful passage through the wrapper method.
    """
    m_impl = mock.Mock(spec=GenerateImageSimilarityBlackboxSaliency)

    # test images
    test_ref_image = np.empty((50, 50))
    test_query_image = np.empty((256, 256, 7))

    # mock _generate result with matching height and width to query image
    exp_res = np.ones((256, 256))
    m_impl._generate.return_value = exp_res

    # mock image descriptor generator
    m_desc_generator = mock.Mock(spec=ImageDescriptorGenerator)

    res = GenerateImageSimilarityBlackboxSaliency.generate(
        m_impl,
        test_ref_image,
        test_query_image,
        m_desc_generator
    )

    m_impl._generate.assert_called_once_with(
        test_ref_image,
        test_query_image,
        m_desc_generator
    )

    assert np.array_equal(res, exp_res)


def test_generate_checks_image_shape() -> None:
    """
    Test that the input images conform to our assumption.
    """
    m_impl = mock.Mock(spec=GenerateImageSimilarityBlackboxSaliency)
    m_desc_generator = mock.Mock(spec=ImageDescriptorGenerator)

    # bad ref image
    test_ref_image = np.empty((256,))
    test_query_image = np.empty((256, 400, 3))
    with pytest.raises(
        ValueError,
        match=r"^Input reference image matrix has an unexpected number of dimensions: 1$"
    ):
        GenerateImageSimilarityBlackboxSaliency.generate(
            m_impl,
            test_ref_image,
            test_query_image,
            m_desc_generator
        )

    # bad query image
    test_ref_image = np.empty((123, 345))
    test_query_image = np.empty((1, 2, 3, 4, 5))
    with pytest.raises(
        ValueError,
        match=r"^Input query image matrix has an unexpected number of dimensions: 5$"
    ):
        GenerateImageSimilarityBlackboxSaliency.generate(
            m_impl,
            test_ref_image,
            test_query_image,
            m_desc_generator
        )


def test_generate_checks_num_output_heatmaps() -> None:
    """
    Test that the `_generate` output is checked to be a single heatmap.
    """
    m_impl = mock.Mock(spec=GenerateImageSimilarityBlackboxSaliency)

    # mock _generate result with multiple heatmaps
    m_impl._generate.return_value = np.empty((8, 100, 200))

    # mock image descriptor generator
    m_desc_generator = mock.Mock(spec=ImageDescriptorGenerator)

    test_ref_image = np.empty((256, 123))
    test_query_image = np.empty((100, 200, 3))
    with pytest.raises(
        UnexpectedDimensionsError,
        match=r"^Expected output to be a 2D heatmap matrix but got matrix with "
              r"shape: \(8, 100, 200\)$"
    ):
        GenerateImageSimilarityBlackboxSaliency.generate(
            m_impl,
            test_ref_image,
            test_query_image,
            m_desc_generator
        )


def test_generate_checks_output_shape() -> None:
    """
    Test that the `_generate` output shape is appropriately checked against that
    of the query image.
    """
    m_impl = mock.Mock(spec=GenerateImageSimilarityBlackboxSaliency)

    test_ref_image = np.empty((256, 123))
    test_query_image = np.empty((100, 200, 3))

    # mock _generate result with non-matching height and width to query image
    m_impl._generate.return_value = np.empty((214, 179))

    # mock image descriptor generator
    m_desc_generator = mock.Mock(spec=ImageDescriptorGenerator)

    with pytest.raises(
        ShapeMismatchError,
        match=r"^Output saliency heatmaps did not have matching height and "
              r"width shape components: \(query\) \(100, 200\) != \(214, 179\) "
              r"\(output\)$"
    ):
        GenerateImageSimilarityBlackboxSaliency.generate(
            m_impl,
            test_ref_image,
            test_query_image,
            m_desc_generator
        )


def test_call_alias() -> None:
    """ Test that __call__ is just an alias to the generate method. """
    m_impl = mock.Mock(spec=GenerateImageSimilarityBlackboxSaliency)
    m_ref_img = mock.Mock(spec=np.ndarray)
    m_query_img = mock.Mock(spec=np.ndarray)
    m_bbox = mock.Mock(spec=ImageDescriptorGenerator)

    expected_return = 'expected return'
    m_impl.generate.return_value = expected_return

    test_ret = GenerateImageSimilarityBlackboxSaliency.__call__(
        m_impl, m_ref_img, m_query_img, m_bbox
    )
    m_impl.generate.assert_called_once_with(m_ref_img, m_query_img, m_bbox)
    assert test_ret == expected_return
