import unittest.mock as mock
import pytest
import numpy as np

from smqtk_descriptors.interfaces.image_descriptor_generator import ImageDescriptorGenerator

from xaitk_saliency import GenerateImageSimilarityBlackboxSaliency
from xaitk_saliency.exceptions import ShapeMismatchError


def test_generate_checks_success() -> None:
    """
    Test successful passage through the wrapper method.
    """
    m_impl = mock.Mock(spec=GenerateImageSimilarityBlackboxSaliency)

    # test images
    test_ref_image = np.empty((50, 50))
    test_query_images = [np.empty((256, 256, 7))] * 3

    # mock _generate result with 3 saliency maps and matching height and width to reference image
    exp_res = np.ones((3, 50, 50))
    m_impl._generate.return_value = exp_res

    # mock image descriptor generator
    m_desc_generator = mock.Mock(spec=ImageDescriptorGenerator)

    res = GenerateImageSimilarityBlackboxSaliency.generate(
        m_impl,
        test_ref_image,
        test_query_images,
        m_desc_generator
    )

    m_impl._generate.assert_called_once_with(
        test_ref_image,
        test_query_images,
        m_desc_generator
    )

    assert np.array_equal(res, exp_res)


def test_generate_checks_image_shape() -> None:
    """
    Test that the input reference image conforms to our assumption.
    """
    m_impl = mock.Mock(spec=GenerateImageSimilarityBlackboxSaliency)
    m_desc_generator = mock.Mock(spec=ImageDescriptorGenerator)

    # too few dimensions
    test_ref_image = np.empty((256,))
    test_query_images = [np.empty((256, 400, 3))] * 2
    with pytest.raises(
        ValueError,
        match=r"^Input reference image matrix has an unexpected number of dimensions: 1$"
    ):
        GenerateImageSimilarityBlackboxSaliency.generate(
            m_impl,
            test_ref_image,
            test_query_images,
            m_desc_generator
        )

    # too many dimensions
    test_ref_image = np.empty((256, 256, 1, 1))
    with pytest.raises(
        ValueError,
        match=r"^Input reference image matrix has an unexpected number of dimensions: 4$"
    ):
        GenerateImageSimilarityBlackboxSaliency.generate(
            m_impl,
            test_ref_image,
            test_query_images,
            m_desc_generator
        )


def test_generate_checks_output_shape() -> None:
    """
    Test that the `_generate` output shape is appropriately checked against that
    of the reference image.
    """
    m_impl = mock.Mock(spec=GenerateImageSimilarityBlackboxSaliency)

    test_ref_image = np.empty((256, 123))
    test_query_images = [np.empty((100, 200, 3))] * 2

    # mock _generate result with matching length to the query images but
    # non-matching height and width to reference image
    m_impl._generate.return_value = np.empty((2, 214, 179))

    # mock image descriptor generator
    m_desc_generator = mock.Mock(spec=ImageDescriptorGenerator)

    with pytest.raises(
        ShapeMismatchError,
        match=r"^Output saliency heatmaps did not have matching height and "
              r"width shape components: \(reference\) \(256, 123\) != \(214, 179\) "
              r"\(output\)$"
    ):
        GenerateImageSimilarityBlackboxSaliency.generate(
            m_impl,
            test_ref_image,
            test_query_images,
            m_desc_generator
        )


def test_generate_checkout_num_saliency_maps() -> None:
    """
    Test that the number of output heatmaps is checked against the number of
    input query images.
    """
    m_impl = mock.Mock(spec=GenerateImageSimilarityBlackboxSaliency)

    test_ref_image = np.empty((55, 44, 6))
    test_query_images = [np.empty((15, 7))] * 7

    # mock _generate result with non-matching length to query images
    m_impl._generate.return_value = np.empty((3, 55, 44))

    # mock image descriptor generator
    m_desc_generator = mock.Mock(spec=ImageDescriptorGenerator)

    with pytest.raises(
        ShapeMismatchError,
        match=r"^Number of output saliency heatmaps did not match number of "
              r"input query images: \(heatmaps\) 3 != 7 \(query images\)$"
    ):
        GenerateImageSimilarityBlackboxSaliency.generate(
            m_impl,
            test_ref_image,
            test_query_images,
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
