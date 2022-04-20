import os

import numpy as np
import pytest

from xaitk_saliency.impls.gen_descriptor_sim_sal.similarity_scoring import SimilarityScoring
from xaitk_saliency import GenerateDescriptorSimilaritySaliency
from smqtk_core.configuration import configuration_test_helper
from tests import DATA_DIR, EXPECTED_MASKS_4x6


class TestSimilarityScoring:

    def test_init_(self) -> None:
        """
        Test if implementation is usable.
        """
        impl = SimilarityScoring()
        assert impl.is_usable() and isinstance(impl, GenerateDescriptorSimilaritySaliency)

    def test_default_param(self) -> None:
        """
        Test default construction.
        """
        impl = SimilarityScoring()
        assert impl.proximity_metric == 'euclidean'

    def test_get_config(self) -> None:
        """
        Test expected configuration behavior.
        """
        impl = SimilarityScoring('hamming')
        for i in configuration_test_helper(impl):
            assert i.proximity_metric == 'hamming'

    def test_metric_args(self) -> None:
        """
        Test non-default metric type.
        """
        impl = SimilarityScoring('hamming')
        assert impl.proximity_metric == 'hamming'

    def test_invalid_metric(self) -> None:
        """
        Test that a ValueError is raised when an invalid cdist proximity_metric
        is passed.
        """
        with pytest.raises(
            ValueError,
            match=r"Chosen comparison metric not supported or may not be "
                  r"available in scipy"
        ):
            SimilarityScoring('invalid metric')

    def test_generate_mismatch_ref_descriptors(self) -> None:
        """
        Test that we appropriately error when the input reference descriptors
        are not the same dimensionality.
        """
        np.random.seed(0)
        test_ref_descr = np.random.rand(16)
        test_query_descrs = np.random.rand(5, 15)  # Different than above
        test_pert_descrs = np.random.rand(32, 16)
        test_masks = np.random.rand(32, 16, 16)

        impl = SimilarityScoring()

        with pytest.raises(
            ValueError,
            match=r"Size of feature vectors between reference and query images do not match."
        ):
            impl.generate(test_ref_descr, test_query_descrs, test_pert_descrs, test_masks)

    def test_generate_mismatched_perturbed(self) -> None:
        """
        Test that we appropriately error when the input perturbation
        descriptors and mask arrays are not equal in first-dimension length.
        """
        np.random.seed(0)
        test_ref_descr = np.random.rand(16)
        test_query_descrs = np.random.rand(1, 16)
        test_pert_descrs = np.random.rand(32, 16)
        test_masks = np.random.rand(30, 16, 16)  # Different than above

        impl = SimilarityScoring()

        with pytest.raises(
            ValueError,
            match=r"Number of perturbation masks and respective feature "
                  r"vector do not match."
        ):
            impl.generate(test_ref_descr, test_query_descrs, test_pert_descrs, test_masks)

    def test_1_featurevec(self) -> None:
        """
        Test basic scoring with a single feature for broadcasting sanity check.
        """
        impl = SimilarityScoring()
        np.random.seed(2)
        ref_feat = np.random.rand(2048)
        query_feats = np.random.rand(2, 2048)
        pertb_feats = np.random.rand(3, 2048)
        pertb_mask = np.random.randint(low=0, high=2, size=(3, 10, 10), dtype='int')
        sal = impl.generate(ref_feat, query_feats, pertb_feats, pertb_mask)
        assert sal.shape == (2, 10, 10)

    def test_standard_featurevec(self) -> None:
        """
        Test basic scoring on known values and non-square masks.
        """
        impl = SimilarityScoring()
        ref_feat = np.array([0.6, 0.7])
        query_feats = np.array([[0.3, 0.5], [0.1, 0.6], [0.4, 0.4]])
        pertb_feats = np.array([[0.25, 0.9], [0.3, 0.45], [0.8, 0.95],
                                [0.55, 0.2], [0.1, 0.75], [0.35, 0.65]])
        sal = impl.generate(ref_feat, query_feats, pertb_feats, EXPECTED_MASKS_4x6)
        standard_sal = np.load(os.path.join(DATA_DIR, 'SimilaritySal.npy'))
        assert sal.shape == (3, 4, 6)
        assert np.allclose(standard_sal, sal)

    def test_512_featdim(self) -> None:
        """
        Test scoring for features of 512 dims.
        """
        impl = SimilarityScoring()
        np.random.seed(2)
        ref_feat = np.random.rand(512)
        query_feats = np.random.rand(1, 512)
        pertb_feats = np.random.rand(15, 512)
        pertb_mask = np.random.randint(low=0, high=2, size=(15, 10, 10), dtype='int')
        sal = impl.generate(ref_feat, query_feats, pertb_feats, pertb_mask)
        assert sal.shape == (1, 10, 10)
