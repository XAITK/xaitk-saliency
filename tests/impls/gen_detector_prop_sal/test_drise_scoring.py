import numpy as np
import pytest
import os
import re

from xaitk_saliency.impls.gen_detector_prop_sal.drise_scoring import DRISEScoring

from xaitk_saliency import GenerateDetectorProposalSaliency
from tests import DATA_DIR, EXPECTED_MASKS_4x6


class TestDRISEScoring:

    def test_init_(self) -> None:
        """
        Test if implementation is usable.
        """
        impl = DRISEScoring()
        assert impl.is_usable() and isinstance(impl, GenerateDetectorProposalSaliency)

    def test_shape_sanity(self) -> None:
        """
        Test basic scoring with a single feature for broadcasting sanity check.
        """
        impl = DRISEScoring()
        rng = np.random.default_rng(seed=0)
        ref_dets = rng.standard_normal((2, 7))
        pert_dets = rng.standard_normal((10, 3, 7))
        pert_mask = rng.integers(
            low=0, high=2, size=(10, 15, 25), dtype='int')
        sal = impl.generate(ref_dets, pert_dets, pert_mask)
        assert sal.shape == (2, 15, 25)

    def test_standard_detection(self) -> None:
        """
        Test basic scoring on known values and non-square masks.
        """
        impl = DRISEScoring()
        ref_dets = np.array([[1, 1, 4, 3, 0.89, 0, 1]])
        pert_dets = np.array([[[1, 2, 6, 6, 0.995, 0.5, 0.5]],
                              [[0, 1, 2, 2, 0.03, 0.2, 0.8]],
                              [[1, 0, 2, 2, 0.81, 0.45, 0.55]],
                              [[1, 1, 6, 6, 0.625, 0.25, 0.75]],
                              [[0, 2, 3, 5, 0.56, 0.03, 0.97]],
                              [[1, 2, 6, 3, 0.07, 0.01, 0.99]], ])
        sal = impl.generate(ref_dets, pert_dets, EXPECTED_MASKS_4x6)
        standard_sal = np.load(os.path.join(DATA_DIR, 'drisesal.npy'))
        assert sal.shape == (1, 4, 6)
        assert np.allclose(standard_sal, sal)

    def test_mask_detections_size_mismatch(self) -> None:
        """
        Test size mismatch between perturbed detections and masks.
        """
        impl = DRISEScoring()
        rng = np.random.default_rng(seed=0)
        ref_dets = rng.standard_normal((2, 7))
        pert_dets = rng.standard_normal((9, 3, 7))  # ONE LESS than pert mask mat.
        pert_mask = rng.integers(
            low=0, high=2, size=(10, 15, 25), dtype='int')
        with pytest.raises(
            ValueError,
            match=r"Number of perturbation masks and respective "
                  r"detections vector do not match."
        ):
            impl.generate(ref_dets, pert_dets, pert_mask)

    def test_detections_classes_mismatch(self) -> None:
        """
        Test mismatch in number of classes between perturbed and reference detections.
        """
        impl = DRISEScoring()
        rng = np.random.default_rng(seed=0)
        ref_dets = rng.standard_normal((2, 8))  # ONE MORE than pert dets mat.
        pert_dets = rng.standard_normal((10, 3, 7))
        pert_mask = rng.integers(
            low=0, high=2, size=(10, 15, 25), dtype='int')
        with pytest.raises(
            ValueError,
            match=re.escape(r"Dimensions of reference detections and "
                            r"perturbed detections do not match. Both "
                            r"should be of dimension (n_classes + 4 + 1).")
        ):
            impl.generate(ref_dets, pert_dets, pert_mask)

    def test_config(self) -> None:

        impl = DRISEScoring()

        assert impl.get_config() == {}
