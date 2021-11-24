from xaitk_saliency.impls.gen_classifier_conf_sal.squared_difference_scoring import SquaredDifferenceScoring
from xaitk_saliency import GenerateClassifierConfidenceSaliency

from tests import EXPECTED_MASKS_4x6

import numpy as np
import pytest


class TestSquaredDifferenceScoring:

    def test_init_(self) -> None:
        """
        Test if implementation is usable.
        """

        impl = SquaredDifferenceScoring()

        assert impl.is_usable and isinstance(impl, GenerateClassifierConfidenceSaliency)

    def test_bad_alignment_confs(self) -> None:
        """
        Test that passing a different number of reference confidences and
        perturbed confidences raises the expected exception.
        """

        test_ref = np.ones((4))
        test_pert = np.ones((4, 3))
        test_masks = np.ones((4, 3, 3))

        impl = SquaredDifferenceScoring()

        with pytest.raises(
            ValueError,
            match=r"Number of classes in original image and perturbed image "
                  r"do not match"
        ):
            impl.generate(test_ref, test_pert, test_masks)

    def test_bad_alignment_masks(self) -> None:
        """
        Test that passing a different number of perturbed confidences and masks
        raises the expected exception.
        """

        test_ref = np.ones((4))
        test_pert = np.ones((4, 4))
        test_masks = np.ones((5, 3, 3))

        impl = SquaredDifferenceScoring()

        with pytest.raises(
            ValueError,
            match=r"Number of perturbation masks and respective "
                  r"confidence lengths do not match."
        ):
            impl.generate(test_ref, test_pert, test_masks)

    def test_scores(self) -> None:
        """
        Test for expected output with known input using one class
        """

        test_ref = np.array([0.9, 0.1])
        test_pert = np.array([
            [0.4, 0.6],
            [0.5, 0.5],
            [0.6, 0.4],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.9, 0.1],
        ])

        impl = SquaredDifferenceScoring()
        sal = impl.generate(test_ref, test_pert, EXPECTED_MASKS_4x6)

        assert np.allclose(sal, EXPECTED_SAL)

    def test_config(self) -> None:

        impl = SquaredDifferenceScoring()

        assert impl.get_config() == {}


EXPECTED_SAL = np.array([
    [1.00, 1.00, 0.64, 0.64, 0.36, 0.36],
    [1.00, 1.00, 0.64, 0.64, 0.36, 0.36],
    [0.16, 0.16, 0.04, 0.04, 0.00, 0.00],
    [0.16, 0.16, 0.04, 0.04, 0.00, 0.00],
])
