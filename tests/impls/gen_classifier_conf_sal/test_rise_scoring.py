import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper

from xaitk_saliency.impls.gen_classifier_conf_sal.rise_scoring import RISEScoring

from tests import DATA_DIR, EXPECTED_MASKS_4x6


class TestRiseScoring:

    def test_init_outofrange_p1(self) -> None:
        """
        Test catching an out of range p1 value.
        """
        with pytest.raises(
            ValueError,
            match=r"Input p1 value of -0\.3 is not within the expected \[0,1\] range\."
        ):
            RISEScoring(p1=-0.3)

        with pytest.raises(
            ValueError,
            match=r"Input p1 value of 5 is not within the expected \[0,1\] range\."
        ):
            RISEScoring(p1=5)

    def test_configuration(self) -> None:
        """
        Test configuration aspects.
        """
        inst = RISEScoring(p1=0.747)
        for inst_i in configuration_test_helper(inst):
            assert inst_i.p1 == 0.747

    def test_bad_alignment(self) -> None:
        """
        Test passing misaligned perturbed confidence vector and masks input.
        """
        test_confs = np.ones((3, 2))
        test_masks = np.ones((4, 3, 3))

        inst = RISEScoring()

        with pytest.raises(
            ValueError,
            match=r"Number of perturbation masks and respective confidence "
                  r"lengths do not match"
        ):
            inst.generate(test_confs[0], test_confs, test_masks)

    def test_1class_scoring(self) -> None:
        """
        Test for expected output when given known input data.
        """
        test_ref_confs = np.array([0])
        # Mock classification results for the test masked regions.
        test_pert_confs = np.array([
            [0.00, 0.33, 0.66,
             0.33, 0.66, 1.00],
            [1.00, 0.66, 0.33,
             0.66, 0.33, 0.00]
        ]).T

        inst = RISEScoring()
        sal = inst.generate(test_ref_confs, test_pert_confs,
                            EXPECTED_MASKS_4x6)

        # This expected output also encodes an expected flip of the test-case
        # saliency map generations (np.flip) which follows the flip in the conf
        # inputs.
        expected_sal = np.load(DATA_DIR / "class_rise_sal.npy")
        assert np.allclose(sal, expected_sal)
