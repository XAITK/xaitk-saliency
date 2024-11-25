import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper
from syrupy.assertion import SnapshotAssertion

from tests import EXPECTED_MASKS_4x6
from tests.test_utils import CustomFloatSnapshotExtension
from xaitk_saliency.impls.gen_classifier_conf_sal.mc_rise_scoring import MCRISEScoring


@pytest.fixture
def snapshot_custom(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(lambda: CustomFloatSnapshotExtension())


class TestMCRiseScoring:
    def test_init_outofrange_config(self) -> None:
        """Test catching an out of range config value."""
        with pytest.raises(ValueError, match=r"Input p1 value of -0\.3 is not within the expected \[0,1\] range\."):
            MCRISEScoring(k=4, p1=-0.3)

        with pytest.raises(ValueError, match=r"Input p1 value of 5 is not within the expected \[0,1\] range\."):
            MCRISEScoring(k=1, p1=5)

        with pytest.raises(ValueError, match=r"Input k value of 0 is not within the expected >0 range\."):
            MCRISEScoring(k=0, p1=0.2)

    def test_configuration(self) -> None:
        """Test configuration aspects."""
        inst = MCRISEScoring(k=2, p1=0.747)
        for inst_i in configuration_test_helper(inst):
            assert np.allclose(inst_i.p1, 0.747)
            assert inst_i.k == 2

    def test_bad_alignment(self) -> None:
        """Test passing misaligned perturbed confidence vector and masks input."""
        test_confs = np.ones((3, 2))
        test_masks = np.ones((3, 4, 3, 3))

        inst = MCRISEScoring(k=3)

        with pytest.raises(
            ValueError,
            match=r"Number of perturbation masks and respective confidence lengths do not match",
        ):
            inst.generate(test_confs[0], test_confs, test_masks)

    def test_2class_scoring(self, snapshot_custom: SnapshotAssertion) -> None:
        """Test for expected output when given known input data."""
        test_ref_confs = np.array([0, 0.2])
        # Mock classification results for the test masked regions.
        test_pert_confs = np.array([[0.00, 0.33, 0.66, 0.33, 0.66, 1.00], [1.00, 0.66, 0.33, 0.66, 0.33, 0.00]]).T

        inst = MCRISEScoring(k=1)
        sal = inst.generate(test_ref_confs, test_pert_confs, np.asarray([EXPECTED_MASKS_4x6]))

        snapshot_custom.assert_match(sal)
