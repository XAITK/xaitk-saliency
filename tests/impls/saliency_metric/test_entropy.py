from __future__ import annotations

from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper
from syrupy.assertion import SnapshotAssertion

from tests.test_utils import CustomFloatSnapshotExtension
from xaitk_saliency.impls.saliency_metric.entropy import Entropy

from .saliency_metric_utils import saliency_metric_assertions

rng = np.random.default_rng()


@pytest.fixture
def snapshot_custom(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(CustomFloatSnapshotExtension)


class TestEntropy:
    """This class contains the unit tests for the functionality of the Entropy metric impl."""

    @pytest.mark.parametrize(
        ("sal_map", "expectation"),
        [
            (
                rng.integers(0, 255, (256, 256), dtype=np.uint8),
                does_not_raise(),
            ),
        ],
        ids=[
            "Single random input saliency map",
        ],
    )
    def test_consistency(
        self,
        sal_map: np.ndarray,
        expectation: AbstractContextManager,
    ) -> None:
        """Test Entropy metric call with various random saliency map inputs and parameters."""
        with expectation:
            inst = Entropy(clip_range=None)

            # Test the compute method directly
            compute_snr = saliency_metric_assertions(
                computation=inst.compute,
                sal_map=sal_map,
            )

            # Test callable
            callable_snr = saliency_metric_assertions(
                computation=inst,
                sal_map=sal_map,
            )

            assert compute_snr == callable_snr

    @pytest.mark.parametrize(
        ("clip_range"),
        [(None), ((-1.0, 1.0)), ((0.0, 1.0)), ((-1.0, 0.0))],
        ids=[
            "None",
            "Full Saliency Entropy",
            "Positive saliency entropy",
            "Negative saliency entropy",
        ],
    )
    def test_regression(
        self,
        clip_range: tuple[float, float] | None,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Regression testing results to detect API changes."""
        local_rng = np.random.default_rng(seed=12345)
        random_sal_map = local_rng.gumbel(loc=0, scale=0.1, size=(64, 64))
        random_sal_map /= random_sal_map.sum()
        inst = Entropy(clip_range=clip_range)
        entropy_metric_value = inst.compute(random_sal_map)
        snapshot_custom.assert_match(entropy_metric_value)

    @pytest.mark.parametrize(
        ("clip_range"),
        [(None), ((-1.0, 1.0))],
        ids=[
            "None",
            "Valid range tuple",
        ],
    )
    def test_configuration(self, clip_range: tuple[float, float] | None) -> None:
        """Ensure get_config is correct."""
        inst = Entropy(clip_range=clip_range)

        for inst_i in configuration_test_helper(inst):
            assert inst_i.clip_range == clip_range

    def test_classname(self) -> None:
        entropy_metric = Entropy(clip_range=None)
        assert entropy_metric.name == "Entropy"
