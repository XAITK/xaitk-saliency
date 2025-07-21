from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

from tests.test_utils import CustomFloatSnapshotExtension
from xaitk_saliency.utils.sal_metrics import (
    compute_ground_truth_coverage,
    compute_iou_coverage,
    compute_saliency_coverage,
    compute_ssd,
    compute_xcorr,
)

# Define two different random number generator seeds
rng_1 = np.random.default_rng(123)
rng_2 = np.random.default_rng(456)

# Define two different Gumbel distribution-based saliency maps
# Note: Gumbel distribution is used to model extreme-values which provide
# a fair approximation for high/low salient feature maps
predicted_saliency_map = rng_1.gumbel(loc=0, scale=0.1, size=(256, 256))
reference_saliency_map = rng_2.gumbel(loc=0, scale=0.1, size=(256, 256))

# Define binary image mask and binary/continuous saliency maps
gt_binary_mask = rng_2.integers(0, 2, size=(256, 256))
saliency_binary_map = rng_1.integers(0, 2, size=(256, 256))
saliency_continous_map = predicted_saliency_map

# Define downsized image mask and saliency maps to test the feature-scaling use-case
gt_downsized_binary_mask = rng_2.integers(0, 2, size=(128, 128))
saliency_downsized_binary_mask = rng_1.integers(0, 2, size=(128, 128))
saliency_downsized_continous_map = rng_1.gumbel(loc=0, scale=0.1, size=(128, 128))


def saliency_metric_assertions(
    computation: Callable,
    sal_map: np.ndarray,
    ref_sal_map: np.ndarray,
) -> None:
    """Test that the inputs are not modified while computing an image metric.

    :param computation: Metric computation function to test.
    :param sal_map: Input saliency map.
    :param ref_sal_map: Reference saliency map for comparison.
    """
    original_sal_map = copy.deepcopy(sal_map)
    original_ref_sal_map = copy.deepcopy(ref_sal_map)
    original_metric_value = computation(original_sal_map, original_ref_sal_map)

    assert np.array_equal(original_sal_map, sal_map), "sal_map modified, data changed"
    assert np.array_equal(original_ref_sal_map, ref_sal_map), "ref_sal_map modified, data changed"

    metric_value = computation(sal_map, ref_sal_map)

    assert original_metric_value == metric_value, "metric_value modified, data changed"


@pytest.fixture
def snapshot_custom(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    return snapshot.use_extension(CustomFloatSnapshotExtension)


class TestComputeSSD:
    """This class contains the unit tests for the functionality of the SSD metric util function."""

    @pytest.mark.parametrize(
        ("sal_map", "ref_sal_map"),
        [(predicted_saliency_map, reference_saliency_map)],
        ids=["Single random input saliency and reference saliency maps"],
    )
    def test_consistency(
        self,
        sal_map: np.ndarray,
        ref_sal_map: np.ndarray,
    ) -> None:
        """Test SSD metric util function with various random saliency map inputs."""

        saliency_metric_assertions(
            computation=compute_ssd,
            sal_map=sal_map,
            ref_sal_map=ref_sal_map,
        )

    @pytest.mark.parametrize(
        ("sal_map", "ref_sal_map"),
        [(predicted_saliency_map, reference_saliency_map)],
        ids=["Single random input saliency and reference saliency maps"],
    )
    def test_regression(
        self,
        sal_map: np.ndarray,
        ref_sal_map: np.ndarray,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Regression testing results to detect API changes."""
        ssd_metric_value = compute_ssd(sal_map=sal_map, ref_sal_map=ref_sal_map)
        snapshot_custom.assert_match(ssd_metric_value)


class TestComputeXCorr:
    """This class contains the unit tests for the functionality of the XCorr metric util function."""

    @pytest.mark.parametrize(
        ("sal_map", "ref_sal_map"),
        [(predicted_saliency_map, reference_saliency_map)],
        ids=["Single random input saliency and reference saliency maps"],
    )
    def test_consistency(
        self,
        sal_map: np.ndarray,
        ref_sal_map: np.ndarray,
    ) -> None:
        """Test XCorr metric util function with various random saliency map inputs."""

        saliency_metric_assertions(
            computation=compute_xcorr,
            sal_map=sal_map,
            ref_sal_map=ref_sal_map,
        )

    @pytest.mark.parametrize(
        ("sal_map", "ref_sal_map"),
        [(predicted_saliency_map, reference_saliency_map)],
        ids=["Single random input saliency and reference saliency maps"],
    )
    def test_regression(
        self,
        sal_map: np.ndarray,
        ref_sal_map: np.ndarray,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Regression testing results to detect API changes."""
        xcorr_metric_value = compute_xcorr(sal_map=sal_map, ref_sal_map=ref_sal_map)
        snapshot_custom.assert_match(xcorr_metric_value)


class TestComputeGroundTruthCoverage:
    """This class contains the unit tests for the functionality of the Ground Truth coverage metric util function."""

    @pytest.mark.parametrize(
        ("saliency_features", "ground_truth_features"),
        [
            (saliency_binary_map, gt_binary_mask),
            (saliency_downsized_binary_mask, gt_binary_mask),
            (saliency_binary_map, gt_downsized_binary_mask),
        ],
        ids=[
            "Single random binary saliency map and image mask",
            "Single random downsized binary saliency map and image mask",
            "Single random binary saliency map and downsized image mask",
        ],
    )
    def test_consistency(
        self,
        saliency_features: np.ndarray,
        ground_truth_features: np.ndarray,
    ) -> None:
        """Test Ground Truth Coverage metric util function with various random saliency map inputs."""

        saliency_metric_assertions(
            computation=compute_ground_truth_coverage,
            sal_map=saliency_features,
            ref_sal_map=ground_truth_features,
        )

    @pytest.mark.parametrize(
        ("saliency_features", "ground_truth_features"),
        [
            (saliency_binary_map, gt_binary_mask),
            (saliency_downsized_binary_mask, gt_binary_mask),
            (saliency_binary_map, gt_downsized_binary_mask),
        ],
        ids=[
            "Single random binary saliency map and image mask",
            "Single random downsized binary saliency map and image mask",
            "Single random binary saliency map and downsized image mask",
        ],
    )
    def test_regression(
        self,
        saliency_features: np.ndarray,
        ground_truth_features: np.ndarray,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Regression testing results to detect API changes."""
        gt_coverage_metric_value = compute_ground_truth_coverage(
            saliency_features=saliency_features,
            ground_truth_features=ground_truth_features,
        )
        snapshot_custom.assert_match(gt_coverage_metric_value)


class TestComputeSaliencyCoverage:
    """This class contains the unit tests for the functionality of the Saliency coverage metric util function."""

    @pytest.mark.parametrize(
        ("saliency_features", "ground_truth_features"),
        [
            (saliency_binary_map, gt_binary_mask),
            (saliency_downsized_binary_mask, gt_binary_mask),
            (saliency_binary_map, gt_downsized_binary_mask),
            (saliency_continous_map, gt_binary_mask),
            (saliency_downsized_continous_map, gt_binary_mask),
            (saliency_continous_map, gt_downsized_binary_mask),
        ],
        ids=[
            "Single random binary saliency map and image mask",
            "Single random downsized binary saliency map and image mask",
            "Single random binary saliency map and downsized image mask",
            "Single random continuous saliency map and binary image mask",
            "Single random downsized continuous saliency map and binary image mask",
            "Single random continuous saliency map and downsized binary image mask",
        ],
    )
    def test_consistency(
        self,
        saliency_features: np.ndarray,
        ground_truth_features: np.ndarray,
    ) -> None:
        """Test Saliency coverage metric util function with various random saliency map inputs."""

        saliency_metric_assertions(
            computation=compute_saliency_coverage,
            sal_map=saliency_features,
            ref_sal_map=ground_truth_features,
        )

    @pytest.mark.parametrize(
        ("saliency_features", "ground_truth_features"),
        [
            (saliency_binary_map, gt_binary_mask),
            (saliency_downsized_binary_mask, gt_binary_mask),
            (saliency_binary_map, gt_downsized_binary_mask),
            (saliency_continous_map, gt_binary_mask),
            (saliency_downsized_continous_map, gt_binary_mask),
            (saliency_continous_map, gt_downsized_binary_mask),
        ],
        ids=[
            "Single random binary saliency map and image mask",
            "Single random downsized binary saliency map and image mask",
            "Single random binary saliency map and downsized image mask",
            "Single random continuous saliency map and binary image mask",
            "Single random downsized continuous saliency map and binary image mask",
            "Single random continuous saliency map and downsized binary image mask",
        ],
    )
    def test_regression(
        self,
        saliency_features: np.ndarray,
        ground_truth_features: np.ndarray,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Regression testing results to detect API changes."""
        saliency_coverage_metric_value = compute_saliency_coverage(
            saliency_features=saliency_features,
            ground_truth_features=ground_truth_features,
        )
        snapshot_custom.assert_match(saliency_coverage_metric_value)


class TestComputeIoUCoverage:
    """This class contains the unit tests for the functionality of the IoU coverage metric util function."""

    @pytest.mark.parametrize(
        ("saliency_features", "ground_truth_features"),
        [
            (saliency_binary_map, gt_binary_mask),
            (saliency_downsized_binary_mask, gt_binary_mask),
            (saliency_binary_map, gt_downsized_binary_mask),
        ],
        ids=[
            "Single random binary saliency map and image mask",
            "Single random downsized binary saliency map and image mask",
            "Single random binary saliency map and downsized image mask",
        ],
    )
    def test_consistency(
        self,
        saliency_features: np.ndarray,
        ground_truth_features: np.ndarray,
    ) -> None:
        """Test IoU coverage metric util function with various random saliency map inputs."""

        saliency_metric_assertions(
            computation=compute_iou_coverage,
            sal_map=saliency_features,
            ref_sal_map=ground_truth_features,
        )

    @pytest.mark.parametrize(
        ("saliency_features", "ground_truth_features"),
        [
            (saliency_binary_map, gt_binary_mask),
            (saliency_downsized_binary_mask, gt_binary_mask),
            (saliency_binary_map, gt_downsized_binary_mask),
        ],
        ids=[
            "Single random binary saliency map and image mask",
            "Single random downsized binary saliency map and image mask",
            "Single random binary saliency map and downsized image mask",
        ],
    )
    def test_regression(
        self,
        saliency_features: np.ndarray,
        ground_truth_features: np.ndarray,
        snapshot_custom: SnapshotAssertion,
    ) -> None:
        """Regression testing results to detect API changes."""
        iou_coverage_metric_value = compute_iou_coverage(
            saliency_features=saliency_features,
            ground_truth_features=ground_truth_features,
        )
        snapshot_custom.assert_match(iou_coverage_metric_value)
