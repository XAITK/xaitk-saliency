"""Utility functions for computing image saliency metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize


def compute_ssd(sal_map: np.ndarray[Any, Any], ref_sal_map: np.ndarray[Any, Any]) -> float:
    """
    Computes the Sum of Squared Differences (SSD) between two saliency maps.

    Args:
        sal_map (np.ndarray): Predicted saliency map of shape (height, width).
        ref_sal_map (np.ndarray): Ground truth/reference saliency map of shape (height, width).

    Returns:
        float: The SSD value between the two saliency maps. If the norm is zero, returns np.inf.
    """
    sum_sq_diff = np.sum(np.power(np.subtract(sal_map, ref_sal_map), 2))
    norm = np.sqrt(np.sum(np.power(sal_map, 2)) * np.sum(np.power(ref_sal_map, 2)))
    if not norm:
        return np.inf
    return sum_sq_diff / norm


def compute_xcorr(sal_map: np.ndarray[Any, Any], ref_sal_map: np.ndarray[Any, Any]) -> float:
    """
    Computes the Normalized Cross-Correlation (NCC) between two saliency maps.

    Args:
        sal_map (np.ndarray): Predicted saliency map of shape (height, width).
        ref_sal_map (np.ndarray): Ground truth/reference saliency map of shape (height, width).

    Returns:
        float: The NCC value between the two saliency maps, ranging from [-1, 1].
    """

    def _normalize(s: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], bool]:
        """Normalizes the saliency map by subtracting the mean and dividing by the standard deviation."""
        s -= s.mean()
        std = s.std()

        if std:
            s /= std

        return s, std == 0

    s1, c1 = _normalize(sal_map.copy())
    s2, c2 = _normalize(ref_sal_map.copy())

    if c1 and not c2:
        return 0.0
    return np.corrcoef(s1.flatten(), s2.flatten())[0, 1]


def compute_iou_coverage(
    saliency_features: np.ndarray[Any, Any],
    ground_truth_features: np.ndarray[Any, Any],
) -> float:
    """Returns the Shared Interest IoU Coverage metric for a single sample.

    IoU Coverage is the number of features in both the ground truth and saliency sets
    divided by the number of features in at least one of the ground truth and saliency
    sets. If one of the inputs is larger in spatial dimensions than the other, the larger
    input is scaled down to the size of the smaller input.

    Args:
        ground_truth_features (np.ndarray): Binary image mask for the GT class/object
            of shape (height, width).
        saliency_features (np.ndarray): Binary saliency map for the predicted class/object
            of shape (height, width).

    Returns:
        float: The computed IoU coverage metric.
    """

    # Determine the target dimensions (smallest among the two)
    target_h = min(ground_truth_features.shape[0], saliency_features.shape[0])
    target_w = min(ground_truth_features.shape[1], saliency_features.shape[1])
    target_shape = (target_h, target_w)

    def _scale_feature_to_target(
        feature: np.ndarray[Any, Any],
        target_shape: tuple[int, int],
    ) -> np.ndarray[Any, Any]:
        """Scales a single 2D feature map to the specified target shape using nearest-neighbor interpolation.

        Args:
            feature (np.ndarray): A 2D array of shape (height, width).
            target_shape (tuple): Target shape as (target_height, target_width).

        Returns:
            np.ndarray: Scaled feature of shape (target_height, target_width).
        """
        # scipy.zoom expects size as (width, height)
        x_zoom_factor = target_shape[1] / feature.shape[1]
        y_zoom_factor = target_shape[0] / feature.shape[0]
        return zoom(feature.astype(np.uint8), (x_zoom_factor, y_zoom_factor), order=1, grid_mode=True, mode="nearest")

    # Scale down if needed.
    if ground_truth_features.shape != target_shape:
        ground_truth_features = _scale_feature_to_target(
            ground_truth_features,
            target_shape,
        )
    if saliency_features.shape != target_shape:
        saliency_features = _scale_feature_to_target(saliency_features, target_shape)

    # Ensure the arrays are binary
    ground_truth_features = ground_truth_features.astype(bool)
    saliency_features = saliency_features.astype(bool)
    # Compute intersection and union
    intersection = np.sum(np.logical_and(ground_truth_features, saliency_features))
    union = np.sum(np.logical_or(ground_truth_features, saliency_features))
    return intersection / np.maximum(union, 1e-10)


def _downsample_to_target(
    arr: np.ndarray[Any, Any],
    target_shape: tuple[int, int],
    is_binary: bool = False,
) -> np.ndarray[Any, Any]:
    """Downsamples the input 2D array to the target spatial shape.

    Args:
        arr (np.ndarray): Input 2D array (height, width).
        target_shape (tuple[int, int]): Tuple consisting of target_height and target_width.
        is_binary (bool): If True, use nearest-neighbor (order=0) without anti-aliasing.

    Returns:
        np.ndarray: The resized array.
    """
    current_shape = arr.shape  # (height, width)
    if current_shape == target_shape:
        return arr
    order = 0 if is_binary else 1
    anti_aliasing = not is_binary
    # Resize the image. Note: preserve_range keeps the original value scale.
    return resize(
        arr,
        (target_shape[0], target_shape[1]),
        preserve_range=True,
        anti_aliasing=anti_aliasing,
        order=order,
    )


def compute_saliency_coverage(
    saliency_features: np.ndarray[Any, Any],
    ground_truth_features: np.ndarray[Any, Any],
) -> np.float64:
    """Returns the Shared Interest Saliency Coverage metric.

    The metric is computed as the saliency in the ground truth region divided by
    the total saliency. If the spatial dimensions of saliency_features and
    ground_truth_features differ, the larger is downsampled to match the smaller.

    Args:
        ground_truth_features (np.ndarray): Binary image mask for the GT class/object
            of shape (height, width).
        saliency_features (np.ndarray): Binary/Continuous saliency map for the predicted class/object
            of shape (height, width).

    Returns:
        numpy.float64: The computed saliency coverage metric.
    """
    # Determine the target spatial shape (the smaller dimensions between the two)
    target_h = min(saliency_features.shape[0], ground_truth_features.shape[0])
    target_w = min(saliency_features.shape[1], ground_truth_features.shape[1])
    target_shape = (target_h, target_w)

    # Downsample the larger array(s) if needed.
    # Assume ground_truth is binary and saliency_features may be continuous.
    if saliency_features.shape != target_shape:
        saliency_features = _downsample_to_target(
            saliency_features,
            target_shape,
            is_binary=False,
        )
    if ground_truth_features.shape != target_shape:
        ground_truth_features = _downsample_to_target(
            ground_truth_features,
            target_shape,
            is_binary=True,
        )

    # Compute the intersection and total saliency
    intersection = np.sum(ground_truth_features * saliency_features)
    explanation_saliency = np.sum(saliency_features)
    return intersection / max(explanation_saliency, 1e-10)


def compute_ground_truth_coverage(
    saliency_features: np.ndarray[Any, Any],
    ground_truth_features: np.ndarray[Any, Any],
) -> np.float64:
    """Returns the Shared Interest Ground Truth Coverage metric.

    The metric is computed as the number of features that occur in both the ground truth
    and saliency feature sets divided by the number of ground truth features.
    If the spatial dimensions differ, the larger array is downsampled to the size of the smaller.

    Args:
        ground_truth_features (np.ndarray): Binary image mask for the GT class/object
            of shape (height, width).
        saliency_features (np.ndarray): Binary saliency map for the predicted class/object
            of shape (height, width).

    Returns:
        numpy.float64: The computed ground truth coverage metric.
    """
    target_h = min(saliency_features.shape[0], ground_truth_features.shape[0])
    target_w = min(saliency_features.shape[1], ground_truth_features.shape[1])
    target_shape = (target_h, target_w)

    # Both inputs are binary; use nearest neighbor resizing.
    if saliency_features.shape[1:] != target_shape:
        saliency_features = _downsample_to_target(
            saliency_features,
            target_shape,
            is_binary=False,
        )
    if ground_truth_features.shape[1:] != target_shape:
        ground_truth_features = _downsample_to_target(
            ground_truth_features,
            target_shape,
            is_binary=True,
        )

    # Compute the intersection and the total ground truth features.
    # Here we sum over all spatial dimensions.
    intersection = np.sum(ground_truth_features * saliency_features)
    ground_truth_saliency = np.sum(ground_truth_features)
    return intersection / max(ground_truth_saliency, 1e-10)
