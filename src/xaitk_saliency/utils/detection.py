from typing import Optional

import numpy as np


def format_detection(
    bbox_mat: np.ndarray,
    classification_mat: np.ndarray,
    objectness: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Combine detection and classification output, with optional objectness
    output, into the combined format required for
    :py:meth:`.GenerateDetectorProposalSaliency.generate` ``*_dets`` input
    parameters.

    We enforce some shape consistency so that we can create a valid output
    matrix. The input bounding box matrix should be of shape ``[nDets x 4]``,
    the classification matrix should be of shape ``[nDets x nClasses]``, and
    the objectness vector, if provided, should be of size ``nDets``.

    If an ``objectness`` score vector is not provided, we assume a vector of
    1's.

    The output of this function is a matrix that is of shape
    ``[nDets x (4+1+nClasses)]``. This is the result of horizontally stacking the
    input in bbox, objectness and classification order. The output matrix
    data-type will follow numpy's rules about safe-casting given the
    combination of input matrix types.

    In exceptions about shape mismatches, index 0 refers to the ``bbox_mat``
    input, index 1 refers to the ``objectness`` vector, and index 2 refers to
    the ``classification_mat``.

    :param bbox_mat: Matrix of bounding boxes. This matrix should have the
        shape ``[nDets x 4]``. The format of each row-vector is not important
        but generally expected to be ``[left, top, right, bottom]`` pixel
        coordinates. This matrix must be of a type that is float-castable.
    :param classification_mat: Matrix of classification scores from the
        detector or detection classifier. This should have the shape of
        ``[nDets x nClasses]``. This matrix must be of a type that is
        float-castable.
    :param objectness: Optional vector of objectness scores for input
        detections. This is optional as not all detection models output this
        aspect. When provided, this should be a vector of ints/floats of size
        ``nDets`` to match the other parameter shapes.

    :raises ValueError: When input matrix shapes are mismatched such that they
        cannot be horizontally stacked.

    :returns: Matrix combining bounding box, objectness and class confidences.
    """
    if objectness is None:
        # Using the smallest sized type (boolean) to represent the 1-value so
        # as to not blowup the output type to something larger than input.
        objectness = np.full((bbox_mat.shape[0], 1), fill_value=True)
    elif objectness.ndim == 1:
        objectness = objectness.reshape(objectness.shape[0], 1)
    return np.hstack([bbox_mat, objectness, classification_mat])
