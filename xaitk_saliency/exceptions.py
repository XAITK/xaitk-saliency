class ShapeMismatchError (Exception):
    """
    Exception for when matrix shape expectations are violated.
    """


class MismatchedLabelsError(Exception):
    """
    Raised when two sets of detections do not have matching class labels.
    """

    def __init__(
        self,
        message: str = "Detections have mismatched class labels."
    ):
        self.message = message
        super().__init__(self.message)
