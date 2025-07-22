"""Implementation of custom exceptions"""


class ShapeMismatchError(Exception):
    """Exception for when matrix shape expectations are violated."""


class MismatchedLabelsError(Exception):
    """Raised when two sets of detections do not have matching class labels."""

    def __init__(self, message: str = "Detections have mismatched class labels.") -> None:
        """Initialize MismatchedLabelsError"""
        self.message = message
        super().__init__(self.message)


class KWCocoImportError(ImportError):
    """KWCOCO Import Error"""

    def __init__(self) -> None:
        """Initialize KWCocoImportError"""
        super().__init__("kwcoco must be installed. Please install via `xaitk-saliency[tools]`.")
