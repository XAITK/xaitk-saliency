"""Implementation of custom exceptions."""


class ShapeMismatchError(Exception):
    """Exception for when matrix shape expectations are violated."""


class MismatchedLabelsError(Exception):
    """Raised when two sets of detections do not have matching class labels."""

    def __init__(self, message: str = "Detections have mismatched class labels.") -> None:
        """Initialize MismatchedLabelsError."""
        self.message = message
        super().__init__(self.message)


class ToolsImportError(ImportError):
    """Tools (click, kwcoco, matplotlib) Import Error."""

    def __init__(self) -> None:
        """Initialize ToolsImportError."""
        super().__init__("This feature requires additional dependencies. Please install via `xaitk-saliency[tools]`.")


class MaiteImportError(ImportError):
    """MAITE Import Error."""

    def __init__(self, class_name: str, *, import_error: ImportError | None = None) -> None:
        """Initialize MaiteImportError."""
        message = f"{class_name} requires the `maite` extra. Install with: `pip install xaitk-saliency[maite]`"
        if import_error is not None:
            message += (
                f"\n\nIf the extra is already installed, the following upstream error may be the cause:"
                f"\n  {type(import_error).__name__}: {import_error}"
            )
        super().__init__(message)
