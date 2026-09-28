import pytest

from .test_notebook_utils import list_error_messages, pyright_analyze


@pytest.mark.notebooks
# require_marker here is an execution gate (needs pyright; only run under -m notebooks),
# not the dependency-canary use seen in the *_optional_deps.py files.
@pytest.mark.usefixtures("require_marker")
@pytest.mark.parametrize(
    ("filepath", "expected_num_errors"),
    [
        ("docs/examples/atari_deepRL_saliency.ipynb", 0),
        ("docs/examples/DRISE.ipynb", 0),
        ("docs/examples/MC_RISE.ipynb", 0),
        ("docs/examples/MNIST_scikit_saliency.ipynb", 0),
        ("docs/examples/OcclusionSaliency.ipynb", 0),
        ("docs/examples/Radial_Image_Perturbation.ipynb", 0),
        ("docs/examples/SimilarityScoring.ipynb", 0),
        ("docs/examples/SuperPixelSaliency.ipynb", 0),
        ("docs/examples/SwappableImplementations.ipynb", 0),
        ("docs/examples/SerializedDetectionSaliency.ipynb", 0),
    ],
)
def test_pyright_nb(filepath: str, expected_num_errors: int) -> None:
    results = pyright_analyze(notebook_path_str=filepath)  # type: ignore
    assert results["summary"]["errorCount"] <= expected_num_errors, list_error_messages(results)  # type: ignore
