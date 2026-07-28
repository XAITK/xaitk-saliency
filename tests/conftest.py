import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def set_numpy_printoptions() -> None:
    """Sets global NumPy print options for the entire test session."""
    np.set_printoptions(
        edgeitems=3,
        threshold=1000,
        floatmode="maxprec",
        precision=8,
        suppress=False,
        linewidth=75,
        nanstr="nan",
        infstr="inf",
        sign="-",
        formatter=None,
        legacy=False,
    )


@pytest.fixture
def require_marker(pytestconfig: pytest.Config) -> None:
    markexpr = pytestconfig.getoption("-m")
    if not markexpr or markexpr == "optional":
        pytest.skip("Test requires marker specified")


# PyTest all marker for DSO unit test component. PyTest requires the Config param, even though it's unused
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:  # noqa: ARG001
    for item in items:
        marker_names = {mark.name for mark in item.iter_markers()}
        if "notebooks" not in marker_names:
            item.add_marker(pytest.mark.optional)
        # `required` is an alias for `core` to satisfy a JATIC requirement that
        # required-feature tests carry a marker literally named `required`.
        if "core" in marker_names:
            item.add_marker(pytest.mark.required)
