import gc
import unittest.mock as mock
from typing import Any

import numpy as np

from xaitk_saliency.interfaces.perturb_image import PerturbImage


class StubImpl(PerturbImage):
    def perturb(self, ref_image: np.ndarray) -> np.ndarray:  # type: ignore[empty-body]
        """Stub impl."""

    def get_config(self) -> dict[str, Any]:  # type: ignore[empty-body]
        """Stub impl."""


def teardown_module() -> None:
    # Destroy the stub class and collect so as to remove it as an
    # implementation of the interface
    global StubImpl
    del StubImpl
    gc.collect()


def test_call_alias() -> None:
    """
    Test that the __call__ instance method is an alias to invoke the perturb
    instance method.
    """
    stub = StubImpl()
    stub.perturb = mock.Mock()  # type: ignore
    m_img = mock.Mock(spec=np.ndarray)
    stub(m_img)
    stub.perturb.assert_called_once_with(m_img)
