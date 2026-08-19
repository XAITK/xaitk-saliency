import gc
import unittest.mock as mock
from typing import Any

import numpy as np
import pytest
from typing_extensions import override

from xaitk_saliency.interfaces.gen_detector_prop_sal import GenerateDetectorProposalSaliency


class StubImpl(GenerateDetectorProposalSaliency):
    @override
    def generate(
        self,
        *,
        ref_dets: np.ndarray,
        perturbed_dets: np.ndarray,
        perturbed_masks: np.ndarray,
    ) -> np.ndarray:
        """Stub impl."""
        return np.zeros((1, 1))

    def get_config(self) -> dict[str, Any]:  # type: ignore[empty-body]
        """Stub impl."""


def teardown_module() -> None:
    # Destroy the stub class and collect so as to remove it as an
    # implementation  of the interface
    global StubImpl
    del StubImpl
    gc.collect()


@pytest.mark.core
def test_call_alias() -> None:
    """Test that the __call__ instance method is an alias to invoke the generate instance method."""
    stub = StubImpl()
    stub.generate = mock.Mock()  # type: ignore
    m_ref_dets = mock.Mock(spec=np.ndarray)
    m_perturbed_dets = mock.Mock(spec=np.ndarray)
    m_perturbed_masks = mock.Mock(spec=np.ndarray)
    stub(ref_dets=m_ref_dets, perturbed_dets=m_perturbed_dets, perturbed_masks=m_perturbed_masks)
    stub.generate.assert_called_once_with(
        ref_dets=m_ref_dets,
        perturbed_dets=m_perturbed_dets,
        perturbed_masks=m_perturbed_masks,
    )
