import gc
from typing import Dict, Any
import unittest.mock as mock

import numpy as np

from xaitk_saliency.interfaces.gen_descriptor_sim_sal import GenerateDescriptorSimilaritySaliency


class StubImpl (GenerateDescriptorSimilaritySaliency):
    def generate(  # type: ignore[empty-body]
        self,
        ref_descr: np.ndarray,
        query_descrs: np.ndarray,
        perturbed_descrs: np.ndarray,
        perturbed_masks: np.ndarray,
    ) -> np.ndarray:
        """ Stub impl """

    def get_config(self) -> Dict[str, Any]:  # type: ignore[empty-body]
        """ Stub impl """


def teardown_module() -> None:
    # Destroy the stub class and collect so as to remove it as an
    # implementation  of the interface
    global StubImpl
    del StubImpl
    gc.collect()


def test_call_alias() -> None:
    """
    Test that the __call__ instance method is an alias to invoke the generate
    instance method.
    """
    stub = StubImpl()
    stub.generate = mock.Mock()  # type: ignore
    m_ref_descr_1 = mock.Mock(spec=np.ndarray)
    m_query_descrs = mock.Mock(spec=np.ndarray)
    m_perturbed_conf = mock.Mock(spec=np.ndarray)
    m_perturbed_masks = mock.Mock(spec=np.ndarray)
    stub(m_ref_descr_1, m_query_descrs, m_perturbed_conf, m_perturbed_masks)
    stub.generate.assert_called_once_with(
        m_ref_descr_1, m_query_descrs, m_perturbed_conf, m_perturbed_masks
    )
