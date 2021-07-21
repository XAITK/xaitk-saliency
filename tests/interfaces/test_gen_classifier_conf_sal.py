import gc
from typing import Dict, Any
import unittest.mock as mock

import numpy as np

from xaitk_saliency.interfaces.gen_classifier_conf_sal import GenerateClassifierConfidenceSaliency


class StubImpl (GenerateClassifierConfidenceSaliency):
    def generate(
        self,
        image_conf: np.ndarray,
        perturbed_conf: np.ndarray,
        perturbed_masks: np.ndarray,
    ) -> np.ndarray:
        """ Stub impl """

    def get_config(self) -> Dict[str, Any]:
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
    m_image_conf = mock.Mock(spec=np.ndarray)
    m_perturbed_conf = mock.Mock(spec=np.ndarray)
    m_perturbed_masks = mock.Mock(spec=np.ndarray)
    stub(m_image_conf, m_perturbed_conf, m_perturbed_masks)
    stub.generate.assert_called_once_with(m_image_conf, m_perturbed_conf, m_perturbed_masks)
