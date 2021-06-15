import gc
from typing import Dict, Any, Generator, Tuple
import unittest.mock as mock

import PIL.Image
import numpy as np

from xaitk_saliency.interfaces.perturb_image import PerturbImage


class StubImpl (PerturbImage):

    def perturb(
        self,
        ref_image: PIL.Image.Image
    ) -> Generator[Tuple[PIL.Image.Image, np.ndarray], None, None]:
        """ Stub impl. """

    def get_config(self) -> Dict[str, Any]:
        """ Stub impl. """


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
    m_img = mock.Mock(spec=PIL.Image.Image)
    stub(m_img)
    stub.perturb.assert_called_once_with(m_img)
