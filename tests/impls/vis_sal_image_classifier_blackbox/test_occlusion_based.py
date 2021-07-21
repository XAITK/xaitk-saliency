import gc
from typing import Dict, Any, Iterator, Sequence, Hashable

import numpy as np
from smqtk_classifier import ClassifyImage
from smqtk_classifier.interfaces.classification_element import CLASSIFICATION_DICT_T
from smqtk_classifier.interfaces.classify_image import IMAGE_ITER_T
from smqtk_core.configuration import configuration_test_helper

from xaitk_saliency import PerturbImage, GenerateClassifierConfidenceSaliency
from xaitk_saliency.impls.vis_sal_image_classifier_blackbox.occlusion_based import PerturbationOcclusion


class TestPerturbationOcclusion:

    def teardown(self) -> None:
        # Collect any temporary implementations so they are not returned during
        # later `*.get_impl()` requests.
        gc.collect()

    def test_configuration(self) -> None:
        """ Test configuration suite using known simple implementations. """
        class StubPI (PerturbImage):
            perturb = None  # type: ignore

            def __init__(self, stub_param: int):
                self.p = stub_param

            def get_config(self) -> Dict[str, Any]:
                return {'stub_param': self.p}

        class StubGen (GenerateClassifierConfidenceSaliency):
            generate = None  # type: ignore

            def __init__(self, stub_param: int):
                self.p = stub_param

            def get_config(self) -> Dict[str, Any]:
                return {'stub_param': self.p}

        test_threads = 87
        test_spi_p = 0
        test_sgn_p = 1
        inst = PerturbationOcclusion(
            StubPI(test_spi_p), StubGen(test_sgn_p), 87
        )
        for inst_i in configuration_test_helper(inst):
            assert inst_i.threads == test_threads
            assert isinstance(inst_i.perturber, StubPI)
            assert inst_i.perturber.p == test_spi_p
            assert isinstance(inst_i.generator, StubGen)
            assert inst_i.generator.p == test_sgn_p

    def test_generate_success(self) -> None:
        """
        Test successfully invoking _generate
        """
        # Stub PerturbImage implementation that just returns ones.
        class StubPI (PerturbImage):
            """ Stub impl that returns known constant masks."""
            def perturb(self, ref_image: np.ndarray) -> np.ndarray:
                return np.ones((3, *ref_image.shape[:2]), dtype=bool)

            get_config = None  # type: ignore

        # Stub saliency map generator that just returns zeros, but the shape
        # should be correct as documented by the interface.
        class StubGen (GenerateClassifierConfidenceSaliency):
            """ Stub impl that returns constant heatmaps. """
            def generate(self, image_conf: np.ndarray, perturbed_conf: np.ndarray,
                         perturbed_masks: np.ndarray) -> np.ndarray:
                return np.zeros((image_conf.shape[0], *perturbed_masks.shape[1:]), dtype=np.float16)

            get_config = None  # type: ignore

        inst = PerturbationOcclusion(
            StubPI(), StubGen()
        )

        # Stub classifier blackbox that returns two class predictions.
        class StubClassifier (ClassifyImage):
            """ Stub that returns a constant classification result. """
            def get_labels(self) -> Sequence[Hashable]:
                return ['a', 'b']

            def classify_images(self, img_iter: IMAGE_ITER_T) -> Iterator[CLASSIFICATION_DICT_T]:
                for _ in img_iter:
                    yield {'a': 1.0, 'b': 0.1}

            get_config = None  # type: ignore

        test_image = np.ones((64, 64, 3), dtype=np.uint8)
        result = inst._generate(test_image, StubClassifier())

        assert result.shape == (2, 64, 64)
