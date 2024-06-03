import numpy as np
from typing import Iterable, Optional
import gc

from smqtk_descriptors.interfaces.image_descriptor_generator import ImageDescriptorGenerator
from smqtk_core.configuration import configuration_test_helper

from xaitk_saliency.impls.gen_image_similarity_blackbox_sal.sbsm import (
    SBSMStack,
    SlidingWindow,
    SimilarityScoring
)

from tests import DATA_DIR


class TestBlackBoxSBSM:

    def teardown(self) -> None:
        # Collect any temporary implementations so they are not returned during
        # later `*.get_impl()` requests.
        gc.collect()  # pragma: no cover

    def test_configuration(self) -> None:
        """
        Test configuration suite.
        """
        inst = SBSMStack(
            window_size=(10, 7),
            stride=(9, 3),
            proximity_metric='rogerstanimoto',
            threads=15
        )

        for inst_i in configuration_test_helper(inst):
            inst_p = inst_i._po._perturber
            inst_g = inst_i._po._generator

            assert isinstance(inst_p, SlidingWindow)
            assert isinstance(inst_g, SimilarityScoring)
            assert inst_p.window_size == (10, 7)
            assert inst_p.stride == (9, 3)
            assert inst_g.proximity_metric == 'rogerstanimoto'
            assert inst_i._po._threads == 15

    def test_generation_rgb(self) -> None:
        """
        Test basic generation functionality with dummy inputs.
        """
        class TestDescriptorGenerator (ImageDescriptorGenerator):
            """
            Dummy descriptor generator that returns known feature vectors.
            """

            def generate_arrays_from_images(
                self,
                img_mat_iter: Iterable[Optional[np.ndarray]]
            ) -> Iterable[np.ndarray]:
                # return repeatable random feature vectors
                rng = np.random.default_rng(0)
                for _ in img_mat_iter:
                    yield rng.random(4)

            get_config = None  # type: ignore

        test_desc_gen = TestDescriptorGenerator()
        test_ref_img = np.full((25, 32, 3), fill_value=255, dtype=np.uint8)
        test_query_imgs = [np.full((27, 28), fill_value=255, dtype=np.uint8)] * 2

        inst = SBSMStack(
            window_size=(4, 5),
            stride=(2, 3),
            proximity_metric='euclidean'
        )

        sal_maps = inst.generate(
            test_ref_img,
            test_query_imgs,
            test_desc_gen
        )

        assert sal_maps.shape == (2, 25, 32)

        exp_res = np.load(DATA_DIR / "exp_sbsm_stack_res.npy")
        assert np.allclose(exp_res, sal_maps)

    def test_fill_prop(self) -> None:
        """
        Test that the `fill` property appropriately gets and sets the
        underlying `PerturbationOcclusion` instance fill instance attribute.
        """
        inst = SBSMStack((2, 2), (1, 1))
        assert inst._po.fill is None
        assert inst.fill is None
        inst.fill = 26
        assert inst.fill == 26
        assert inst._po.fill == 26
