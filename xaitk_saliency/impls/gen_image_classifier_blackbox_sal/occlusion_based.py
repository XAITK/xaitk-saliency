from typing import Dict, Any, Type, TypeVar

import numpy as np
from smqtk_classifier import ClassifyImage
from smqtk_core.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict,
)

from xaitk_saliency.interfaces.perturb_image import PerturbImage
from xaitk_saliency.interfaces.gen_classifier_conf_sal import GenerateClassifierConfidenceSaliency
from xaitk_saliency.interfaces.gen_image_classifier_blackbox_sal import GenerateImageClassifierBlackboxSaliency
from xaitk_saliency.utils.masking import occlude_image_streaming


C = TypeVar("C", bound="PerturbationOcclusion")


class PerturbationOcclusion (GenerateImageClassifierBlackboxSaliency):
    """
    Generator composed of modular perturbation and occlusion-based algorithms.

    :param perturber: PerturbImage implementation instance for generating
        masks that will dictate occlusion.
    :param generator: Implementation instance for generating saliency masks
        given occlusion masks and classifier outputs.
    """

    def __init__(
        self,
        perturber: PerturbImage,
        generator: GenerateClassifierConfidenceSaliency,
        threads: int = 0
    ):
        self.perturber = perturber
        self.generator = generator
        self.threads = threads

    def _generate(
        self,
        ref_image: np.ndarray,
        blackbox: ClassifyImage
    ) -> np.ndarray:
        perturbation_masks = self.perturber(ref_image)
        class_list = blackbox.get_labels()
        # Input one thing so assume output of one thing.
        ref_conf_dict = list(blackbox.classify_images([ref_image]))[0]
        ref_conf_vec = np.asarray([ref_conf_dict[la] for la in class_list])
        pert_conf_mat = np.empty(
            (perturbation_masks.shape[0], ref_conf_vec.shape[0]),
            dtype=ref_conf_vec.dtype
        )
        pert_conf_it = blackbox.classify_images(
            occlude_image_streaming(
                ref_image, perturbation_masks,
                threads=self.threads
            )
        )
        for i, pc in enumerate(pert_conf_it):
            pert_conf_mat[i] = [pc[la] for la in class_list]

        # Compose classification results into a matrix for the generator
        # algorithm.
        return self.generator(
            ref_conf_vec,
            pert_conf_mat,
            perturbation_masks,
        )

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        cfg = super().get_default_config()
        cfg['perturber'] = make_default_config(PerturbImage.get_impls())
        cfg['generator'] = make_default_config(GenerateClassifierConfidenceSaliency.get_impls())
        return cfg

    @classmethod
    def from_config(
        cls: Type[C],
        config_dict: Dict,
        merge_default: bool = True
    ) -> C:
        config_dict = dict(config_dict)  # shallow-copy
        config_dict['perturber'] = from_config_dict(
            config_dict['perturber'],
            PerturbImage.get_impls()
        )
        config_dict['generator'] = from_config_dict(
            config_dict['generator'],
            GenerateClassifierConfidenceSaliency.get_impls()
        )
        return super().from_config(config_dict, merge_default=merge_default)

    def get_config(self) -> Dict[str, Any]:
        return {
            "perturber": to_config_dict(self.perturber),
            "generator": to_config_dict(self.generator),
            "threads": self.threads,
        }
