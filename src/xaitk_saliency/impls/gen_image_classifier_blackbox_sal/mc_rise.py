"""Implementation of MC-RISE saliency stack"""

from __future__ import annotations

import itertools
from collections.abc import Generator, Iterable, Sequence
from typing import Any

import numpy as np
from smqtk_classifier.interfaces.classify_image import ClassifyImage
from smqtk_descriptors.utils.parallel import parallel_map
from typing_extensions import override

from xaitk_saliency.impls.gen_classifier_conf_sal.mc_rise_scoring import MCRISEScoring
from xaitk_saliency.impls.perturb_image.mc_rise import MCRISEGrid
from xaitk_saliency.interfaces.gen_image_classifier_blackbox_sal import GenerateImageClassifierBlackboxSaliency


class MCRISEStack(GenerateImageClassifierBlackboxSaliency):
    """
    Encapsulation of the perturbation-occlusion method using specifically the
    MC-RISE implementations of the component algorithms.

    This more specifically encapsulates the MC-RISE method as presented
    in their paper and code. See references in the :class:`MCRISEGrid`
    and :class:`MCRISEScoring` documentation.

    This implementation shares the `p1` probability and 'k' number colors
    with the internal `MCRISEScoring` instance use, to make use of the
    debiasing described in the MC-RISE paper. Debiasing is always on.
    """

    def __init__(
        self,
        n: int,
        s: int,
        p1: float,
        fill_colors: Sequence[Sequence[int]],
        seed: int | None,
        threads: int = 0,
    ) -> None:
        """
        :param n: int
            Number of random masks used in the algorithm. E.g. 1000.
        :param s: int
            Spatial resolution of the small masking grid. E.g. 8.
            Assumes square grid.
        :param p1: float
            Probability of the grid cell being set to 1 (otherwise 0).
            This should be a float value in the [0, 1] range. E.g. 0.5.
        :param fill_colors: Sequence[Sequence[int]]
            The fill colors to be used when generating masks.
        :param seed: Optional[int]
            A seed to pass into the constructed random number generator to allow
            for reproducibility
        :param threads: int
            The number of threads to utilize when generating masks.
            If this is <=0 or None, no threading is used and processing
            is performed in-line serially.

        :raises: ValueError
            If no fill colors are provided.
        :raises: ValueError
            If provided fill colors have differing numbers of channels.
        """
        if len(fill_colors) == 0:
            raise ValueError("At least one fill color must be provided")
        for fill_color in fill_colors:
            if len(fill_color) != len(fill_colors[0]):
                raise ValueError("All fill colors must have the same number of channels")
        if seed is not None:
            self._perturber = MCRISEGrid(n=n, s=s, p1=p1, k=len(fill_colors), seed=seed, threads=threads)
        else:
            self._perturber = MCRISEGrid(n=n, s=s, p1=p1, k=len(fill_colors), threads=threads)
        self._generator = MCRISEScoring(k=len(fill_colors), p1=p1)
        self._threads = threads
        self._fill_colors = fill_colors

    @staticmethod
    def _work_func(ref_image: np.ndarray, i_: int, m: np.ndarray, f: np.ndarray) -> np.ndarray:
        s: tuple = (...,)
        if ref_image.ndim > 2:
            s = (..., None)  # add channel axis for multiplication

        # Just the [H x W] component.
        img_shape = ref_image.shape[:2]

        m_shape = m.shape
        if m_shape != img_shape:
            raise ValueError(
                f"Input mask (position {i_}) did not the shape of the input image: {m_shape} != {img_shape}",
            )
        img_m = np.empty_like(ref_image)

        np.add((m[s] * ref_image), f, out=img_m, casting="unsafe")

        return img_m

    @staticmethod
    def _occlude_image_streaming(
        *,
        ref_image: np.ndarray,
        masks: Iterable[np.ndarray],
        fill_values: Iterable[np.ndarray],
        threads: int | None,
    ) -> Generator[np.ndarray, None, None]:
        if threads is None or threads < 1:
            for i, (mask, fill) in enumerate(zip(masks, fill_values)):
                yield MCRISEStack._work_func(ref_image=ref_image, i_=i, m=mask, f=fill)
        else:
            ref_image = np.stack([ref_image for _ in fill_values], axis=0)
            # NOTE: Never pass by keyword to avoid the iterables being wrongly
            # zipped and assigned to kwargs.
            yield from parallel_map(
                MCRISEStack._work_func,
                ref_image,
                itertools.count(),
                masks,
                fill_values,
                cores=threads,
                use_multiprocessing=False,
            )

    @override
    def _generate(self, ref_image: np.ndarray, blackbox: ClassifyImage) -> np.ndarray:
        """
        Warning: this implementation returns a different shape than is typically expected by this interface.
        Instead of returning `[nClasses x H x W]`, `[kColors x nClasses x H x W] saliency maps will be returned.

        :param ref_image: np.ndarray
            Reference image over which visual saliency heatmaps will be generated.
        :param blackbox: ClassifyImage
            The black-box classifier handle to perform arbitrary operations on in order to deduce visual saliency.

        :return: np.ndarray
            A number of visual saliency heatmaps equivalent in number to the quantity of class labels output
            by the black-box classifier and configured number of colors.
        """
        k_perturbation_masks = self._perturber(ref_image)

        # Collapse k individual colored masks into n multi-color masks
        perturbation_masks = 1 - np.sum(1 - k_perturbation_masks, axis=0)
        fill_values = np.zeros((*k_perturbation_masks.shape[1:], len(self._fill_colors[0])))
        for mask_idx, k_masks in enumerate(np.swapaxes(k_perturbation_masks, 0, 1)):
            k_masks = 1 - np.repeat(k_masks[..., np.newaxis], len(self._fill_colors[0]), axis=3)
            for fill_idx, fill_value in enumerate(self._fill_colors):
                fill_values[mask_idx] += k_masks[fill_idx] * fill_value
        fill_values = np.clip(fill_values, 0, 255)

        class_list = blackbox.get_labels()
        # Input one thing so assume output of one thing.
        ref_conf_dict = list(blackbox.classify_images([ref_image]))[0]
        ref_conf_vec = np.asarray([ref_conf_dict[la] for la in class_list])
        pert_conf_mat = np.empty((perturbation_masks.shape[0], ref_conf_vec.shape[0]), dtype=ref_conf_vec.dtype)

        pert_conf_it = blackbox.classify_images(
            MCRISEStack._occlude_image_streaming(
                ref_image=ref_image,
                masks=perturbation_masks,
                fill_values=fill_values,
                threads=self._threads,
            ),
        )
        for i, pc in enumerate(pert_conf_it):
            pert_conf_mat[i] = [pc[la] for la in class_list]

        # Compose classification results into a matrix for the generator
        # algorithm.
        return self._generator(
            reference=ref_conf_vec,
            perturbed=pert_conf_mat,
            perturbed_masks=k_perturbation_masks,
        )

    @override
    def get_config(self) -> dict[str, Any]:
        # It turns out that our configuration here is nearly equivalent to that given
        # and retrieved from the MCRISEGrid implementation
        c = self._perturber.get_config()
        del c["k"]
        c["fill_colors"] = self._fill_colors
        return c
