"""Implementation of MC-RISE perturbation mask generation."""

from __future__ import annotations

__all__ = ["MCRISEGrid"]

from typing import Any

import numpy as np
from skimage.transform import resize
from smqtk_descriptors.utils.parallel import parallel_map
from typing_extensions import override

from xaitk_saliency.interfaces.perturb_image import PerturbImage


class MCRISEGrid(PerturbImage):
    """Generate MC-RISE masks based on Hatakeyama et al.

    https://openaccess.thecvf.com/content/ACCV2020/papers/Hatakeyama_Visualizing_Color-wise_Saliency_of_Black-Box_Image_Classification_Models_ACCV_2020_paper.pdf

    Example:
        >>> n_colors, n_masks, height, width = 2, 3, 24, 32
        >>> ref_image = np.zeros((height, width, 3), dtype=np.uint8)
        >>> perturber = MCRISEGrid(n=n_masks, s=8, p1=0.5, k=n_colors, seed=0)
        >>> masks = perturber(ref_image)
        >>> masks.shape == (n_colors, n_masks, height, width)
        True
    """

    def __init__(
        self,
        *,
        n: int,
        s: int,
        p1: float,
        k: int,
        seed: int | None = None,
        threads: int | None = 4,
    ) -> None:
        """Generate a set of random binary masks.

        :param n: int
            Number of random masks used in the algorithm. E.g. 1000.
        :param s: int
            Spatial resolution of the small masking grid. E.g. 8.
            Assumes square grid.
        :param p1: float
            Probability of the grid cell being set to 1 (otherwise 0).
            This should be a float value in the [0, 1] range. E.g. 0.5.
        :param k: int
            Number of colors to use.
        :param seed: Optional[int]
            A seed to pass into the constructed random number generator to allow
            for reproducibility
        :param threads: int
            The number of threads to utilize when generating masks.
            If this is <=0 or None, no threading is used and processing
            is performed in-line serially.

        :raises: ValueError
            If p1 not in [0, 1].
        :raises: ValueError
            If k < 1.
        """
        if p1 < 0 or p1 > 1:
            raise ValueError(f"Input p1 value of {p1} is not within the expected [0,1] range.")

        if k <= 0:
            raise ValueError(f"Input k value of {k} is not within the expected >0 range.")

        self.n = n
        self.s = s
        self.p1 = p1
        self.k = k
        self.seed = seed
        self.threads = threads

        # Generate a set of random grids of small resolution
        rng = np.random.default_rng(seed)
        grid: np.ndarray = rng.random((n, s, s)) < p1

        indiv_color_masks = np.empty((self.k, self.n, self.s, self.s))
        for g_idx, g in enumerate(grid):
            g_shape = g.shape
            # Randomly choose fill colors for each pixel
            fill_mask = rng.integers(low=0, high=k, size=g_shape)
            for fill_idx in range(self.k):
                # Keep the masked pixels where the current idx (color) was selected
                indiv_color_mask = np.where(~g, np.where(fill_mask == fill_idx, False, True), True)
                indiv_color_masks[fill_idx][g_idx] = indiv_color_mask

        self.grid = indiv_color_masks
        self.grid.astype("float32")

    @override
    def perturb(self, ref_image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return MC-RISE masks, with one mask set per color.

        Warning: this implementation returns a different shape than typically expected by this interface.
        Instead of `[nMasks x Height x Width]`, masks of shape `[kColors x nMasks x Height x Width]`
        are returned.

        :param ref_image: np.ndarray
            Reference image to generate perturbations from.

        :return: np.ndarray
            Mask matrix with shape `[kColors x nMasks x Height x Width]`.
        """
        input_size = np.shape(ref_image)[:2]
        num_colors = self.k
        num_masks = self.n
        grid = self.grid
        s = self.s
        shift_rng = np.random.default_rng(self.seed)
        # Shape format: [H x W], Inherits from `input_size`
        cell_size = np.ceil(np.array(input_size) / s)
        up_size = (s + 1) * cell_size

        masks = np.empty((num_colors, num_masks, *input_size), dtype=grid.dtype)

        def work_func(i_: int) -> np.ndarray:
            return self._mask_for_index(
                i_,
                grid=grid,
                input_size=input_size,
                up_size=up_size,
                shift_rng=shift_rng,
            )

        threads = self.threads
        if threads is None or threads < 1:
            for i in range(num_masks):
                masks[:, i, ...] = work_func(i)
        else:
            for i, m in enumerate(
                parallel_map(  # noqa: FKA100 - parallel_map takes *sequences, which cannot be passed by keyword
                    work_func,
                    range(num_masks),
                    cores=threads,
                    use_multiprocessing=False,
                ),
            ):
                masks[:, i, ...] = m

        return masks

    def _mask_for_index(
        self,
        index: int,
        *,
        grid: np.ndarray[Any, Any],
        input_size: tuple[int, ...],
        up_size: np.ndarray[Any, Any],
        shift_rng: np.random.Generator,
    ) -> np.ndarray[Any, Any]:
        """Resize one MC-RISE grid across all colors and crop a random shift."""
        cell_h, cell_w = np.ceil(np.array(input_size) / self.s)[:2]
        input_h, input_w = input_size[:2]
        y = shift_rng.integers(low=0, high=cell_h)
        x = shift_rng.integers(low=0, high=cell_w)
        k_masks = np.empty((self.k, input_h, input_w))
        for color_idx in range(self.k):
            k_masks[color_idx] = resize(
                image=grid[color_idx, index, ...],
                output_shape=up_size,
                order=1,
                mode="reflect",
                anti_aliasing=False,
            )[y : y + input_h, x : x + input_w]
        return k_masks

    @override
    def get_config(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "s": self.s,
            "p1": self.p1,
            "k": self.k,
            "seed": self.seed,
            "threads": self.threads,
        }
