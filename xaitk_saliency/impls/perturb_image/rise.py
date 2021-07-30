"""
RISE MIT License (https://github.com/eclique/RISE/blob/master/LICENSE):

    MIT License

    Copyright (c) 2018 Vitali Petsiuk

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""
from typing import Optional, Dict, Any
import numpy as np
from skimage.transform import resize
from smqtk_descriptors.utils import parallel_map

from xaitk_saliency.interfaces.perturb_image import PerturbImage


class RISEGrid (PerturbImage):
    """
    Based on Petsiuk et. al: http://bmvc2018.org/contents/papers/1064.pdf

    Implementation is borrowed from the original authors:
    https://github.com/eclique/RISE/blob/master/explanations.py
    """

    def __init__(
        self,
        n: int,
        s: int,
        p1: float,
        seed: Optional[int] = None,
        threads: Optional[int] = 4,
    ):
        """
        Generate a set of random binary masks
        :param n:
            Number of random masks used in the algorithm. E.g. 1000.
        :param s:
            Spatial resolution of the small masking grid. E.g. 8.
            Assumes square grid.
        :param p1:
            Probability of the grid cell being set to 1 (otherwise 0).
            This should be a float value in the [0, 1] range. E.g. 0.5.
        :param seed:
            A seed to pass into the constructed random number generator to allow
            for reproducibility
        :param threads: The number of threads to utilize when generating masks.
            If this is <=0 or None, no threading is used and processing
            is performed in-line serially.
        """
        if p1 < 0 or p1 > 1:
            raise ValueError(
                f"Input p1 value of {p1} is not within the expected [0,1] "
                f"range."
            )

        self.n = n
        self.s = s
        self.p1 = p1
        self.seed = seed
        self.threads = threads

        # Generate a set of random grids of small resolution
        grid: np.ndarray = np.random.default_rng(seed).random((n, s, s)) < p1
        grid = grid.astype('float32')

        self.grid = grid

    def perturb(
        self,
        ref_image: np.ndarray
    ) -> np.ndarray:
        input_size = np.shape(ref_image)[:2]
        num_masks = self.n
        grid = self.grid
        s = self.s
        shift_rng = np.random.default_rng(self.seed)
        # Shape format: [H x W], Inherits from `input_size`
        cell_size = np.ceil(np.array(input_size) / s)
        up_size = (s + 1) * cell_size

        masks = np.empty((num_masks, *input_size), dtype=grid.dtype)

        # Expanding index accesses for repetition efficiency.
        cell_h, cell_w = cell_size[:2]
        input_h, input_w = input_size[:2]

        def work_func(i_: int) -> np.ndarray:
            # Random shifts
            y = shift_rng.integers(0, cell_h)
            x = shift_rng.integers(0, cell_w)
            mask = resize(
                grid[i_], up_size, order=1, mode='reflect', anti_aliasing=False
            )[y:y + input_h, x:x + input_w]
            return mask

        threads = self.threads
        if threads is None or threads < 1:
            for i in range(num_masks):
                masks[i, ...] = work_func(i)
        else:
            for i, m in enumerate(parallel_map(
                work_func, range(num_masks),
                cores=threads,
                use_multiprocessing=False,
            )):
                masks[i, ...] = m

        return masks

    def get_config(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "s": self.s,
            "p1": self.p1,
            "seed": self.seed,
            "threads": self.threads,
        }
