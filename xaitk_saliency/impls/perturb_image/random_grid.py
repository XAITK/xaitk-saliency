import numpy as np
from skimage.transform import resize
from typing import Optional, Dict, Any, Tuple

from xaitk_saliency import PerturbImage
from smqtk_descriptors.utils import parallel_map


class RandomGrid (PerturbImage):
    """
    Generate masks using a random grid of set cell size. If the chosen cell
    size does not divide an image evenly, then the grid is over-sized and
    the resulting mask is centered and cropped. Each mask is also shifted
    randomly by a maximum of half the cell size in both x and y.

    This method is based on RISE (http://bmvc2018.org/contents/papers/1064.pdf)
    but aims to address the changing cell size, given images of different
    sizes, aspect of that implementation. This method keeps cell size constant
    and instead adjusts the overall grid size for different sized images.

    :param n: Number of masks to generate.
    :param s: Dimensions of the grid cells in pixels. E.g. (3, 4) would use
        a grid of 3x4 pixel cells.
    :param p1: Probability of a grid cell being set to 1 (not occluded).
        This should be a float value in the [0, 1] range.
    :param seed: A seed to use for the random number generator, allowing
        for masks to be reproduced.
    :param threads: Number of threads to use when generating masks. If this
        is <=0 or None, no threading is used and processing is performed
        in-line serially.
    """

    def __init__(
        self,
        n: int,
        s: Tuple[int, int],
        p1: float,
        seed: Optional[int] = None,
        threads: Optional[int] = None,
    ):
        self.n = n
        self.s = s
        self.p1 = p1
        self.seed = seed
        self.threads = threads

    def perturb(
        self,
        ref_img: np.ndarray,
    ) -> np.ndarray:

        num_masks = self.n
        s = np.array(self.s)
        p1 = self.p1
        threads = self.threads
        grid_rng = np.random.default_rng(self.seed)
        shift_rng = np.random.default_rng(self.seed)

        img_shape = np.array(ref_img.shape[:2])

        # Pad grid by one cell in both dimensions to allow for shifting
        grid_size = (img_shape // s) + 1

        # Additional padding if cell size does not divide the image evenly
        # This is required to allow shifting at full range
        grid_size += img_shape % s != 0

        grid = grid_rng.random((num_masks, *grid_size)) < p1
        grid = grid.astype(np.float32)

        masks = np.empty((num_masks, *img_shape), dtype=grid.dtype)

        up_size = grid_size * s

        # Offsets to center mask on grid
        center_y, center_x = (up_size - img_shape) // 2

        # Maximum shifting offsets
        max_shift_y, max_shift_x = s // 2

        # Expand index for efficiency
        img_h, img_w = img_shape

        def work_func(_i: int) -> np.ndarray:
            if max_shift_x != 0:
                shift_x = shift_rng.integers(-max_shift_x, max_shift_x)
            else:
                shift_x = 0

            if max_shift_y != 0:
                shift_y = shift_rng.integers(-max_shift_y, max_shift_y)
            else:
                shift_y = 0

            # Total offset: offset to center + random shift
            offset_x = center_x + shift_x
            offset_y = center_y + shift_y

            mask = resize(grid[_i], up_size, order=1)
            mask = mask[offset_y:offset_y + img_h, offset_x:offset_x + img_w]

            return mask

        if threads is None or threads < 1:
            for i in range(num_masks):
                masks[i, ...] = work_func(i)
        else:
            for i, m in enumerate(parallel_map(
                work_func, range(num_masks),
                cores=threads,
                use_multiprocessing=False
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
