import PIL.Image
from typing import Optional, Tuple, Dict, Any, Generator
import numpy as np
from skimage.transform import resize
from smqtk_descriptors.utils import parallel_map

from xaitk_saliency.interfaces.perturb_image import PerturbImage


class RISEPertubation (PerturbImage):
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
            Probability of the grid cell being set to 1 (otherwise 0). E.g. 0.5.
        :param seed:
            A seed to pass into the constructed random number generator to allow
            for reproducibility
        :param threads: The number of threads to utilize when generating images and
            masks. If this is <=0 or None, no threading is used and processing
            is performed in-line serially.
        """
        self.n = n
        self.s = s
        self.p1 = p1
        self.seed = seed
        self.threads = threads

        # Generate a set of random grids of small resolution
        grid = np.random.default_rng(seed).random((n, s, s)) < p1
        grid = grid.astype('float32')

        self.grid = grid

    def perturb(
        self,
        ref_image: PIL.Image.Image
    ) -> Generator[Tuple[PIL.Image.Image, np.ndarray], None, None]:
        image_from_array = PIL.Image.fromarray
        ref_mat = np.asarray(ref_image)
        ref_mode = ref_image.mode
        input_size = (ref_image.height, ref_image.width)
        mul_slice: Tuple = (...,)
        if ref_mat.ndim > 2:
            mul_slice = (..., None)  # add channel axis for multiplication

        grid = self.grid
        num_masks = self.n
        s = self.s

        shift_rng = np.random.default_rng(self.seed)
        cell_size = np.ceil(np.array(input_size) / s)
        up_size = (s + 1) * cell_size

        def work_func(i_: int) -> Tuple[PIL.Image.Image, np.ndarray]:
            # Random shifts
            x = shift_rng.integers(0, cell_size[0])
            y = shift_rng.integers(0, cell_size[1])
            mask = resize(
                grid[i_], up_size, order=1, mode='reflect', anti_aliasing=False
            )[x:x + input_size[0], y:y + input_size[1]]
            # probably the majority cost? TODO: Test performance cost
            img_m = (mask[mul_slice] * ref_mat).astype(ref_mat.dtype)
            img_p = image_from_array(img_m, mode=ref_mode)
            return img_p, mask

        threads = self.threads
        if threads is None or threads <= 0:
            for i in range(num_masks):
                yield work_func(i)
        else:
            for img, m in parallel_map(
                work_func, range(num_masks),
                cores=self.threads,
                use_multiprocessing=False,
            ):
                yield img, m

    def get_config(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "s": self.s,
            "p1": self.p1,
            "seed": self.seed,
            "threads": self.threads,
        }
