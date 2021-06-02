from xaitk_saliency.interfaces.perturb_image import PerturbImage
import PIL.Image
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from skimage.transform import resize


class RISEPertubation (PerturbImage):
    def _generate_masks(self, input_size: Tuple[int, int]) -> np.ndarray:
        """
        Randomly crop, upscale, and interpolate binary masks to generate a set
        of random masks to apply to the image.
        :param (int, int) input_size:
            Size of the model's input as (rows, cols). Binary masks are upsampled to
            this resolution to be applied to (multiplied with) the input later.
            E.g. (224, 224).
        """
        masks = np.empty((self.N, *input_size))
        cell_size = np.ceil(np.array(input_size) / self.s)
        # Upscale factor
        up_size = (self.s + 1) * cell_size
        for i in range(self.N):
            # Random shifts
            x = self.rng.integers(0, cell_size[0])
            y = self.rng.integers(0, cell_size[1])
            # Linear upsampling and random cropping
            masks[i, :, :] = resize(self.grid[i], up_size, order=1, mode='reflect',
                                    anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]
        # Reshape brings this to: (N, input_size[0], input_size[1], 1)
        return masks

    def __init__(
        self,
        N: int,
        s: int,
        p1: float,
        seed: Optional[float] = None
    ):
        """
        Generate a set of random binary masks
        :param int N:
            Number of random masks used in the algorithm. E.g. 1000.
        :param int s:
            Spatial resolution of the small masking grid. E.g. 8.
        :param float p1:
            Probability of the grid cell being set to 1 (otherwise 0). E.g. 0.5.
            Assumes square grid.
        :param Optional[float] seed:
            A seed to pass into the constructed random number generator to allow
            for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self.N = N
        self.s = s
        self.p1 = p1
        self.seed = seed

        # Generate a set of random grids of small resolution
        grid = self.rng.random((N, s, s)) < p1
        grid = grid.astype('float32')

        self.grid = grid

    def perturb(
        self,
        ref_image: PIL.Image.Image,
    ) -> Tuple[List[PIL.Image.Image], np.ndarray]:
        ref_mat = np.asarray(ref_image)
        input_size = (ref_image.height, ref_image.width)
        masks = self._generate_masks(input_size=input_size)
        mask_lhs = masks
        if ref_mat.ndim > 2:
            mask_lhs = masks.reshape(-1, *input_size, 1)
        masked_images = (mask_lhs * ref_mat).astype(ref_mat.dtype)
        masked_images_pil = [
            PIL.Image.fromarray(
                m,
                mode=ref_image.mode
            )
            for m in masked_images
        ]
        return masked_images_pil, masks

    def get_config(self) -> Dict[str, Any]:
        return {
            "N": self.N,
            "s": self.s,
            "p1": self.p1,
            "seed": self.seed
        }
