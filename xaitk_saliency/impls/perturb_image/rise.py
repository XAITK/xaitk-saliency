from xaitk_saliency.interfaces.perturb_image import PerturbImage
import PIL.Image
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from skimage.transform import resize


class RISEPertubation (PerturbImage):

    def __init__(
        self,
        N: int,
        s: int,
        p1: float,
        input_size: Tuple[int, int],
        seed: Optional[float] = None
    ):
        """
        Generate a set of random masks to apply to the image.
        :param int N:
            Number of random masks used in the algorithm. E.g. 1000.
        :param int s:
            Spatial resolution of the small masking grid. E.g. 8.
        :param float p1:
            Probability of the grid cell being set to 1 (otherwise 0). E.g. 0.5.
            Assumes square grid.
        :param (int, int) input_size:
            Size of the model's input. Smaller masks are upsampled to this resolution
            to be applied to (multiplied with) the input later. E.g. (224, 224)
        """
        self.rng = np.random.default_rng(seed)
        self.N = N
        self.s = s
        self.p1 = p1
        self.seed = seed
        # Size of each grid cell after upsampling
        cell_size = np.ceil(np.array(input_size) / s)
        # Upscale factor
        up_size = (s + 1) * cell_size

        # Generate a set of random grids of small resolution
        grid = self.rng.random((N, s, s)) < p1
        grid = grid.astype('float32')

        masks = np.empty((N, *input_size))

        for i in range(N):
            # Random shifts
            x = self.rng.integers(0, cell_size[0])
            y = self.rng.integers(0, cell_size[1])
            # Linear upsampling and random cropping
            masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                    anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]
        # Reshape brings this to: (N, input_size[0], input_size[1], 1)
        self.masks = masks
        self.input_size = input_size

    def perturb(
        self,
        ref_image: PIL.Image.Image,
    ) -> Tuple[List[PIL.Image.Image], np.ndarray]:
        ref_mat = np.asarray(ref_image)
        mask_lhs = self.masks
        if ref_mat.ndim > 2:
            mask_lhs = self.masks.reshape(-1, *self.input_size, 1)
        masked_images = (mask_lhs * ref_mat).astype(ref_mat.dtype)
        masked_images_pil = [
            PIL.Image.fromarray(
                m,
                mode=ref_image.mode
            )
            for m in masked_images
        ]
        return masked_images_pil, self.masks

    def get_config(self) -> Dict[str, Any]:
        return {
            "N": self.N,
            "s": self.s,
            "p1": self.p1,
            "input_size": self.input_size,
            "seed": self.seed
        }
