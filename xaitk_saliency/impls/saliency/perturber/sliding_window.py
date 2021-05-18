from typing import Tuple, List

from xaitk_saliency import PerturbImage
from xaitk_saliency.utils.masking import generate_masked_images

import numpy as np
import PIL.Image


class SlidingWindow(PerturbImage):
    """
    Implementation of a sliding window based perturbation taking a reference
    image and generating some number perturbations of the image along with
    paired mask matrices indicating where perturbations have occurred and
    to what amount.

    The resolution of perturbation is dependant upon the window size and
    stride of the sliding window.
    """

    def __init__(self,
                 window_size: int,
                 stride: int) -> None:
        self.window_size = window_size
        self.stride = stride

    def get_config(self) -> dict:
        return {"window_size": self.window_size,
                "stride": self.stride}

    def generate_block_masks(
        self,
        window_size: int,
        stride: int,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Generates sliding window type binary masks used in augment() to
        mask an image.
        :param window_size: the block window size (with value 0, other areas
            with value 1)
        :param stride: the sliding step
        :param image_size: the mask size which should be the same to the image size

        :return: the sliding window style masks with values ranging between [0, 1].
        """
        rows = np.arange(0 + stride - window_size, image_size[0], stride)
        cols = np.arange(0 + stride - window_size, image_size[1], stride)

        mask_num = len(rows) * len(cols)
        masks = np.ones((mask_num, image_size[0], image_size[1]),
                        dtype=np.uint8)
        i = 0
        for r in rows:
            for c in cols:
                if r < 0:
                    r1 = 0
                else:
                    r1 = r
                if r + window_size > image_size[0]:
                    r2 = image_size[0]
                else:
                    r2 = r + window_size
                if c < 0:
                    c1 = 0
                else:
                    c1 = c
                if c + window_size > image_size[1]:
                    c2 = image_size[1]
                else:
                    c2 = c + window_size
                masks[i, r1:r2, c1:c2] = 0
                i += 1
        mask_shape = [-1] + [*image_size] + [1]
        masks = masks.reshape(mask_shape)
        return masks

    def perturb(
        self,
        ref_image: PIL.Image.Image
    ) -> Tuple[List[PIL.Image.Image], np.ndarray]:

        img_dims = ref_image.size
        self.masks = self.generate_block_masks(self.window_size,
                                               self.stride, img_dims)
        masked_images = generate_masked_images(self.masks, ref_image)
        return masked_images, self.masks
