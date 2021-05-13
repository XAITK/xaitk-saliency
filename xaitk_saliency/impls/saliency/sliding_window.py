from typing import Iterable, Tuple, List

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
                window_size,
                stride):
        self.window_size = window_size
        self.stride = stride
    
    def get_config(self):
        return {"window_size": self.window_size,
               "stride": self.stride}

    def generate_block_masks(
        self,
        window_size: int,
        stride: int,
        image_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        Generates sliding window type binary masks used in augment() to
        mask an image. 
        :param window_size: the block window size (with value 0, other areas
            with value 1)
        :param stride: the sliding step
        :param image_size: the mask size which should be the same to the image
            size
            
        :return: the sliding window style masks with values ranging between [0, 1].
        """
        rows = np.arange(0 + stride - window_size, image_size[0], stride)
        cols = np.arange(0 + stride - window_size, image_size[1], stride)

        mask_num = len(rows) * len(cols)
        masks = np.ones((mask_num, image_size[0], image_size[1]),
                        dtype=np.float64)
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
        """
        Transform an input reference image into a number of perturbed
        variations along with mask matrices indicating the perturbed regions.

        Output images should have the same shape as the input reference image,
        including channels.

        Output mask matrices should be congruent in length to the number of
        perturbed images output, as well as share the same height and width
        dimensions.
        These masks should indicate the regions in the corresponding perturbed
        image that has been modified.
        Values should be in the [0, 1] range, where a value closer to 1.0
        indicate areas of the image that are *unperturbed*.
        Note that output mask matrices *may be* of a floating-point type in
        order to allow for fractional perturbation. 
        
        The resolution of masking is controlled by the window_size and stride
        of siding window.

        :param ref_image:
            Reference image to generate perturbations from.
        :return: Tuple of perturbed images and the window masks detailing perturbation
            areas.
        """
        img_dims = ref_image.size
        self.masks = self.generate_block_masks(self.window_size, \
                                               self.stride, img_dims)
        masked_images = generate_masked_images(self.masks, ref_image)
        return masked_images, self.masks
        
    def __call__(
        self,
        ref_image: PIL.Image.Image
    ) -> Tuple[List[PIL.Image.Image], np.ndarray]:
        """
        Alias for :meth:`.PerturbImage.perturb`.
        """
        return self.perturb(ref_image)