import io
from typing import Any, Dict, Generator, Sequence, Tuple

import numpy as np
import PIL.Image
from skimage.transform import resize
from smqtk_dataprovider.impls.data_element.memory import DataMemoryElement
from smqtk_descriptors import DescriptorGenerator

from xaitk_saliency.interfaces.saliency import (
    ImageSaliencyAugmenter,
    ImageSaliencyMapGenerator,
    SaliencyBlackbox,
)


class RISEAugmenter (ImageSaliencyAugmenter):

    def __init__(self, N: int, s: int, p1: float, input_size: Tuple[int, int]):
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
        self.N = N
        self.s = s
        self.p1 = p1
        # Size of each grid cell after upsampling
        cell_size = np.ceil(np.array(input_size) / s)
        # Upscale factor
        up_size = (s + 1) * cell_size

        # Generate a set of random grids of small resolution
        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        masks = np.empty((N, *input_size))

        for i in range(N):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and random cropping
            masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                    anti_aliasing=False)[x:x + input_size[0], y:y + input_size[1]]
        # Reshape brings this to: (N, input_size[0], input_size[1], 1)
        self.masks = masks.reshape(-1, *input_size, 1)
        self.input_size = input_size

    def get_config(self) -> Dict[str, Any]:
        return {
            'N': self.N,
            's': self.s,
            'p1': self.p1,
            'input_size': self.input_size,
        }

    def augment(
        self,
        image_mat: np.ndarray
    ) -> Tuple[Sequence[PIL.Image.Image], np.ndarray]:
        # If image is grayscale
        if len(image_mat.shape) == 2:
            image_mat = np.expand_dims(image_mat, 2).repeat(3, axis=2)
        return self.masks * image_mat, self.masks


class RISEGenerator (ImageSaliencyMapGenerator):

    def __init__(self, input_size: Tuple[int, int]):
        """
        Interface for randomized input sampling based explanations for blackbox models
        https://arxiv.org/abs/1806.07421
        """
        self.org_hw = input_size

    def get_config(self) -> Dict[str, Any]:
        return {
            'input_size': self.org_hw
        }

    def generate(
        self,
        image_mat: np.ndarray,
        augmenter: ImageSaliencyAugmenter,
        descriptor_generator: DescriptorGenerator,
        blackbox: SaliencyBlackbox
    ) -> PIL.Image.Image:
        resized_img = resize(image_mat, self.org_hw, order=1)
        masked_images, masks = augmenter.augment(resized_img)

        def iter_aug_img_data_elements() -> Generator[DataMemoryElement, None, None]:
            for a in masked_images:
                buff = io.BytesIO()
                a.save(buff, format="bmp")
                de = DataMemoryElement(buff.getvalue(),
                                       content_type='image/bmp')
                yield de

        scores = blackbox.transform(descriptor_generator.generate_elements(
            iter_aug_img_data_elements()
        ))

        # Compute a weighted average of masks w.r.t. the scores
        saliency_map: np.ndarray = np.average(masks, axis=0, weights=scores)
        saliency_map = np.squeeze(saliency_map)
        # Normalize
        saliency_map /= masks.mean(axis=0)
        # Resize back to the original image shape
        saliency_map = resize(saliency_map, image_mat.shape, order=1)

        # At this point the saliency map will be in some range [a, b], 0 <= a <= b <= 1.
        # The absolute values characterize the average score of the masked image and
        # therefore have some important information. However, for visualization purposes,
        # the saliency map can be rescaled to [0, 1].
        # saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

        # TODO: This is not returning currently a PIL Image.
        return saliency_map
