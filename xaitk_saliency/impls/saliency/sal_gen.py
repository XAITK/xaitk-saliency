import copy
import io
from typing import Any, Dict, Generator

import cv2
from matplotlib import pyplot as plt
import numpy as np
import PIL.Image
from smqtk_dataprovider.impls.data_element.memory import DataMemoryElement
from smqtk_descriptors import DescriptorGenerator

from xaitk_saliency.interfaces.saliency import (
    ImageSaliencyAugmenter,
    ImageSaliencyMapGenerator,
    SaliencyBlackbox,
)


class LogitImageSaliencyMapGenerator(ImageSaliencyMapGenerator):
    """
    Interface for the method of generation of a saliency map given an image
    augmentation and blackbox algorithms.
    """
    def __init__(self, threshold: float = 0.2):
        self.thresh = threshold

    def get_config(self) -> Dict[str, Any]:
        return {
            'threshold': self.thresh,
        }

    def overlay_saliency_map(
        self,
        sa_map: np.ndarray,
        org_img: np.ndarray
    ) -> PIL.Image.Image:
        """
        Overlay the saliency map on top of original image

        :param sa_map: saliency map
        :param org_img: Original image

        :return: Overlaid image
        """
        plt.switch_backend('agg')
        height = float(org_img.shape[0])
        width = float(org_img.shape[1])
        fig = plt.figure(dpi=int(height))
        fig.set_size_inches((width / height), 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(org_img)
        ax.imshow(sa_map, cmap='jet', alpha=0.5)
        fig.canvas.draw()
        np_data = np.fromstring(fig.canvas.tostring_rgb(),
                                dtype=np.uint8, sep='')
        np_data = np_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        im = PIL.Image.fromarray(np_data)
        plt.close()
        return im

    def weighted_avg(self, scalar_vec: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """
        :param scalar_vec: Array of floats.
        :param masks: Array of image masks congruent in first-dim size with
            `scalar_vec`.
        :return: Weighted saliency mask in floating point.
        """
        masks = masks.reshape(-1, 224, 224, 1)
        cur_filters = copy.deepcopy(masks[:, :, :, 0])
        count = masks.shape[0] - np.sum(cur_filters, axis=0)

        for i in range(len(cur_filters)):
            cur_filters[i] = (1.0 - cur_filters[i]) * np.clip(scalar_vec[i],
                                                              a_min=0.0, a_max=None)
        res_sa = np.sum(cur_filters, axis=0) / count

        res_sa = np.clip(res_sa,
                         a_min=((np.max(res_sa)) * self.thresh),
                         a_max=None)
        return res_sa

    def generate(
        self,
        image_mat: np.ndarray,
        augmenter: ImageSaliencyAugmenter,
        descriptor_generator: DescriptorGenerator,
        blackbox: SaliencyBlackbox
    ) -> PIL.Image.Image:
        # [width, height] shape of the input image matrix
        org_wh = tuple(image_mat.shape[:2][::-1])

        image_mat_resized = cv2.resize(image_mat,
                                       (224, 224),
                                       interpolation=cv2.INTER_LINEAR)
        augs, masks = augmenter.augment(image_mat_resized)

        def iter_aug_img_data_elements() -> Generator[DataMemoryElement, None, None]:
            for a in augs:
                buff = io.BytesIO()
                a.save(buff, format="bmp")
                de = DataMemoryElement(buff.getvalue(),
                                       content_type='image/bmp')
                yield de

        scalar_vec = blackbox.transform(descriptor_generator.generate_elements(
            iter_aug_img_data_elements()
        ))
        final_sal_map = self.weighted_avg(scalar_vec, masks)
        final_sal_map = cv2.resize(
            final_sal_map, org_wh,
            interpolation=cv2.INTER_LINEAR
        )
        sal_map_ret = self.overlay_saliency_map(final_sal_map, image_mat)
        return sal_map_ret
