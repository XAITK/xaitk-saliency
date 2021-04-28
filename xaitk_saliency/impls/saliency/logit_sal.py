import io
import logging
from typing import Any, Dict, Iterable, Sequence, Tuple, Type

import PIL.Image
import numpy as np
from smqtk_core.configuration import (
    from_config_dict,
    to_config_dict,
)
from smqtk_dataprovider.impls.data_element.memory import DataMemoryElement
from smqtk_descriptors import DescriptorElement, DescriptorGenerator

from xaitk_saliency.interfaces.saliency import (
    ImageSaliencyAugmenter,
    SaliencyBlackbox,
)
from xaitk_saliency.utils.masking import generate_masked_images


LOG = logging.getLogger(__name__)


class LogitSaliencyBlackbox(SaliencyBlackbox):
    """
    This implementation yields some floating point scalar value
    for a given masked base image descriptor element that signifies the
    difference between the confidence value of the query image and masked
    image descriptors, used by class implementations of
    'ImageSaliencyMapGenerator'.
    """

    def __init__(
        self,
        pos_descriptors: Sequence[DescriptorElement],
        neg_descriptors: Sequence[DescriptorElement],
        rel_index: Any,  # "RelevancyIndex"; TODO: switch to RankRelevancy
        T_descr: DescriptorElement
    ):
        """
        :param pos_descriptors: is a set of positive descriptors.
        :param neg_descriptors: is a set of negative descriptors.
        :param rel_index:
            Plugin implementation of the algorithms used to generate relevance
            index used to rank images.
        :param T_descr:
            Base image feature descriptor
        """
        self.pos_de = pos_descriptors
        self.neg_de = neg_descriptors
        self.ADJs = (self.pos_de, self.neg_de)
        self.rel_index = rel_index
        self.T_descr = T_descr

    @classmethod
    def from_iqr_session(
        cls: Type["LogitSaliencyBlackbox"],
        iqrs: Any,  # "IqrSession"
        descr_generator: DescriptorGenerator,
        base_image: PIL.Image.Image
    ) -> "LogitSaliencyBlackbox":
        assert iqrs
        pos = list(iqrs.positive_descriptors | iqrs.external_positive_descriptors)
        neg = list(iqrs.negative_descriptors |
                   iqrs.external_negative_descriptors)
        # TODO: This requires the relevancy package breakout for the types used
        #       here.
        rel_index: Any = from_config_dict(
            to_config_dict(iqrs.rel_index),
            []  # RelevancyIndex.get_impls()
        )
        buff = io.BytesIO()
        base_image.save(buff, format="bmp")
        de = DataMemoryElement(buff.getvalue(),
                               content_type='image/bmp')
        T_descr = descr_generator.generate_one_element(de)
        return LogitSaliencyBlackbox(pos, neg, rel_index, T_descr)

    def get_config(self) -> Dict[str, Any]:
        return {
            'pos_descriptors': self.pos_de,
            'neg_descriptors': self.neg_de,
            'rel_index': self.rel_index,
            'T_descr': self.T_descr,
        }

    def transform(self, descriptors: Iterable[DescriptorElement]) -> np.ndarray:
        descriptors_list = list(descriptors)
        rel_train_set = [single_descr for single_descr in descriptors_list]
        rel_train_set.append(self.T_descr)
        self.rel_index.build_index(rel_train_set)
        RI_scores = self.rel_index.rank(*self.ADJs)
        diff = np.ones(len(descriptors_list))
        base_RI = RI_scores[rel_train_set[-1]]
        for i in range(len(descriptors_list)):
            diff[i] = RI_scores[rel_train_set[i]] - base_RI
        return diff


class LogitImageSaliencyAugmenter(ImageSaliencyAugmenter):
    """
    Robust Augmenter that yields a number of augmentations
    of an input image, as well as preserved-area masks
    used in saliency map generation.

    :param window_size: the block window size (with value 0, other areas
        with value 1)
    :param stride: the sliding step
    """

    def __init__(self, window_size: int = 50, stride: int = 20):
        self.window_size = window_size
        self.stride = stride
        self.masks = self.generate_block_masks(
            window_size=window_size,
            stride=stride
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            'window_size': self.window_size,
            'stride': self.stride,
        }

    def generate_block_masks(
        self,
        window_size: int,
        stride: int,
        image_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        Generates sliding window type binary masks used in augment() to
        mask an image. The Images are resized to 224x224 to
        enable re-use of masks.

        :param window_size: the block window size (with value 0, other areas
            with value 1)
        :param stride: the sliding step
        :param image_size: the mask size which should be the same to the image
            size

        :return: the sliding window style masks
        """
        rows = np.arange(0 + stride - window_size, image_size[0], stride)
        cols = np.arange(0 + stride - window_size, image_size[1], stride)

        mask_num = len(rows) * len(cols)
        LOG.debug('mask_num: {}'.format(mask_num))
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

    def augment(
        self,
        image_mat: np.ndarray
    ) -> Tuple[Sequence[PIL.Image.Image], np.ndarray]:
        img = PIL.Image.fromarray(image_mat)
        masked_images = generate_masked_images(self.masks, img)
        return masked_images, self.masks
