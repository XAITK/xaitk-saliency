import io
import logging
import os
from typing import Any, Dict, Iterable, Sequence, Tuple, Type

import numpy as np
import PIL.Image
from sklearn.metrics.pairwise import euclidean_distances
from smqtk_dataprovider.impls.data_element.memory import DataMemoryElement
from smqtk_descriptors import DescriptorElement, DescriptorGenerator
from tqdm import tqdm

from xaitk_saliency.interfaces.saliency import (
    ImageSaliencyAugmenter,
    SaliencyBlackbox,
)
from xaitk_saliency.utils.masking import generate_masked_images


LOG = logging.getLogger(__name__)


class SBSMSaliencyBlackbox(SaliencyBlackbox):
    """
    SBSM_SaliencyBlackbox function that yields some floating point
    scalar value for a given masked base image descriptor element that
    signifies the proximity between the query image and masked image
    descriptors, used by class implementations of 'ImageSaliencyMapGenerator'.

    :param query_f: Feature of query image.
    :param base_descr: Base image descriptor
    """

    def __init__(self, query_f: np.ndarray, base_descr: np.ndarray):
        self.query_f = query_f
        self.base_descr = base_descr

    @classmethod
    def from_iqr_session(
        cls: Type["SaliencyBlackbox"],
        iqrs: Any,  # "IqrSession"
        descr_generator: DescriptorGenerator,
        base_image: PIL.Image.Image
    ) -> "SaliencyBlackbox":
        assert iqrs
        if (len(iqrs.external_positive_descriptors)) != 1:
            raise ValueError("Saliency generation in class``{}`` "
                             "only supports one query sample, but"
                             "recieved more than 1 external positive."
                             .format(cls.__name__))
        # Implementation only ever considers the first query descriptor, so
        # we reflect such a selection here based on the previous incorrect
        # impl.
        # query_f = [ext_pos.vector() for ext_pos in \
        #    iqrs.external_positive_descriptors]
        query_f = list(iqrs.external_positive_descriptors)[0].vector()
        buff = io.BytesIO()
        base_image.save(buff, format="bmp")
        de = DataMemoryElement(buff.getvalue(),
                               content_type='image/bmp')
        base_descr = descr_generator.generate_one_element(de)
        base_descr_vec = base_descr.vector()
        assert base_descr_vec is not None  # we literally just generated it...
        return SBSMSaliencyBlackbox(query_f, base_descr_vec)

    def get_config(self) -> Dict[str, Any]:
        return {
            'query_f': self.query_f,
            'base_descr': self.base_descr,
        }

    def transform(self, descriptors: Iterable[DescriptorElement]) -> np.ndarray:
        org_dis = abs(euclidean_distances(self.query_f.reshape(1, -1),
                                          self.base_descr.reshape(1, -1)))
        descriptors_list = list(descriptors)
        vector_list = [d.vector() for d in descriptors_list]
        if None in vector_list:
            raise RuntimeError("One or more input descriptor elements did not "
                               "refer to an actual vector (some None-valued).")
        diff = np.ones(len(descriptors_list))
        query_f_reshaped = self.query_f[0].reshape(1, -1)
        for i in range(len(vector_list)):
            v = vector_list[i]
            assert v is not None
            diff[i] = max(abs(euclidean_distances(
                v.reshape(1, -1),
                query_f_reshaped
            )) - org_dis, 0)
        return diff


class SBSMImageSaliencyAugmenter(ImageSaliencyAugmenter):
    """
    Algorithm that yields a number of augmentations of an input image,
    as well as preserved-area masks, used for use in saliency map generation.
    For a more robust augmenter that can work for different image_sizes
    use implementation class 'Logit_ImageSaliencyAugmenter'.

    :param window_size: the block window size (with value 0, other areas
        with value 1)
    :param stride: the sliding step
    """

    def __init__(self, window_size: int = 20, stride: int = 4):
        self.window_size = window_size
        self.stride = stride
        self.masks = self.generate_block_masks(self.window_size, self.stride)

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
        # The augmenter only supports certain factors of window_size and stride
        # for a more robust augmenter use LogitImageSaliencyAugmenter
        if (image_size[0] - window_size) % stride != 0:
            raise ValueError("Change window size and stride to satisfy "
                             "condition:(image_height-window_size)%stride=0")
        if not os.path.isfile('block_mask_{}_{}.npy'.format(window_size, stride)):
            grid_num_r = ((image_size[0] - window_size) // stride) + 1
            grid_num_c = ((image_size[1] - window_size) // stride) + 1
            mask_num = grid_num_r * grid_num_c
            masks = np.ones((mask_num, image_size[0], image_size[1]), dtype=np.float32)
            i = 0
            for r in tqdm(np.arange(0, image_size[0] - window_size + 1, stride), total=grid_num_r,
                          desc="Generating rows"):
                for c in np.arange(0, image_size[1] - window_size + 1, stride):
                    masks[i, r:r + window_size, c:c + window_size] = 0.0
                    i += 1
            masks = masks.reshape(-1, *image_size)
            masks.tofile('block_mask_{}_{}.npy'.format(window_size, stride))
        else:
            LOG.debug("Loading masks from file block_mask_{}_{}.npy".format(window_size, stride))
            masks = np.fromfile('block_mask_{}_{}.npy'.format(window_size, stride),
                                dtype=np.float32).reshape(-1, *image_size)
        return masks

    def augment(
        self,
        image_mat: np.ndarray
    ) -> Tuple[Sequence[PIL.Image.Image], np.ndarray]:
        """
        Takes in an image matrix and returns its augmented version
        :param numpy.ndarray image_mat:
            Image matrix to be augmented.
        :return: A PIL.Image of augmented image matrices as well as numpy array of masks
            that indicate the regions in the augmented images that are
            unmodified with respect to the input image (preserved regions).
            Returned augmented images should be in the dimension format
            [index, height, width [,channel]] with the the same data type as
            the input image matrix.
            Returned masks should be in the dimension format
            [index, height, width,channel] with the boolean data type.
        :rtype: (PIL.Image, numpy.ndarray)
        """
        img = PIL.Image.fromarray(image_mat)
        masked_images = generate_masked_images(self.masks, img)
        return masked_images, self.masks
