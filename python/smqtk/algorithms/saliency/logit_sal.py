import numpy as np
from tqdm import tqdm
import copy
from PIL import Image
import os
from sklearn.metrics.pairwise import euclidean_distances
import six
from smqtk.algorithms.relevancy_index import get_relevancy_index_impls
from smqtk.algorithms import get_descriptor_generator_impls
from smqtk.algorithms import SmqtkAlgorithm
from smqtk.algorithms.saliency import SaliencyBlackbox,ImageSaliencyAugmenter
from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.utils import plugin
    
__author__ = "bhavan.vasu@kitware.com"


class Logit_SaliencyBlackbox (SaliencyBlackbox):
    def __init__(self, ADJs, rel_index, descr_gen, base_image):
        """
        Blackbox function that produces some floating point scalar value for a
        given descriptor element.
        """
    
        self.ADJs = ADJs
    
        self.rel_index = rel_index
    
        self.descr_gen = descr_gen
     
        self.base_image = base_image
    
    @classmethod
    def is_usable(cls):
        """
        Check whether this implementation is available for use.
        Required valid presence of svm and svmutil modules
        :return:
            Boolean determination of whether this implementation is usable.
        :rtype: bool
        """
        
        return SaliencyBlackbox and euclidean_distances

        
    @classmethod
    def from_iqr_session(cls, iqrs, descr_gen, base_image): 
        """
        Create an ``SaliencyBlackbox`` instance from an
        :class:`smqtk.iqr.IqrSession` instance.
        Not all implementations of ``SaliencyBlackbox`` implement
        this method and may raise a ``NotImplementedError`` exception.
        :raises NotImplementedError:
            Construction from a ``IqrSession`` is not defined for this
            implementation.
        :return: A new instance of of a class implementing the
            ``SaliencyBlackbox`` class.
        :rtype: SaliencyBlackbox
        """
        try:
            assert iqrs

            assert descr_gen.is_usable

            pos = list(iqrs.positive_descriptors | iqrs.external_positive_descriptors)

            neg = list(iqrs.negative_descriptors | iqrs.external_negative_descriptors)

            ADJs = (pos, neg)

            rel_index=plugin.from_plugin_config( plugin.to_plugin_config
            (iqrs.rel_index), get_relevancy_index_impls())

        except:
            raise NotImplementedError("The ``from_iqr_session`` classmethod is "
                                  "not implemented for class ``{}``."
                                  .format(cls.__name__))

        return Logit_SaliencyBlackbox(ADJs, rel_index, descr_gen, base_image)

    def get_config(self):
        
        return {
            'ready': True,
        }

    def transform(self, descriptors):
        """
        Transform some descriptor element into a saliency scalar.
        :param collections.Iterable[smqtk.representation.DescriptorElement] descriptors:
            Descriptor to get the saliency of.
        :param type base image descriptor vector   
        :return: The saliency value for the given descriptor.
        :rtype: numpy.ndarray[float]
        """
        uuid_bas=[]
        buff = six.BytesIO()
        (self.base_image).save(buff, format="png")
        de = DataMemoryElement(buff.getvalue(),
                               content_type='image/png')
        
        #TODO: Expand to multiple queries
        descriptors_list=list(descriptors)
        rel_train_set=[single_descr for single_descr in descriptors_list]
        T_descr=self.descr_gen.compute_descriptor(de)
        rel_train_set.append(T_descr)
        self.rel_index.build_index(rel_train_set)
        RI_scores=self.rel_index.rank(*self.ADJs) 
        diff = np.ones(len(descriptors_list))
        for i in range(len(descriptors_list)):
            diff[i]= RI_scores[rel_train_set[i]] - RI_scores[rel_train_set[-1]]
        return diff

class Logit_ImageSaliencyAugmenter(ImageSaliencyAugmenter):
    """
    Algorithm that yields a number of augmentations of an input image, as well
    as preserved-area masks, used for use in saliency map generation.
    """
    def __init__(self, window_size,stride):

        self.window_size = window_size

        self.stride=stride

        self.masks = self.generate_block_masks(window_size=window_size,stride=stride)

    @classmethod
    def is_usable(cls):
        """
        Check whether this implementation is available for use.
        Required valid presence of svm and svmutil modules
        :return:
            Boolean determination of whether this implementation is usable.
        :rtype: bool
        """
        return ImageSaliencyAugmenter and np
    
    @classmethod
    def get_config(self):

        return {
            'window_size': self.window_size,
            'stride': self.stride,
        }

    def generate_block_masks(self, window_size=50, stride=20, image_size=(224,224)):
        """
        Generating the sliding window style masks.
        
        :param window_size: the block window size (with value 0, other areas with value 1)
        :type window_size: int
        
        :param stride: the sliding step
        :type stride: int
        
        :param image_size: the mask size which should be the same to the image size
        :type image_size: tuple
        
        :return: the sliding window style masks
        :rtype: numpy array
        """
        rows = np.arange(0 + stride - window_size, image_size[0], stride)
        cols = np.arange(0 + stride - window_size, image_size[1], stride)

        mask_num = len(rows) * len(cols)
        self._log.debug('mask_num: {}'.format(mask_num))
        masks = np.ones((mask_num, image_size[0], image_size[1]), dtype=np.float64)
        i = 0
        for r in rows:
            for c in cols:
                if r<0:
                    r1 = 0
                else:
                    r1 = r
                if r + window_size > image_size[0]:
                    r2 = image_size[0]
                else:
                    r2 = r + window_size
                if c<0:
                    c1 = 0
                else:
                    c1 = c
                if c + window_size > image_size[1]:
                    c2 = image_size[1]
                else:
                    c2 = c + window_size
                masks[i, r1:r2, c1:c2] = 0
                i += 1

        masks = masks.reshape(-1, *image_size, 1)
        return masks


    def generate_masked_imgs(self, masks, img):
        """
        Apply the N filters/masks onto one input image
        :param index: mask index
        :return: masked images
        """
        masked_imgs = []
        for mask in masks:
            masked_img = np.multiply(mask, img, casting='unsafe')
            masked_imgs.append(Image.fromarray(np.uint8(masked_img)))
        return masked_imgs
    
    def augment(self, image_mat):
        
        """
        :param numpy.ndarray image_mat:
            Image matrix to be augmented.
        :return: A numpy arrays of augmented image matrices as well as masks
            that indicate the regions in the augmented images that are
            unmodified with respect to the input image (preserved regions).
            Returned augmented images should be in the dimension format
            [index, height, width [,channel]] with the the same data type as
            the input image matrix.
            Returned masks should be in the dimension format
            [index, height, width,channel] with the boolean data type.
        :rtype: PIL.Image.array
        """
         
        masked_images = self.generate_masked_imgs(self.masks, image_mat)
        return (masked_images, self.masks)

SALIENCY_BLACKBOX_CLASS = Logit_SaliencyBlackbox
IMG_SALIENCY_AUGMENTER_CLASS = Logit_ImageSaliencyAugmenter
