from PIL import Image
import numpy as np
import os
from sklearn.metrics.pairwise import euclidean_distances
import six
from smqtk.algorithms.relevancy_index import get_relevancy_index_impls
from smqtk.algorithms import get_descriptor_generator_impls
from smqtk.algorithms import SmqtkAlgorithm
from smqtk.algorithms.saliency import SaliencyBlackbox,ImageSaliencyAugmenter
from smqtk.representation.data_element.memory_element import DataMemoryElement
from smqtk.utils import plugin

    
class Logit_SaliencyBlackbox (SaliencyBlackbox):
    """
    Logit_SaliencyBlackbox function yields some floating point scalar value
    for a given masked base image descriptor element that signifies the 
    difference between the confidence value of the query image and masked
    image descriptors, used by class implementations of
    'ImageSaliencyMapGenerator'.
    """

    def __init__(self, pos_descriptors, neg_descriptors, rel_index, T_descr):
        """
        :param list[smqtk.representation.DescriptorMemoryElement] 
            pos_descriptors: is a set of positive descriptors.
        :param list[smqtk.representation.DescriptorMemoryElement]
               neg_descriptors: is a set of negative descriptors.
        :param rel_index: Plugin implementation of the algorithms used to 
        generate relevance index used to rank images
        :type: A new instance of a class implementing the
            ``RelevancyIndex`` class.
        :param smqtk.representation.DescriptorElement
              T_descr: Base image feature descriptor  
        """
    
        self.pos_de = pos_descriptors
        self.neg_de =  neg_descriptors
        self.ADJs = (self.pos_de, self.neg_de)
        self.rel_index = rel_index
        self.T_descr = T_descr
    
    @classmethod
    def is_usable(self):
        """
        Check whether this implementation is available for use.
        Required valid presence of Ajudications and a new instance of 
        a class implementing the ``RelevancyIndex`` class.
        :return: Boolean determination of whether implementation is usable.
        :rtype: bool
        """

        valid = True
        return valid
        
    @classmethod
    def from_iqr_session(cls, iqrs, descr_gen, base_image): 
        """
        Create an ``SaliencyBlackbox`` instance from iqrs session, descriptor
        generator and base_image.
        :param smqtk.iqr.IqrSession iqrs:`smqtk.iqr.IqrSession` instance.
        :param smqtk.algorithms.DescriptorGenerator descr_gen:
            The descriptor generator used by smqtk.
        :param PIL.Image base_image: The Base image for which we 
           need to calculate a saliency map.
        :return: A new instance of a class implementing the
            ``SaliencyBlackbox`` class.
        :rtype: SaliencyBlackbox
        """

        assert iqrs
        pos = list(iqrs.positive_descriptors |\
          iqrs.external_positive_descriptors)
        neg = list(iqrs.negative_descriptors |\
          iqrs.external_negative_descriptors)
        rel_index = plugin.from_plugin_config\
         (plugin.to_plugin_config(iqrs.rel_index),\
          get_relevancy_index_impls())
        buff = six.BytesIO()
        base_image.save(buff, format="bmp")
        de = DataMemoryElement(buff.getvalue(),
                               content_type='image/bmp')
        T_descr=descr_gen.compute_descriptor(de)
        return Logit_SaliencyBlackbox(pos, neg, rel_index, T_descr)

    def get_config(self):
        """
        Returns a JSON-compliant dictionary that could be passed to 
        the class's ``from_config`` method to produce an instance 
        with identical configuration.
        In the common case, this involves naming the keys of the dictionary
        based on the initialization argument names as if it were to be passed
        to the constructor via dictionary expansion.
        :return: JSON type compliant configuration dictionary.
        :rtype: dict
        """
        
        return {
            'pos_descriptors': self.pos_de,
            'neg_descriptors': self.neg_de,
            'rel_index': self.rel_index,
            'T_descr': self.T_descr,
        }

    def transform(self, descriptors):
        """
        Transform some descriptor element into a saliency scalar.
        :param Iterable[smqtk.representation.DescriptorElement] 
           descriptors:Descriptor of augmentations to get their scalar value.
        :return: The saliency value for the given descriptor.
        :rtype: numpy.ndarray[float]
        """
        
        descriptors_list=list(descriptors)
        rel_train_set=[single_descr for single_descr in descriptors_list]
        rel_train_set.append(self.T_descr)
        self.rel_index.build_index(rel_train_set)
        RI_scores=self.rel_index.rank(*self.ADJs) 
        diff = np.ones(len(descriptors_list))
        base_RI = RI_scores[rel_train_set[-1]]
        for i in range(len(descriptors_list)):
            diff[i]= RI_scores[rel_train_set[i]] - base_RI
        return diff


class Logit_ImageSaliencyAugmenter(ImageSaliencyAugmenter):
    """
    Robust Augmenter that yields a number of augmentations 
    of an input image, as well as preserved-area masks,
    used for use in saliency map generation. 
    """

    def __init__(self, window_size=50, stride=20):
        """
        :param int window_size: the block window size 
        (with value 0, other areas with value 1)
        :param int stride: the sliding step
        """

        self.window_size = window_size
        self.stride = stride
        self.masks = self.generate_block_masks(window_size=window_size,\
        stride=stride)

    @classmethod
    def is_usable(cls):
        """
        Check whether this implementation is available for use.
        Required valid to be True
        :return:
            Boolean determination of whether this implementation is usable.
        :rtype: bool
        """
        
        valid  = True
        return valid
    
    @classmethod
    def get_config(self):
        """
        Return a JSON-compliant dictionary that could be passed to 
        this class's ``from_config`` method to produce an 
        instance with identical configuration.
        In the common case, this involves naming the keys of the dictionary
        based on the initialization argument names as if it were to be passed
        to the constructor via dictionary expansion.
        :return: JSON type compliant configuration dictionary.
        :rtype: dict
        """

        return {
            'window_size': self.window_size,
            'stride': self.stride,
        }

    def generate_block_masks(self, window_size, stride, image_size=(224,224)):
        """
        Generates sliding window type binary masks used in augment() to 
        mask an image. The Images are resized to 224x224 to 
        enable re-use of masks Generating the sliding window style masks.
        :param int window_size: the block window size 
        (with value 0, other areas with value 1)
        :param int stride: the sliding step
        :param tuple image_size: the mask size which should be the 
        same to the image size
        :return: the sliding window style masks
        :rtype: numpy.ndarray
        """

        rows = np.arange(0 + stride - window_size, image_size[0], stride)
        cols = np.arange(0 + stride - window_size, image_size[1], stride)

        mask_num = len(rows) * len(cols)
        self._log.debug('mask_num: {}'.format(mask_num))
        masks = np.ones((mask_num, image_size[0], image_size[1])\
                                              , dtype=np.float64)
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
        mask_shape = [-1] + [*image_size] + [1]
        masks = masks.reshape(mask_shape)
        return masks

    def generate_masked_imgs(self, masks, img):
        """
        Apply the masks onto one input image
        :param numpy.ndarray masks: sliding window type masks in 
        [1, Height, Weight, 1] format.
        :param numpy.ndarray img: Original base image
        :return: List masked images
        :rtype: List of PIL Images 
        """

        masked_imgs = []
        for mask in masks:
            masked_img = np.multiply(mask, img, casting='unsafe')
            masked_imgs.append(Image.fromarray(np.uint8(masked_img)))
        return masked_imgs
    
    def augment(self, image_mat):        
        """
        Takes in an image matrix and returns its augmented version
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
        :rtype: (PIL.Image, numpy.ndarray)
        """
         
        masked_images = self.generate_masked_imgs(self.masks, image_mat)
        return (masked_images, self.masks)

SALIENCY_BLACKBOX_CLASS = Logit_SaliencyBlackbox
IMG_SALIENCY_AUGMENTER_CLASS = Logit_ImageSaliencyAugmenter
