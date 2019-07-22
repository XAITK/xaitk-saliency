import numpy as np
from tqdm import tqdm
import copy
from PIL import Image
import os
from sklearn.metrics.pairwise import euclidean_distances
import six
from smqtk.algorithms import SmqtkAlgorithm
from smqtk.algorithms.saliency import SaliencyBlackbox,ImageSaliencyAugmenter
from smqtk.representation.data_element.memory_element import DataMemoryElement

__author__ = "bhavan.vasu@kitware.com"


class SBSM_SaliencyBlackbox (SaliencyBlackbox):
    
    """
    Blackbox function that produces some floating point scalar value for a
    given descriptor element.
    """

    def __init__(self, query_f, descr_gen, base_image):
        
        self.query_f=query_f
        
        self.descr_gen=descr_gen
        
        self.base_image=base_image

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
        #Check if iqrs and descriptor_generator is usable
        try:
            assert iqrs
            assert descr_gen.is_usable
            query_f=[ext_pos.vector() for ext_pos in iqrs.external_positive_descriptors]

        except:
            raise NotImplementedError("The ``from_iqr_session`` classmethod is "
                                  "not implemented for class ``{}``."
                                  .format(cls.__name__))
        return SBSM_SaliencyBlackbox(query_f,descr_gen,base_image)
       
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
        uuid_to_base_desc=self.descr_gen.compute_descriptor(de)
        #TODO: Expand to multiple external positive and negative
        org_dis=abs(euclidean_distances(self.query_f[0].reshape(1, -1)
                                ,(uuid_to_base_desc.vector()).reshape(1, -1)))
        descriptors_list=list(descriptors)
        diff = np.ones(len(descriptors_list))
        for i in range(len(descriptors_list)):
            diff[i]=max(abs(euclidean_distances(descriptors_list[i].vector().reshape(1, -1),
                                                self.query_f[0].reshape(1, -1)))-org_dis,0)
        return diff

class SBSM_ImageSaliencyAugmenter (ImageSaliencyAugmenter):
    """
    Algorithm that yields a number of augmentations of an input image, as well
    as preserved-area masks, used for use in saliency map generation.
    """

    @classmethod
    def is_usable(cls):
        """
        Check whether this implementation is available for use.
        Required valid presence of svm and svmutil modules
        :return:
            Boolean determination of whether this implementation is usable.
        :rtype: bool
        """
        
        return ImageSaliencyAugmenter and tqdm and np
    
    def __init__(self, window_size=20, stride=4):

        self.window_size = window_size

        self.stride = stride

        self.masks = self.generate_block_masks(window_size=self.window_size,stride=self.stride)

    def get_config(self):

        return {
            'window_size': self.window_size,
            'stride': self.stride,
        }


    def generate_block_masks(self,window_size=20, stride=4, image_size=(224, 224)):
        """The Images are resized to 224x224 to enable re-use of masks
        Generating the sliding window style masks
        :param window_size: the block window size (with value 0, other areas with value 1)
        :type window_size: int
        :param stride: the sliding step
        :type stride: int
        :param image_size: the mask size which should be the same to the image size
        :type image_size: tuple (default: (224, 224))
        :return: the sliding window style masks
        :rtype: torch.cuda.Tensor
        """
        try:
            
            assert (image_size[0]-window_size)%stride==0  

        except:
     
            self._log.debug("Change window size and stride to pass assert")

        if not os.path.isfile('block_mask_{}_{}.npy'.format(window_size, stride)):
            grid_num_r = ((image_size[0] - window_size) // stride)+1
            grid_num_c = ((image_size[1] - window_size) // stride)+1
            mask_num = grid_num_r * grid_num_c
            masks = np.ones((mask_num, image_size[0], image_size[1]), dtype=np.float32)
            i = 0
            for r in tqdm(np.arange(0, image_size[0] - window_size+1, stride), total=grid_num_r, desc="Generating rows"):
                for c in np.arange(0, image_size[1] - window_size+1, stride):
                    masks[i, r:r + window_size, c:c + window_size] = 0.0
                    i += 1

            masks = masks.reshape(-1, *image_size)
            masks.tofile('block_mask_{}_{}.npy'.format(window_size, stride))
        else:
            self._log.debug("Loading masks from file block_mask_{}_{}.npy".format(window_size, stride))
            masks = np.fromfile('block_mask_{}_{}.npy'.format(window_size, stride),
                                dtype=np.float32).reshape(-1,  *image_size)
        return masks


    def generate_masked_imgs(self,masks, img):
        """
        Apply the N filters/masks onto one input image
        :param index: mask index
        :return: masked images
        """

        if (img.ndim == 2):
            channels = 1 

        if (img.ndim == 3):
            channels = img.shape[-1]

        masked_imgs = []
        masked_img=copy.deepcopy(img)
        for cnt,mask in enumerate(masks):
            for ind in range(channels):
                masked_img[:,:,ind] = np.multiply(mask, img[:,:,ind])
            masked_imgs.append(Image.fromarray(masked_img))
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
        :rtype: (numpy.ndarray, numpy.ndarray)
        """
         
        masked_images=self.generate_masked_imgs(self.masks,image_mat)
        return (masked_images,self.masks)

SALIENCY_BLACKBOX_CLASS=SBSM_SaliencyBlackbox
IMG_SALIENCY_AUGMENTER_CLASS=SBSM_ImageSaliencyAugmenter
