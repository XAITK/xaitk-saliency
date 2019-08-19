from PIL import Image
import os
from sklearn.metrics.pairwise import euclidean_distances
import six
from smqtk.algorithms import SmqtkAlgorithm
from smqtk.algorithms.saliency import SaliencyBlackbox,ImageSaliencyAugmenter
from smqtk.representation.data_element.memory_element import DataMemoryElement

try:
    import numpy as np
    from tqdm import tqdm
    import copy
except ImportError as ex:
    logging.getLogger(__name__).warning("Failed to import numpy/tqdm \
         /copy module: %s", str(ex))
    np = None
    tqdm = None
    copy = None


class SBSM_SaliencyBlackbox (SaliencyBlackbox):
    """
    SBSM_SaliencyBlackbox function that yields some floating point 
    scalar value for a given masked base image descriptor element that 
    signifies the proximity between the query image and masked image
    descriptors, used by class implementations of 'ImageSaliencyMapGenerator'.
    """

    def __init__(self, query_f, base_descr):
        """
        :param query_f: Feature of query image
        :param type: smqtk.representation.DescriptorElement
        :param base_descr: Base image descriptor
        :param type: smqtk.representation.DescriptorElement
        """    

        self.query_f = query_f        
        self.base_descr = base_descr

    @classmethod
    def is_usable(cls):
        """
        Check whether this implementation is available for use.
        Required valid presence of query image feature 
        and base image descriptor
        :return:
            Boolean determination of whether this implementation is usable.
        :rtype: bool
        """

        valid = True
        if not valid:
            cls.get_logger().debug("six python module cannot be imported")
        return valid
                
    @classmethod
    def from_iqr_session(cls, iqrs, descr_gen, base_image): 
        """
        Create an ``SaliencyBlackbox`` instance from iqrs session,
        descriptor generator and base_image.
        :param iqrs:`smqtk.iqr.IqrSession` instance.
        :param type: smqtk.iqr.IqrSession
        :param descr_gen: The descriptor generator used by smqtk.
        :param type: smqtk.algorithms.DescriptorGenerator.
        :param base_image: The Base image for which we need to
        calculate a saliency map.
        :param type: PIL Image of the base image.  
        :return: A new instance of a class implementing the
            ``SaliencyBlackbox`` class.
        :rtype: SaliencyBlackbox
        """

        assert iqrs
        if (len(iqrs.external_positive_descriptors)) != 1:
            raise NotImplementedError("Saliency generation in class``{}`` "
                                  "only supports one query sample, but"
                                  "recieved more than 1 external positive."
                                  .format(cls.__name__))
        query_f = [ext_pos.vector() for ext_pos in \
           iqrs.external_positive_descriptors]
        buff = six.BytesIO()
        base_image.save(buff, format="bmp")
        de = DataMemoryElement(buff.getvalue(),
                               content_type='image/bmp')
        base_descr = descr_gen.compute_descriptor(de)
        return SBSM_SaliencyBlackbox(query_f, base_descr)
       
    def get_config(self):
        """
        Return a JSON-compliant dictionary that could be passed to this
        class's ``from_config`` method to produce an instance with identical
        configuration.
        In the common case, this involves naming the keys of the dictionary
        based on the initialization argument names as if it were to be passed
        to the constructor via dictionary expansion.
        :return: JSON type compliant configuration dictionary.
        :rtype: dict
        """

        return {
            'query_f': self.query_f,
            'base_descr': self.base_descr,
        }

    def transform(self, descriptors):
        """
        Transform some descriptor element into a saliency scalar.
        :param collections.Iterable[smqtk.representation.DescriptorElement]
        descriptors:
        Descriptor to get the saliency of.
        :param type base image descriptor vector   
        :return: The saliency value for the given descriptor.
        :rtype: numpy.ndarray[float]
        """
     
        org_dis = abs(euclidean_distances(self.query_f[0].reshape(1, -1)
                                ,(self.base_descr.vector()).reshape(1, -1)))
        descriptors_list = list(descriptors)
        diff = np.ones(len(descriptors_list))
        query_f_reshaped = self.query_f[0].reshape(1, -1)
        for i in range(len(descriptors_list)):
            diff[i] = max(abs(euclidean_distances \
              (descriptors_list[i].vector().reshape(1, -1), \
               query_f_reshaped))-org_dis,0)
        return diff


class SBSM_ImageSaliencyAugmenter (ImageSaliencyAugmenter):
    """
    Algorithm that yields a number of augmentations of an input image,
    as well as preserved-area masks, used for use in saliency map generation.
    For a more robust augmenter that can work for different image_sizes 
    use implementation class 'Logit_ImageSaliencyAugmenter'.
    """

    def __init__(self, window_size=20, stride=4):
        """
        :param window_size: the block window size 
        (with value 0, other areas with value 1)
        :type window_size: int
        :param stride: the sliding step
        :type stride: int
        """

        self.window_size = window_size
        self.stride = stride
        self.masks = self.generate_block_masks(self.window_size,self.stride)

    @classmethod
    def is_usable(cls):
        """
        Check whether this implementation is available for use.
        Required valid presence of tqdm and copy modules
        :return:
            Boolean determination of whether this implementation is usable.
        :rtype: bool
        """

        valid = (tqdm is not None) and (copy is not None)
        if not valid:
            cls.get_logger().debug("tqdm or copy python \
                  module cannot be imported")
        return valid

    def get_config(self):
        """
        Return a JSON-compliant dictionary that could be passed to
        this class's ``from_config`` method to produce an instance 
        with identical configuration.
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

    def generate_block_masks(self, window_size, stride, image_size=(224, 224)):
        """
        Generates sliding window type binary masks used in augment() to mask
        an image. The Images are resized to 224x224 to enable re-use of masks
        Generating the sliding window style masks
        :param window_size: the block window size
          (with value 0, other areas with value 1)
        :type window_size: int
        :param stride: the sliding step
        :type stride: int
        :param image_size: the mask size which should
          be the same to the image size
        :type image_size: tuple (default: (224, 224))
        :return: the sliding window style masks
        :rtype: numpy.ndarray
        """

        
        #The augmenter only supports certain factors of window_size and stride
        #for a more robust augmenter use Logit_ImageSaliencyAugmenter
        if (image_size[0]-window_size)%stride != 0: 
            raise ValueError("Change window size and stride to satisfy "
                              "condition:(image_height-window_size)%stride=0")
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

    def generate_masked_imgs(self, masks, img):
        """
        Apply the masks onto one input image
        :param masks: sliding window type masks in [1, Height, Weight, 1] format.
        :param type: numpy.ndarray
        :param img: Original base image
        :param type: numpy.ndarray 
        :return: List masked images
        :rtype: List of PIL Images 
        """

        if (img.ndim == 2):
            channels = 1 
        if (img.ndim == 3):
            channels = 3
        masked_imgs = []
        masked_img = copy.deepcopy(img)
        for cnt,mask in enumerate(masks):
            for ind in range(channels):
                masked_img[:,:,ind] = np.multiply(mask, img[:,:,ind])
            masked_imgs.append(Image.fromarray(np.uint8(masked_img)))
        return masked_imgs
        
    def augment(self, image_mat):
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
         
        masked_images = self.generate_masked_imgs(self.masks,image_mat)
        return (masked_images,self.masks)

SALIENCY_BLACKBOX_CLASS=SBSM_SaliencyBlackbox
IMG_SALIENCY_AUGMENTER_CLASS=SBSM_ImageSaliencyAugmenter
