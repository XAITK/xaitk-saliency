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
    
    def __init__(self,query_f,descr_gen,base_image):
        self.query_f=query_f
        self.descr_gen=descr_gen
        self.base_image=base_image

    """
    Blackbox function that produces some floating point scalar value for a
    given descriptor element.
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
        #TODO:appropriate returns
        return euclidean_distances and np

        
    @classmethod
    def from_iqr_session(cls, iqrs,descr_gen,base_image): #add descriptor_generator and base image
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
        try:#ADD ASSERT
            query_f=[ext_pos.vector() for ext_pos in iqrs.external_positive_descriptors]
        except:#if iqrs and descriptor_generator is not usable    
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
        #TODO:remove iterator
        def temp_ter_bas():
            #for imgs in base_image:
            buff = six.BytesIO()
            (self.base_image).save(buff, format="png")
            de = DataMemoryElement(buff.getvalue(),
                               content_type='image/png')
            uuid_bas.append(de.uuid())
            yield de
        
        uuid_to_base_desc=self.descr_gen.compute_descriptor_async(temp_ter_bas())
        #TODO: Expand to multiple queries
        org_dis=abs(euclidean_distances(self.query_f[0].reshape(1, -1)
                                ,(uuid_to_base_desc[uuid_bas[0]].vector()).reshape(1, -1)))
        descriptors_list=list(descriptors)
        diff = np.ones(len(descriptors_list))
        import pdb
        #pdb.set_trace()
        for i in range(len(descriptors_list)):
            diff[i]=max(abs(euclidean_distances(descriptors_list[i].vector().reshape(1, -1),
                                                self.query_f[0].reshape(1, -1)))-org_dis,0)
        return diff

class SBSM_ImageSaliencyAugmenter (ImageSaliencyAugmenter):
    @classmethod
    def is_usable(cls):
        """
        Check whether this implementation is available for use.
        Required valid presence of svm and svmutil modules
        :return:
            Boolean determination of whether this implementation is usable.
        :rtype: bool
        """
        #TODO:appropriate returns
        return euclidean_distances and np
    
    def __init__(self,image_sizes):
        self.image_size=image_sizes
        self.org_dim=None
        self.masks=self.generate_block_masks()
        #self.masks=self.generate_block_masks_from_gridsize()
    """
    Algorithm that yields a number of augmentations of an input image, as well
    as preserved-area masks, used for use in saliency map generation.
    """
    def get_config(self):
        return {
            'grid_size': self.grid_size,
        }


    def generate_block_masks(self,grid_size=20, stride=4, image_size=(224, 224)):
        """COmment about resize
        Generating the sliding window style masks
        :param grid_size: the block window size (with value 0, other areas with value 1)
        :type grid_size: int
        :param stride: the sliding step
        :type stride: int
        :param image_size: the mask size which should be the same to the image size
        :type image_size: tuple (default: (224, 224))
        :return: the sliding window style masks
        :rtype: torch.cuda.Tensor
        """
        assert (image_size[0]-grid_size)%stride==0  
        if not os.path.isfile('block_mask_{}_{}.npy'.format(grid_size, stride)):
            grid_num_r = ((image_size[0] - grid_size) // stride)+1
            grid_num_c = ((image_size[1] - grid_size) // stride)+1
            mask_num = grid_num_r * grid_num_c
            print("Number of masks",mask_num)
            masks = np.ones((mask_num, image_size[0], image_size[1]), dtype=np.float32)
            i = 0
            for r in tqdm(np.arange(0, image_size[0] - grid_size+1, stride), total=grid_num_r, desc="Generating rows"):
                for c in np.arange(0, image_size[1] - grid_size+1, stride):
                    masks[i, r:r + grid_size, c:c + grid_size] = 0.0
                    i += 1

            masks = masks.reshape(-1, *image_size)
            masks.tofile('block_mask_{}_{}.npy'.format(grid_size, stride))
        else:
            masks = np.fromfile('block_mask_{}_{}.npy'.format(grid_size, stride),
                                dtype=np.float32).reshape(-1,  *image_size)
        return masks

    def generate_block_masks_from_gridsize(self,image_size=[224,224], grid_size=(15,15)):
        """
        Generating the sliding window style masks.
     
        :param image_size: the mask size which should be the same as the image size
        :type image_size: tuple 
    
        :param grid_size: the number of rows and columns
        :type grid_size: tuple of ints (default: (5, 5))
    
        :return: the sliding window style masks
        :rtype: numpy array 
        """
        window_size = (image_size[0]//grid_size[0], image_size[1]//grid_size[1])
        stride = window_size
        grid_num_r = (image_size[0] - window_size[0]) // stride[0] + 1
        grid_num_c = (image_size[1] - window_size[1]) // stride[1] + 1
        mask_num = grid_num_r * grid_num_c
        print('mask_num {}'.format(mask_num))
        masks = np.ones((mask_num, image_size[0], image_size[1]), dtype=np.float32)
        i = 0
        for r in np.arange(0, image_size[0] - window_size[0] + 1, stride[0]):
            for c in np.arange(0, image_size[1] - window_size[1] + 1, stride[1]):
                masks[i, r:r + window_size[0], c:c + window_size[1]] = 0.0
                i += 1

        masks = masks.reshape(-1, *image_size)
        return masks


    def generate_masked_imgs(self,masks, img):
        """
        Apply the N filters/masks onto one input image
        :param index: mask index
        :return: masked images
        """
        masked_imgs = []
        self.org_dim=np.shape(img)
        masked_img=copy.deepcopy(img)
        for cnt,mask in enumerate(masks):
            masked_img[:,:,0] = np.multiply(mask, img[:,:,0])
            masked_img[:,:,1] = np.multiply(mask, img[:,:,1])
            masked_img[:,:,2] = np.multiply(mask, img[:,:,2])
            masked_imgs.append(Image.fromarray(masked_img))
        return masked_imgs
    
    def _get_smap_size(self):    
        
        return self.org_dim[0],self.org_dim[1],self.org_dim[2]
    
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
