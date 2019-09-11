from ._interface import ImageSaliencyAugmenter, ImageSaliencyMapGenerator

import six
import numpy as np
from skimage.transform import resize
from smqtk.representation.data_element.memory_element import DataMemoryElement


class RISEAugmenter (ImageSaliencyAugmenter):
    def __init__(self, N, s, p1, input_size):
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
        self.masks = masks.reshape(-1, *input_size, 1)
        self.input_size = input_size
    
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
            [index, height, width] with the boolean data type.
        :rtype: (numpy.ndarray, numpy.ndarray)
        """
        # If image is grayscale
        if len(image_mat.shape) == 2:
            image_mat = np.expand_dims(image_mat, 2).repeat(3, axis=2)
        return self.masks * image_mat, self.masks
            
        
class RISEGenerator (ImageSaliencyMapGenerator):
    def generate(self, image_mat, augmenter, descriptor_generator,
                 blackbox):
        """
        Generate an image saliency heat-map matrix given a blackbox's behavior
        over the descriptions of an augmented base image.

        :param numpy.ndarray image_mat:
            Numpy image matrix of the format [height, width [,channel]] that is
            to be augmented.

        :param ImageSaliencyAugmenter augmenter:
            Augmentation algorithm following
            the :py:class:`ImageSaliencyAugmenter` interface.

        :param smqtk.algorithms.DescriptorGenerator descriptor_generator:
            A descriptor generation algorithm following
            the :py:class:`smqtk.algorithms.DescriptorGenerator` interface.

        :param SaliencyBlackbox blackbox:
            Blackbox algorithm implementation following
            the :py:class:`SaliencyBlackbox` interface.

        :return: A :py:class:`numpy.ndarray` matrix of the same [height, width]
            shape as the input image matrix but of floating-point type within
            the range of [0,1], where areas of higher value represent more
            salient regions according to the given blackbox algorithm.
        :rtype: numpy.ndarray[float]
        """
        
        resized_img = resize(image_mat, augmenter.input_size, order=1)
        masked_images, masks = augmenter.augment(resized_img)
        
        idx_to_uuid = []
        def iter_aug_img_data_elements():
            for a in masked_images:
                buff = six.BytesIO()
                (a).save(buff, format="bmp")
                de = DataMemoryElement(buff.getvalue(),
                                       content_type='image/bmp')
                idx_to_uuid.append(de.uuid())
                yield de

        uuid_to_desc = descriptor_generator.compute_descriptor_async(iter_aug_img_data_elements())
        scores = blackbox.transform((uuid_to_desc[uuid] for uuid in idx_to_uuid))
        
        # Compute a weighted average of masks w.r.t. the scores
        saliency_map = np.average(masks, axis=0, weights=scores)
        saliency_map = np.squeeze(saliency_map)
        # Normalize
        saliency_map /= masks.mean(axis=0)
        # Resize back to the original image shape
        saleincy_map = resize(saliency_map, image_mat.shape, order=1)
        
        # At this point the saliency map will be in some range [a, b], 0 <= a <= b <= 1.
        # The absolute values characterize the average score of the masked image and 
        # therefore have some important information. However, for visualization purposes,
        # the saliency map can be rescaled to [0, 1].
        # saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        
        return saliency_map
            
