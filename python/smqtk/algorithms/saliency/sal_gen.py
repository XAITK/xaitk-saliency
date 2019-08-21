from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime
import six
import numpy as np
import PIL
import cv2
import copy
from smqtk.algorithms.saliency import ImageSaliencyMapGenerator
from smqtk.algorithms.descriptor_generator import DescriptorGenerator
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.representation.data_element.memory_element \
      import DataMemoryElement
from skimage.transform import resize


class Logit_ImageSaliencyMapGenerator(ImageSaliencyMapGenerator):
    """
    Interface for the method of generation of a saliency map given an image
    augmentation and blackbox algorithms.
    """
    def __init__(self,threshold=0.2):
        
        self.thresh = threshold

        self.org_hw = None
   
    def get_config(self):

        return {
            'threshold': self.thresh,
        }

    @classmethod
    def is_usable(cls):
        """
        Check whether this implementation is available for use.
        Required valid presence of svm and svmutil modules
        :return:
            Boolean determination of whether this implementation is usable.
        :rtype: bool
        """
        return plt and np

    def overlay_saliency_map(self,sa_map, org_img):
        """
        overlay the saliency map on top of original image
        :param numpy.array sa_map: saliency map
        :param numpy.array org_img: Original image
        :return: Overlayed image
        :rtype: PIL.Image
        """
        plt.switch_backend('agg')
        height = float(self.org_hw[0])
        width = float(self.org_hw[1])
        fig = plt.figure(dpi=int(height))
        fig.set_size_inches((width / height), 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(org_img)
        ax.imshow(sa_map, cmap='jet', alpha=0.5)
        fig.canvas.draw()
        np_data = np.fromstring(fig.canvas.tostring_rgb(), \
           dtype=np.uint8, sep='')
        np_data = np_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        im = PIL.Image.fromarray(np_data)
        plt.close()
        return im

    def weighted_avg(self,scalar_vec,masks):
        masks = masks.reshape(-1,224,224,1)
        cur_filters = copy.deepcopy(masks[:,:,:,0])
        count = masks.shape[0] - np.sum(cur_filters, axis=0)
        count = np.ones(count.shape)

        for i in range(len(cur_filters)):
            cur_filters[i] = (1.0 - cur_filters[i]) * np.clip(scalar_vec[i], \
                       a_min=0.0, a_max=None)
        res_sa = np.sum(cur_filters, axis=0) / count

        res_sa = np.clip(res_sa, a_min=((np.max(res_sa)) * self.thresh), \
                    a_max = None)
        return res_sa

    def generate(self, base_image, augmenter, descriptor_generator,
                 blackbox):
        """
        Generate an image saliency heat-map matrix given a blackbox's behavior
        over the descriptions of an augmented base image.
        :param numpy.ndarray base_image:
             Numpy matrix of the format [height, width [,channel]] that is
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
        :return: A :py:class:`PIL.Image` of the same [height, width]
            shape as the input image matrix but of floating-point type within
            the range of [0,1], where areas of higher value represent more
            salient regions according to the given blackbox algorithm.
        :rtype: PIL.Image
        """

        self.org_hw = np.shape(base_image)[0:2]
        base_image_resized = cv2.resize(base_image,(224,224) \
                     ,interpolation=cv2.INTER_LINEAR)
        augs, masks = augmenter.augment(base_image_resized)
        idx_to_uuid = []
        def iter_aug_img_data_elements():
            for a in augs:
               buff = six.BytesIO()
               (a).save(buff, format="bmp")
               de = DataMemoryElement(buff.getvalue(),
                                   content_type='image/bmp')
               idx_to_uuid.append(de.uuid())
               yield de
        uuid_to_desc = descriptor_generator.compute_descriptor_async \
                    (iter_aug_img_data_elements())
        scalar_vec = blackbox.transform( \
                     (uuid_to_desc[uuid] for uuid in idx_to_uuid))
        final_sal_map = self.weighted_avg(scalar_vec,masks)
        final_sal_map = cv2.resize(final_sal_map, \
         (self.org_hw[1], self.org_hw[0]),interpolation=cv2.INTER_LINEAR)
        sal_map_ret = self.overlay_saliency_map(final_sal_map,base_image)
        return sal_map_ret

IMG_SALIENCY_GENERATOR_CLASS=Logit_ImageSaliencyMapGenerator
