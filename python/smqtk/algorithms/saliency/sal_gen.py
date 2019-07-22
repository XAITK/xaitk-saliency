from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime
import six
import numpy as np
import PIL
import copy
from smqtk.algorithms.saliency import ImageSaliencyMapGenerator
from smqtk.algorithms.descriptor_generator import DescriptorGenerator
from smqtk.representation.data_element.file_element import DataFileElement
from smqtk.representation.data_element.memory_element import DataMemoryElement


class Logit_ImageSaliencyMapGenerator(ImageSaliencyMapGenerator):
    """
    Interface for the method of generation of a saliency map given an image
    augmentation and blackbox algorithms.
    """
    def __init__(self,threshold=0.2):
        
        self.thresh=threshold
   
    def get_config(self):

        return {
            'threshold': 0.2,
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
        #TODO:appropriate returns
        return plt and np
    
    def generate(self,base_image, augmenter, descriptor_generator,
                 blackbox):
        """
        Generate an image saliency heat-map matrix given a blackbox's behavior
        over the descriptions of an augmented base image.
        :param PIL Image image_mat:
            PIL Image of the RGB format that is
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
        org_hw=base_image.size
        org_img=copy.deepcopy(base_image)
        base_image_PIL=base_image.resize((224,224) ,PIL.Image.BILINEAR)
        augs, masks = augmenter.augment(np.array(base_image_PIL))
        idx_to_uuid = []
        def iter_aug_img_data_elements():
            for a in augs:
               buff = six.BytesIO()
               (a).save(buff, format="png")
               de = DataMemoryElement(buff.getvalue(),
                                   content_type='image/png')
               idx_to_uuid.append(de.uuid())
               yield de

        uuid_to_desc=descriptor_generator.compute_descriptor_async(iter_aug_img_data_elements())

        scalar_vec = blackbox.transform((uuid_to_desc[uuid] for uuid in idx_to_uuid))
        def overlay_saliency_map(sa_map, org_img):
            """
            overlay the saliency map on top of original image
            :param sa_map: saliency map
            :type sa_map: numpy.array
            :param org_img: Original image
            :type org_img: numpy.array
            :return: Overlayed image
            :rtype: PIL Image
            """
            plt.switch_backend('agg')
            sizes = np.shape(sa_map)
            height = float(sizes[0])
            width = float(sizes[1])
            sa_map.resize(sizes[0],sizes[1])
            fig = plt.figure(dpi=int(height))
            fig.set_size_inches((width / height), 1, forward=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(org_img)
            ax.imshow(sa_map, cmap='jet', alpha=0.5)
            fig.canvas.draw()
            np_data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            np_data = np_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            im = PIL.Image.fromarray(np_data)
            im_size = np.shape(im)
            org_h =  im_size[1]
            org_w = im_size[0]
            im = im.resize((org_w, org_h), PIL.Image.BILINEAR)
            plt.close()
            return im

        def weighted_avg(scalar_vec,masks):
            masks = masks.reshape(-1,224,224,1)
            cur_filters = copy.deepcopy(masks[:,:,:,0])
            count = masks.shape[0] - np.sum(cur_filters, axis=0)
            count = np.ones(count.shape)

            for i in range(len(cur_filters)):
                cur_filters[i] = (1.0 - cur_filters[i]) * np.clip(scalar_vec[i], a_min=0.0, a_max=None)
            res_sa = np.sum(cur_filters, axis=0) / count
            
            sa_max = np.max(res_sa)
            res_sa = np.clip(res_sa, a_min=sa_max * self.thresh, a_max = None)
            return PIL.Image.fromarray(res_sa)

        final_sal_map=weighted_avg(scalar_vec,masks)
        final_sal_map_PIL=final_sal_map.resize((org_hw), PIL.Image.BILINEAR)
        sal_map_ret=overlay_saliency_map(np.array(final_sal_map_PIL),np.array(org_img))
        return sal_map_ret

IMG_SALIENCY_GENERATOR_CLASS=Logit_ImageSaliencyMapGenerator
