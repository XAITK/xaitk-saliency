import abc

from smqtk.algorithms import SmqtkAlgorithm
from smqtk.representation.data_element.memory_element import DataMemoryElement


class SaliencyBlackbox (SmqtkAlgorithm):
    """
    Blackbox function that produces some floating point scalar value for a
    given descriptor element.
    """

    @classmethod
    def from_iqr_session(cls, iqrs, descr_generator, base_elem):
        """
        Create an ``SaliencyBlackbox`` instance from an
        :class:`smqtk.iqr.IqrSession` instance.

        Not all implementations of ``SaliencyBlackbox`` implement
        this method and may raise a ``NotImplementedError`` exception.

        :raises NotImplementedError:
            Construction from a ``IqrSession`` is not defined for this
            implementation.

        :param smqtk.iqr.IqrSession iqrs:
            An :py:class:`smqtk.iqr.IqrSession` instance to seed construction
            if a blackbox instance.
        :param descr_generator:
            Descriptor generator instance.
        :param base_elem:
            A :py:class:`smqtk.representation.DataElement` representing the
            data over which a saliency map may be generated.

        :return: A new instance of of a class implementing the
            ``SaliencyBlackbox`` class.
        :rtype: SaliencyBlackbox

        """
        raise NotImplementedError("The ``from_iqr_session`` classmethod is "
                                  "not implemented for class ``{}``."
                                  .format(cls.__name__))

    @abc.abstractmethod
    def transform(self, descriptors):
        """
        Transform some number of descriptor elements into a saliency scalar
        values.

        :param collections.Iterable[smqtk.representation.DescriptorElement] descriptors:
            Descriptors to get the saliency values of.

        :return: The saliency value for the given descriptor.
        :rtype: numpy.ndarray[float]
        """


class ImageSaliencyAugmenter (SmqtkAlgorithm):
    """
    Algorithm that yields a number of augmentations of an input image, as well
    as preserved-area masks, used for use in saliency map generation.
    """

    @abc.abstractmethod
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
        :rtype: (PIL.Image, numpy.ndarray)
        """


class ImageSaliencyMapGenerator (SmqtkAlgorithm):
    """
    Interface for the method of generation of a saliency map given an image
    augmentation and blackbox algorithms.
    """

    @abc.abstractmethod
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
