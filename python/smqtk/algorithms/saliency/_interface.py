import abc

from smqtk.algorithms import SmqtkAlgorithm
from smqtk.representation.data_element.memory_element import DataMemoryElement


class SaliencyBlackbox (SmqtkAlgorithm):
    """
    Blackbox function that produces some floating point scalar value for a
    given descriptor element.
    """

    @classmethod
    def from_iqr_session(cls, iqrs):
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
        raise NotImplementedError("The ``from_iqr_session`` classmethod is "
                                  "not implemented for class ``{}``."
                                  .format(cls.__name__))

    @abc.abstractmethod
    def transform(self, descriptors):
        """
        Transform some descriptor element into a saliency scalar.

        :param collections.Iterable[smqtk.representation.DescriptorElement] descriptors:
            Descriptor to get the saliency of.

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
        :rtype: (numpy.ndarray, numpy.ndarray)
        """
