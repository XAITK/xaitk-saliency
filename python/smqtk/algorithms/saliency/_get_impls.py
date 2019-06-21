from ._interface import SaliencyBlackbox, ImageSaliencyAugmenter
from smqtk.utils import plugin


def get_saliency_blackbox_impls(reload_modules=False):
    """
    Discover and return discovered ``SaliencyBlackbox`` classes. Keys in the returned
    map are the names of the discovered classes, and the paired values are the
    actual class type objects.

    We search for implementation classes in:
        - modules next to this file this function is defined in (ones that
          begin with an alphanumeric character),
        - python modules listed in the environment variable
          :envvar:`SALIENCY_BLACKBOX_PATH`
            - This variable should contain a sequence of python module
              specifications, separated by the platform specific PATH separator
              character (``;`` for Windows, ``:`` for unix)

    Within a module we first look for a helper variable by the name
    ``SALIENCY_BLACKBOX_CLASS``, which can either be a single class object or
    an iterable of class objects, to be specifically exported. If the variable
    is set to None, we skip that module and do not import anything. If the
    variable is not present, we look at attributes defined in that module for
    classes that descend from the given base class type. If none of the above
    are found, or if an exception occurs, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class object of type :class:`SaliencyBlackbox`
        whose keys are the string names of the classes.
    :rtype: dict[str, type]

    """
    this_dir = os.path.abspath(os.path.dirname(__file__))
    env_var = "SALIENCY_BLACKBOX_PATH"
    helper_var = "SALIENCY_BLACKBOX_CLASS"
    base_class = SaliencyBlackbox
    # __package__ resolves to the containing module of this module, or
    # `smqtk.algorithms.saliency` in this case.
    return plugin.get_plugins(__package__, this_dir, env_var, helper_var,
                              base_class, reload_modules=reload_modules)


def get_image_saliency_augmenter_impls(reload_modules=False):
    """
    Discover and return discovered ``ImageSaliencyAugmenter`` classes. Keys in
    the returned map are the names of the discovered classes, and the paired
    values are the actual class type objects.

    We search for implementation classes in:
        - modules next to this file this function is defined in (ones that
          begin with an alphanumeric character),
        - python modules listed in the environment variable
          :envvar:`IMG_SALIENCY_AUGMENTER_PATH`
            - This variable should contain a sequence of python module
              specifications, separated by the platform specific PATH separator
              character (``;`` for Windows, ``:`` for unix)

    Within a module we first look for a helper variable by the name
    ``IMG_SALIENCY_AUGMENTER_CLASS``, which can either be a single class object
    or an iterable of class objects, to be specifically exported. If the
    variable is set to None, we skip that module and do not import anything. If
    the variable is not present, we look at attributes defined in that module
    for classes that descend from the given base class type. If none of the
    above are found, or if an exception occurs, the module is skipped.

    :param reload_modules: Explicitly reload discovered modules from source.
    :type reload_modules: bool

    :return: Map of discovered class object of type
        :class:`ImageSaliencyAugmenter` whose keys are the string names of the
        classes.
    :rtype: dict[str, type]

    """
    this_dir = os.path.abspath(os.path.dirname(__file__))
    env_var = "IMG_SALIENCY_AUGMENTER_PATH"
    helper_var = "IMG_SALIENCY_AUGMENTER_CLASS"
    base_class = ImageSaliencyAugmenter
    # __package__ resolves to the containing module of this module, or
    # `smqtk.algorithms.saliency` in this case.
    return plugin.get_plugins(__package__, this_dir, env_var, helper_var,
                              base_class, reload_modules=reload_modules)

