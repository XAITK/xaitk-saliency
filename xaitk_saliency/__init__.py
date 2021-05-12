import pkg_resources

from .interfaces.perturb_image import PerturbImage  # noqa: F401
from .interfaces.vis_sal_similarity import ImageSimilaritySaliencyMapGenerator  # noqa: F401
from .interfaces.vis_sal_classifier import ImageClassifierSaliencyMapGenerator  # noqa: F401


# It is known that this will fail if SMQTK-Core is not "installed" in the
# current environment. Additional support is pending defined use-case-driven
# requirements.
__version__ = pkg_resources.get_distribution(__name__).version
