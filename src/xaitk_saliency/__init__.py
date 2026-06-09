"""Define the xaitk-saliency package"""

from importlib import metadata

from .interfaces.gen_classifier_conf_sal import GenerateClassifierConfidenceSaliency  # noqa: F401
from .interfaces.gen_descriptor_sim_sal import GenerateDescriptorSimilaritySaliency  # noqa: F401
from .interfaces.gen_detector_prop_sal import GenerateDetectorProposalSaliency  # noqa: F401
from .interfaces.gen_image_classifier_blackbox_sal import GenerateImageClassifierBlackboxSaliency  # noqa: F401
from .interfaces.gen_image_similarity_blackbox_sal import GenerateImageSimilarityBlackboxSaliency  # noqa: F401
from .interfaces.gen_object_detector_blackbox_sal import GenerateObjectDetectorBlackboxSaliency  # noqa: F401
from .interfaces.perturb_image import PerturbImage  # noqa: F401

__version__ = metadata.version("xaitk-saliency")
