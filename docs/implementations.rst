===============
Implementations
===============

Included with this toolkit are a number of implementations for the interfaces
described in the previous section.
Unlike the interfaces, which declare operation and use case, implementations
provide variations on *how* to satisfy the interface-defined use case, varying
trade-offs, or results implications.

.. When adding new classes within implementations, sort them alphabetically.

------------------
Image Perturbation
------------------

.. autosummary::
    :toctree: _implementations
    :template: custom-module-template.rst
    :recursive:

    xaitk_saliency.impls.perturb_image.mc_rise
    xaitk_saliency.impls.perturb_image.random_grid
    xaitk_saliency.impls.perturb_image.rise
    xaitk_saliency.impls.perturb_image.sliding_radial
    xaitk_saliency.impls.perturb_image.sliding_window

------------------
Heatmap Generation
------------------

.. autosummary::
    :toctree: _implementations
    :template: custom-module-template.rst
    :recursive:

    xaitk_saliency.impls.gen_detector_prop_sal.drise_scoring
    xaitk_saliency.impls.gen_classifier_conf_sal.mc_rise_scoring
    xaitk_saliency.impls.gen_classifier_conf_sal.occlusion_scoring
    xaitk_saliency.impls.gen_classifier_conf_sal.rise_scoring
    xaitk_saliency.impls.gen_descriptor_sim_sal.similarity_scoring
    xaitk_saliency.impls.gen_classifier_conf_sal.squared_difference_scoring

------------------------------
End-to-End Saliency Generation
------------------------------

Image Classification
--------------------

.. autosummary::
    :toctree: _implementations
    :template: custom-module-template.rst
    :recursive:

    xaitk_saliency.impls.gen_image_classifier_blackbox_sal.mc_rise
    xaitk_saliency.impls.gen_image_classifier_blackbox_sal.occlusion_based
    xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise
    xaitk_saliency.impls.gen_image_classifier_blackbox_sal.slidingwindow

Image Similarity
----------------

.. autosummary::
    :toctree: _implementations
    :template: custom-module-template.rst
    :recursive:

    xaitk_saliency.impls.gen_image_similarity_blackbox_sal.occlusion_based
    xaitk_saliency.impls.gen_image_similarity_blackbox_sal.sbsm

Object Detection
----------------

.. autosummary::
    :toctree: _implementations
    :template: custom-module-template.rst
    :recursive:

    xaitk_saliency.impls.gen_object_detector_blackbox_sal.occlusion_based
    xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise
