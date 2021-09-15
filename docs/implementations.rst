=========================
Saliency Implementations
=========================

Included with this toolkit are a number of implementations for the interfaces
described in the previous section.
Like those interfaces, these initial implementations focus of black-box visual
saliency and are separated by use-case.
Each implementation provides functionality for a different perturbation or
heatmap generation algorithm specific to that use-case.

-------------------
Image Perturbation
-------------------

Class: RISEGrid
----------------
.. autoclass:: xaitk_saliency.impls.perturb_image.rise.RISEGrid
  :members:

Class: SlidingWindow
---------------------
.. autoclass:: xaitk_saliency.impls.perturb_image.sliding_window.SlidingWindow
  :members:

-------------------
Heatmap Generation
-------------------

Class: OcclusionScoring
------------------------
.. autoclass:: xaitk_saliency.impls.gen_classifier_conf_sal.occlusion_scoring.OcclusionScoring
  :members:

Class: RISEScoring
-------------------
.. autoclass:: xaitk_saliency.impls.gen_classifier_conf_sal.rise_scoring.RISEScoring
  :members:

Class: SimilarityScoring
-------------------------
.. autoclass:: xaitk_saliency.impls.gen_descriptor_sim_sal.similarity_scoring.SimilarityScoring
  :members:

Class: DRISEScoring
--------------------
.. autoclass:: xaitk_saliency.impls.gen_detector_prop_sal.drise_scoring.DRISEScoring
  :members:

---------------
Combined Stack
---------------

Class: PerturbationOcclusion
-----------------------------
.. autoclass:: xaitk_saliency.impls.gen_image_classifier_blackbox_sal.occlusion_based.PerturbationOcclusion
  :members:

Class: RISEStack
-----------------
.. autoclass:: xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise.RISEStack
  :members:

Class: SlidingWindowStack
--------------------------
.. autoclass:: xaitk_saliency.impls.gen_image_classifier_blackbox_sal.slidingwindow.SlidingWindowStack
  :members:
