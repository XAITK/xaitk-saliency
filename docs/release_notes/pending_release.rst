Pending Release Notes
=====================


Updates / New Features
----------------------

Interfaces

* Update `PerturbImage` to only output perturbation masks, dropping physical
  image perturbation output. Since this output generation was the same across
  all known implementations, that part has been split out into a utility
  function.

* Added support for positive and negative saliency values as output by the
  saliency map generation interfaces.

Implementations

* Add ``RISEScoring`` and ``DebiasedRISEScoring`` implementations, along
  with new ``weight_regions_by_scalar_rise`` functionality.

Utils

* Masking

  * Added utility functions for occluded image generation that was previously
    duplicated across `PerturbImage` implementations. Added both batch and
    streaming versions of this utility.


Fixes
-----

Implementations

* Fix saliency map normalization in both ``OcclusionScoring`` as well as
  ``SimilarityScoring`` to disallow cross-class pollution in the norm.
