Pending Release Notes
=====================

Updates / New Features
----------------------

Docker

* Update Dockerfile install of poetry, and make separate & specific directory
  copies.

Utils

* Updated logging format of occlusion masking benchmark utility.

Fixes
-----

Tests

* Fix various floating point equality comparisons to not use exact match.

* Fix random number usage from numpy to use `np.random.default_rng`.

* Fix error being masked in `test_sal_on_coco_dets`
