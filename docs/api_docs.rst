==============================================
API
==============================================

The xaitk-saliency API consists of a number of object-oriented functor interfaces for saliency heatmap generation.
These initial interfaces focus on black-box visual saliency.
We define the two high-level requirements for this initial task: reference image perturbation in preparation for
black-box testing, and saliency heatmap generation utilizing black-box inputs.
We define a few similar interfaces for performing the saliency heatmap generation, separated by the intermediate
algorithmic use cases: image similarity, classification, and object detection.
We explicitly do not require an abstraction for the black-box operations to fit inside.
This is intended to allow for applications using these interfaces while leveraging existing functionality, which only
need to perform data formatting to fit the input defined here.
Note, however, that some interfaces are defined for certain black-box concepts as part of the SMQTK ecosystem (e.g.
in `SMQTK-Descriptors <https://github.com/Kitware/SMQTK-Descriptors>`_, `SMQTK-Classifier <https://github
.com/Kitware/SMQTK-Classifier>`_, `SMQTK-Relevancy <https://github.com/Kitware/SMQTK-Relevancy>`_, and other SMQTK-*
modules).


These interfaces are based on the plugin and configuration features provided by SMQTK-Core to allow convenient hooks
into implementation, discoverability, and factory generation from runtime configuration.
This allows for both opaque discovery of interface implementations from a class-method on the interface class object,
as well as instantiation of a concrete instance via a JSON-like configuration fed in from an outside resource.

.. figure:: figures/api-docs-fig-01.png

   Figure 1: Abstract Interface Inheritance.

---------------------------
Perturbed Image Generation
---------------------------

The PerturbImage interface abstracts the behavior of taking a reference image and generating some number perturbations
of the image along with paired mask matrices that indicate where perturbations have occurred and to what amount.

Implementations should impart no side effects on the input image.

Immediate candidates for implementation of this interface are occlusion-based saliency algorithms [3] that perform
perturbations on image pixels.

Interface: PerturbImage
------------------------
.. autoclass:: xaitk_saliency.interfaces.perturb_image.PerturbImage
   :members:

-----------------------------------------
Image Occlusion via Perturbation Masks
-----------------------------------------

A common intermediate step in this process is applying the generated perturbation masks to imagery to produce occluded
images.
We provide two utility functions as baseline implementation to perform this step:

* ``xaitk_saliency.utils.masking.occlude_image_batch`` - performs the transformation as a batch operation

* ``xaitk_saliency.utils.masking.occlude_image_streaming`` - performs the transformation in a streaming method with
  optional parallelization in that streaming

While the batch version is simpler and in many cases the faster of the two versions, the streaming version may be more
applicable to large image masks or when a great deal of masks are being input, where in such cases the batch version
would exceed available memory.

.. autofunction:: xaitk_saliency.utils.masking.occlude_image_batch

.. autofunction:: xaitk_saliency.utils.masking.occlude_image_streaming


----------------------------
Saliency Heatmap Generation
----------------------------

These interfaces comprise a family of siblings that all perform a similar transformation, but require different
standard inputs.
There is no standard to rule them all without being so abstract that it would break the concept of interface
abstraction, or the ability to substitute any arbitrary implementations of the interface without interrupting successful
execution.
Each interface is intended to handle different black-box outputs from different algorithmic categories.
In the future, as additional algorithmic categories are identified for which saliency map generation is applicable,
additional interfaces may be defined and added to this initial repertoire.

Interface: GenerateDescriptorSimilaritySaliency
-----------------------------------------------

This interface proposes that implementations require externally generated feature-vectors for two reference images
between which we are trying to discern the feature-space saliency.
This also requires the feature-vectors for perturbed images as well as the masks of the perturbations as would be
output from a ``PerturbImage`` implementation.
We expect perturbations to be relative to the second reference image feature-vector.

An immediate candidate implementation for this interface is the SBSM algorithm [1].

.. autoclass:: xaitk_saliency.interfaces.gen_descriptor_sim_sal.GenerateDescriptorSimilaritySaliency
   :members:

Interface: GenerateClassifierConfidenceSaliency
------------------------------------------------

This interface proposes that implementations transform black-box image classification scores into saliency heatmaps.
This should require a sequence of per-class confidences predicted on the reference image, a number of per-class
confidences as predicted on perturbed images, as well as the masks of the reference image perturbations (as would be
output from a ``PerturbImage`` implementation).

Implementations should use this input to generate a visual saliency heatmap for each input “class” in the input. This
is both an effort to vectorize the operation for optimal performance, as well as to allow some algorithms to take
advantage of differences in classification behavior for other classes to influence heatmap generation.
For classifiers that generate many class label predictions, it is intended that only a subset of relevant class
predictions need be provided here if computational performance is a consideration.

An immediate candidate implementation for this interface is the RISE algorithm [2] and occlusion-based saliency
algorithms [3] that generate saliency heatmaps.

.. autoclass:: xaitk_saliency.interfaces.gen_classifier_conf_sal.GenerateClassifierConfidenceSaliency
   :members:

Interface: GenerateDetectorProposalSaliency
-------------------------------------------

This interface proposes that implementations transform black-box image object detection predictions into visual
saliency heatmaps.
This should require externally generated object detection predictions over some image, along with predictions for
perturbed images and the permutation masks for those images as would be output from a ``PerturbImage`` implementation.
Object detection representations used here would need to encapsulate localization information (i.e. bounding box
regions), class scores, and objectness scores (if applicable to the detector, such as YOLOv3).
Object detections are converted into (4+1+nClasses) vectors (4 indices for bounding box locations, 1 index for
objectness, and nClasses indices for different object classes).

Implementations should use this input to generate a visual saliency heatmap for each input detection.
We assume that an input detection is coupled with a single truth class (or a single leaf node in a hierarchical
structure).
Input detections on the reference image may be drawn from ground truth or predictions as desired by the use case.
As for perturbed image detections, we expect those to usually be decoupled from the source of reference image
detections, which is why below we formulate the shape of perturbed image detects with ``nProps`` instead of ``nDets``
(though the value of that axis may be the same in some cases).

A candidate implementation for this interface is the D-RISE [4] algorithm.

.. autoclass:: xaitk_saliency.interfaces.gen_detector_prop_sal.GenerateDetectorProposalSaliency
   :members:

Detection formatting helper
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :py:meth:`.GenerateDetectorProposalSaliency.generate` method takes in a
specifically formatted matrix that combines 3 different aspects of common
detector model outputs:
  * bounding boxes
  * objectness scores
  * classification scores

We provide a helper function to merge distinct output data into the unified
format.

.. autofunction:: xaitk_saliency.utils.detection.format_detection

---------------------------------------------
Black-Box Saliency Image Generation
---------------------------------------------

Unlike the previous saliency heatmap generation interfaces, this interface uses a black-box classifier as input along
with a reference image to generate visual saliency heatmaps.

A candidate implementation for this interface is the ``PerturbationOcclusion`` implementation or one of its
sub-implementations (``RISEStack`` or ``SlidingWindowStack``).


Interface: GenerateImageClassifierBlackboxSaliency
---------------------------------------------------

.. autoclass:: xaitk_saliency.interfaces.gen_image_classifier_blackbox_sal.GenerateImageClassifierBlackboxSaliency
   :members:

------------------
Code Examples
------------------

Generating Perturbed Images and Masks::

    import PIL.Image
    import numpy as np
    import numpy.typing as npt
    from xaitk_saliency import PerturbImage
    from xaitk_saliency.utils.masking import occlude_image_batch


    # Define an implementation, or use a discovered plugin.
    # This does not need to be defined in-line, but may be instead
    # imported from some alternative module, or found via plugin
    # discovery.
    class PerturbImageImplementation (PerturbImage):
       def perturb(
           self,
           ref_image: npt.ArrayLike
       ) -> np.ndarray:
           ...


    ...

    perturb_image = PerturbImageImplementation()

    ...

    test_image = np.asarray(PIL.Image.open("some/test/image.png"))

    # Generate perturbed images and perturbation masks
    mask_array = perturb_image(test_image)
    perturbed_images = occlude_image_batch(test_image, mask_array)

    # Returned sequences should be congruent in length.
    assert perturbed_images.shape == mask_array.shape

    # Do application-appropriate things with the pairs!
    for img, mask in zip(image_seq, mask_array):
       render(img, mask)

Generating Similarity-based Saliency Heatmaps::

    import PIL.Image
    import numpy as np
    from xaitk_saliency import PerturbImage
    from xaitk_saliency import GenerateDescriptorSimilaritySaliency
    from xaitk_saliency.utils.masking import occlude_image_batch
    from MyIntegration import describe_images  # type: ignore


    # Pretend we have implementations of the standard interfaces.
    class PerturbImageImplementation (PerturbImage):
       ...


     class GenerateDescriptorSimilaritySaliencyImplementation (GenerateDescriptorSimilaritySaliency):
       ...


    # Initializing an implementation of perturbation-based algorithms
    perturb_image = PerturbImageImplementation()

    # Initializing an implementation of similarity-based saliency generator
    similarity_saliency = GenerateDescriptorSimilaritySaliencyImplementation()

    ...

    # Loading test image1 from file
    test_image_1 = np.asarray(PIL.Image.open("some/test/image1.png"))
    # Loading reference image 2 from file
    ref_image_2 = np.asarray(PIL.Image.open("some/test/image2.png"))

    # Generate perturbed images and perturbation masks on reference image on which
    # saliency needs to be computed.
    mask_array = perturb_image(ref_image_2)
    perturbed_images = occlude_image_batch(ref_image_2, mask_array)

    # Compute descriptors for the test, reference and perturbed image.
    # This part may be specific to your application or integration.
    # The output here is expected to be in the shape [nInputs x nFeats].
    test_img_descr, ref_img_descr = describe_images([test_image_1, ref_image_2])
    perturb_descr = describe_images(perturbed_images)

    # Compute the final similarity based-saliency map using original features from
    # both the test and reference images, along with descriptors computed on the
    # perturbed versions of the reference image and masks used to perturb the
    # reference image
    similarity_saliency_map = similarity_saliency(
      test_img_descr,  # shape: [nFeats]
      ref_img_descr,  # shape: [nFeats]
      perturb_descr,  # shape: [len(perturbed_images), nFeats]
      mask_array  # shape: [len(perturbed_images), ref_image_2.height, ref_image_2.width]
    )
    # The shape of the output heatmap should be congruent to the shape of input
    # perturbation masks.
    assert similarity_saliency_map.shape == mask_array[0].shape

Generating Classification-based Saliency Heatmaps::

    import PIL.Image
    import numpy as np
    from xaitk_saliency import PerturbImage
    from xaitk_saliency import GenerateClassifierConfidenceSaliency
    from xaitk_saliency.utils.masking import occlude_image_batch
    from MyIntegration import classify_images  # type: ignore


    # Pretend we have implementations of the standard interfaces.
    class PerturbImageImplementation (PerturbImage):
      ...


    class GenerateClassifierConfidenceSaliencyImplementation (GenerateClassifierConfidenceSaliency):
      ...


    # Initializing an implementation of perturbation-based algorithms
    perturb_image = PerturbImageImplementation()

    # Initializing an implementation of classifier-based saliency generator
    classifier_saliency = GenerateClassifierConfidenceSaliencyImplementation()

    ...

    # Loading reference image from file
    ref_image = np.asarray(PIL.Image.open("some/test/image.png"))

    # Generate perturbed images and perturbation masks on
    # reference image on which saliency needs to be computed
    mask_array = perturb_image(ref_image)
    perturbed_images = occlude_image_batch(ref_image, mask_array)

    # Compute class confidence predictions for reference and perturbed images.
    # We assume for this example that this black-box image classification function
    # returns a matrix of class label confidences with different class labels
    # corresponding to different columns of the output matrix, whose shape will be
    # [nInputs x nClasses].
    ref_class_confs = classify_images([ref_image])[0]
    perturbed_class_confs = classify_images(perturbed_images)

    # We will also show the example case where we do not want to pass along all
    # class confidences for saliency map generation, but only a select few.
    # Maybe this would be defined by some interface or configuration.
    pertinent_class_indices = [1, 4, 10]
    ref_class_confs2 = ref_class_confs[pertinent_class_indices]
    perturbed_class_confs2 = perturbed_class_confs[..., pertinent_class_indices]

    # Computing the final classifier-based saliency map using
    # classifier confidence on the original feature vector of an reference image
    # along with the classifier confidence on all descriptors computed on the
    # perturbed versions of the reference image and masks used to perturb the reference
    # image
    classifier_saliency_map = classifier_saliency(
      ref_class_confs2,  # shape: [len(pertinent_class_indices)]
      perturbed_class_confs2,  # shape: [len(perturbed_images), len(pertinent_class_indices)]
      mask_array  # shape: [len(perturbed_images), ref_image.height, ref_image.width]
    )
    # There should be an equal number of saliency maps output as the number of
    # distinct class confidences input:
    assert len(classifier_saliency_map) == len(pertinent_class_indices)
    # The shape of the output heatmap should be congruent to the shape of input
    # perturbation masks.
    assert classifier_saliency_map[0].shape == mask_array[0].shape


------------------
References
------------------

1. Dong B, Collins R, Hoogs A. Explainability for Content-Based Image Retrieval. InCVPR Workshops 2019 Jun (pp. 95-98).
2. Petsiuk V, Das A, Saenko K. Rise: Randomized input sampling for explanation of black-box models. arXiv preprint arXiv:1806.07421. 2018 Jun 19.
3. Zeiler MD, Fergus R. Visualizing and understanding convolutional networks (2013). arXiv preprint arXiv:1311.2901. 2013.
4. Petsiuk V, Jain R, Manjunatha V, Morariu VI, Mehra A, Ordonez V, Saenko K. Black-box explanation of object detectors via saliency maps. arXiv preprint arXiv:2006.03204. 2020 Jun 5.
