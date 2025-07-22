==============
How-To Guides
==============

This section provides practical, task-oriented examples demonstrating how to apply XAITK-Saliency
across a range of domains—from image classification and object detection to explainability in
similarity scoring and reinforcement learning. These guides are implemented as Jupyter notebooks
located in the ``docs/examples/`` folder of the repository.

Each notebook walks through how to accomplish a specific task using XAITK-Saliency's tools and APIs.
For further detail on these APIs, refer to the Reference topics :ref:`implementations` and
:ref:`interfaces`.

.. toctree::
   :hidden:

    examples/covid_classification.ipynb
    examples/MNIST_scikit_saliency.ipynb
    examples/ModelComparisonWithSaliency.ipynb
    examples/SerializedDetectionSaliency.ipynb
    examples/VIAME_OcclusionSaliency.ipynb
    examples/SwappableImplementations.ipynb
    examples/MC_RISE.ipynb
    examples/Radial_Image_Perturbation.ipynb
    examples/SuperPixelSaliency.ipynb
    examples/atari_deepRL_saliency.ipynb
    examples/SimilarityScoring.ipynb
    examples/OcclusionSaliency.ipynb
    examples/DRISE.ipynb

Image Classification
--------------------

- **Classifying COVID-19 in Chest X-rays**

  Interpret model predictions on chest X-ray images using saliency maps.
  `View notebook <examples/covid_classification.html>`__.

- **Generating Saliency for MNIST with scikit-learn**

  Apply saliency techniques to scikit-learn classifiers on the MNIST dataset.
  `View notebook <examples/MNIST_scikit_saliency.html>`__.

- **Comparing Saliency Across Models**

  Visualize and compare explanations from different classifiers.
  `View notebook <examples/ModelComparisonWithSaliency.html>`__.

Object Detection
----------------

- **Generating Detection Saliency via Serialization**

  Produce saliency maps for serialized detections in COCO format.
  `View notebook <examples/SerializedDetectionSaliency.html>`__.

- **Applying Occlusion Saliency in VIAME**

  Perform occlusion-based saliency analysis for classifying marine species
  with the VIAME toolkit.
  `View notebook <examples/VIAME_OcclusionSaliency.html>`__.

Advanced Saliency Techniques
----------------------------

- **Swapping Saliency Techniques in a Classification Pipeline**

  Modularize and switch between saliency methods in an application workflow.
  `View notebook <examples/SwappableImplementations.html>`__.

- **Estimating Saliency with Multi-Color RISE**

  Generate saliency maps with uncertainty quantification using MC-RISE.
  `View notebook <examples/MC_RISE.html>`__.

- **Applying Radial Perturbations to Images**

  Analyze model sensitivity by applying radial perturbations to input images.
  `View notebook <examples/Radial_Image_Perturbation.html>`__.

- **Generating Superpixel-Based Saliency Maps**

  Use superpixels as spatial units for interpretable saliency mapping.
  `View notebook <examples/SuperPixelSaliency.html>`__.

Other Applications
------------------

- **Applying Saliency to Atari Game Agents**

  Visualize saliency in deep reinforcement learning agents trained on Atari games.
  `View notebook <examples/atari_deepRL_saliency.html>`__.

- **Explaining Similarity Scores with Saliency**

  Use saliency maps to interpret similarity scoring systems.
  `View notebook <examples/SimilarityScoring.html>`__.

Related Resources
-----------------

For step-by-step walkthroughs and foundational concepts, refer to:

- `Occlusion Saliency Tutorial <examples/OcclusionSaliency.html>`__
- `DRISE Tutorial <examples/DRISE.html>`__
- :doc:`xaitk_explanation` and :doc:`design` – Conceptual guides to XAITK-Saliency's architecture and
  philosophy
