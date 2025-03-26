==============
How-To Guides
==============

This section provides practical examples that demonstrate how to apply XAITK-Saliency to a variety
of tasks, ranging from image classification and object detection to explainability for similarity
scoring and reinforcement learning agents. These are structured as Jupyter notebooks in the
``docs/examples/`` folder of the repository. Each example shows how to
accomplish a specific task using the provided tools and APIs (see the
Reference section topics :ref:`implementations` and :ref:`interfaces`).

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

Visualize and compare how different classifiers explain predictions.
`View notebook <examples/ModelComparisonWithSaliency.html>`__.

Object Detection
----------------

- **Generating Detection Saliency via Serialization**

Produce saliency maps for serialized detections in COCO format.
`View notebook <examples/SerializedDetectionSaliency.html>`__.

- **Applying Occlusion Saliency in VIAME**

Run occlusion-based saliency analysis in the VIAME marine species detection toolkit.
`View notebook <examples/VIAME_OcclusionSaliency.html>`__.

Advanced Saliency Techniques
----------------------------

- **Swapping Saliency Techniques in a Classification Pipeline**

How to modularize and switch between saliency map generation techniques within a reusable application workflow.
`View notebook <examples/SwappableImplementations.html>`__.

- **Estimating Saliency with Multi-Color RISE**

Use MC-RISE to generate saliency maps with uncertainty quantification.
`View notebook <examples/MC_RISE.html>`__.

- **Applying Radial Perturbations to Images**

Analyze model sensitivity by introducing radial perturbations.
`View notebook <examples/Radial_Image_Perturbation.html>`__.

- **Generating Superpixel-Based Saliency Maps**

Use superpixels as spatial units for generating interpretable saliency.
`View notebook <examples/SuperPixelSaliency.html>`__.

Other Applications
------------------

- **Applying Saliency to Atari Game Agents**

Visualize saliency on agents trained via deep reinforcement learning on Atari environments.
`View notebook <examples/atari_deepRL_saliency.html>`__.

- **Explaining Similarity Scores with Saliency**

Use saliency maps to interpret similarity scoring systems.
`View notebook <examples/SimilarityScoring.html>`__.

Related Resources
-----------------

If you need broader context or foundational theory, see:

* `Occlusion Saliency Tutorial <examples/OcclusionSaliency.html>`__ and `DRISE Tutorial <examples/DRISE.html>`__ –
  Step-by-step tutorials to get started
* :doc:`xaitk_explanation` and :doc:`design` – Conceptual guides to XAITK-Saliency's architecture and approach
