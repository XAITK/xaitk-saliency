Introduction
============

The XAITK-Saliency package implements a class of XAI algorithms known
as `saliency algorithms`. The basic machine learning application pipeline is shown in Figure 1:

.. figure:: figures/intro-fig-01.png

   Figure 1: A basic AI pipeline.

In this scenario, an AI algorithm operates on an input (text, image,
etc.) to produce some sort of output (classification, detection, etc.) Saliency algorithms build on
this to produce visual explanations in the form of saliency maps as shown in Figure 2:

.. figure:: figures/intro-fig-02.png

   Figure 2: The AI pipeline augmented with a saliency algorithm.

At a high level, saliency maps are typically colored heatmaps applied
to the input, highlighting regions that are somehow significant to
the AI. Figure 3 shows sample saliency maps for text and images.

.. figure:: figures/intro-fig-03.png

   Figure 3: Sample saliency maps for text (left, from `Tuckey et al.
   <https://arxiv.org/abs/1907.05664>`_) and images (right, from `Dong et
   al. <https://openaccess.thecvf.com/content_CVPRW_2019/html/Explainable_AI/Dong_Explainability_for_Content-Based_Image_Retrieval_CVPRW_2019_paper.html>`_)

.. note:: The XAITK-Saliency toolkit currently focuses on providing saliency
          maps for images.

Image Saliency Maps: An Intuitive Introduction
----------------------------------------------

Figure 4 shows a deep learning pipeline for recognizing objects in
images; pixels in the image (in green) are processed by the pipeline
to produce the output (in orange). Here, the system has been trained
to recognize 1000 object categories. Its output is a list of 1000 numbers,
one for each object type; each number is between 0 and 1, representing
the system's estimate of whether or not that particular object is in
the image.

.. figure:: figures/cnn-tagged-scaled.png

   Figure 4: Typical object recognition CNN architecture; the image
   (green, left) is processed left to right through the CNN to produce an
   output vector (orange, right). Diagram from `Krizhevsky et
   al. <https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf>`_


This operation is "all or nothing" at both ends: the entire image must
be processed, and the the entire vector must be output. There is no
mechanism by which a subset of the output can be traced back to a
subset of the input. Yet it seems reasonable to ask questions such as:

* The input image is a camel; why did the system give a higher
  likelihood to "horse" than "camel"?

* The input image is a kitchen, but the system gave a high likelihood
  to "beach ball". What parts of the image were responsible for this?

* The input image contains two dogs, and the system gave a high
  likelihood for "dog". How would this change if one of the dogs
  wasn't in the image?

* The input image contains a dog and a cat, and the system gave a high
  likelihood for "cat" but not "dog". How will the system respond if
  the cat is removed?

At some level, these questions require a degree of *introspection*;
the system must produce not only the output, but also some information
about **how** the output was produced. There are two popular
approaches to this:

1) The **white box** approach: the system is altered to open or expose
   the internal state of the model. This state is examined while the
   system is generating the output.

2) The **black box** approach: the model is not opened; instead, we
   create a series of *related images* which perturb or change the
   original image in some way. By comparing the original output to the
   output for the related images, we can deduce certain aspects of the
   model's behavior.

Let's take a look at the pros and cons of these two approaches.

White Box Methods
^^^^^^^^^^^^^^^^^

*Explanation options are correlated to how the model operation can be
inspected and interpreted. pros: closely tied to the individual model, thus can leverage
specific knowledge about its architecture. Measures the actual
computation which generates the output. Cons: specific to individual
models / classes of models; harder to generalize across models.
Requires modifying the model implementation to gain access; may
require updating as the implementation evolves.*

Black Box Methods
^^^^^^^^^^^^^^^^^
*Explanation options are correlated to how the related input are
generated. pros: indpendent of the model; operates across all models; does not
require access to the model implementation. cons: requires extra work
to generate and process the related images; makes only indirect /
differential observations about the original input / output pair;
generally more resource intensive than white-box.*

XAITK-Saliency Map Algorithms
--------------------------------
*Discuss the provided XAITK-Saliency algorithms in terms of the above.*
