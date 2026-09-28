AUKUS Container
===============

A user of T&E tools requested a container that would accept an input image to generate
saliency maps. Once the maps are generated, the user expects the maps to be saved to disk
and the container will shut down. In order to fulfill this request, the AUKUS container was
created.

Given an input image, a XAITK saliency configuration file, and a HuggingFace model name,
the AUKUS container is able to find detections and generate saliency maps for the input image.
Each saliency map will be saved to a given output directory as individual images. Once all
saliency maps are saved, the container will terminate.

How to Use
----------
To run the AUKUS container, use the following command:
``docker run -v /path/to/input:/root/input/:ro -v /path/to/output:/root/output/ xaitk-saliency``
This will mount the inputs to the correct locations and use the default args. The default
args will attempt to load an image named ``example_img.jpeg``, save saliency maps to
``/root/output``, load a config file named ``config.json``, and load the
``facebook/detr-resnet-50`` model. The ``example_img.jpeg`` and ``config.json`` must be
in the directory mounted to ``/root/input/``.

If the user wants to use different arguments, the container expects the following
arguments:

   * ``image_file``: input image
   * ``output_dir``: directory to write saliency maps to
   * ``config_file``: configuration file specifying the ``GenerateObjectDetectorBlackboxSaliency`` for saliency map
     generation
   * ``hugging_face_model_name``: name of HuggingFace model to use

Please note the values for ``image_file`` and ``config_file`` should be written from the
perspective of the container (i.e. ``/path/on/container/image_file.jpeg`` instead of
``/path/on/local/machine/image_file.jpeg``)

Limitations
-----------

Currently, the main limitation of the AUKUS container is only being able to use HuggingFace
models for object detections. While this allows for some freedom of choice in model,
users will not only be limited to use HuggingFace models, but also need to access
HuggingFace models during execution.
