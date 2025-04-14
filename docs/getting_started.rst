Getting Started with xaitk-saliency
===================================

The xaitk-saliency package is an explainable AI (XAI) framework and toolkit for visual saliency algorithm interfaces
and implementations, built for analytics and autonomy applications.

Saliency maps provide a visual form of explanation by (typically) overlaying a heatmap on the input, highlighting
regions that the AI model deems "important" for its predictions.

Saliency methods are typically categorized as:

* **White-box:** Requiring access to the internal state of the AI model.

* **Black-box:** Operating without any knowledge of the model internals.

Because black-box methods are often better suited to testing and evaluation (T&E) scenarios—where internal model
access may be restricted—**xaitk-saliency prioritizes black-box saliency techniques.**

Example: A First Look at xaitk-saliency
---------------------------------------
This `associated project <https://github.com/XAITK/xaitk-saliency-web-demo>`_ features a local web application that
demonstrates visual saliency generation through a user interface (UI). It provides an example of how saliency
maps produced byxaitk-saliencycan be integrated into a UI to support model prediction exploration and reasoning.
The application is built using the `trame framework <https://kitware.github.io/trame/>`_.

Gallery
^^^^^^^

.. |image1| image:: https://raw.githubusercontent.com/XAITK/xaitk-saliency-web-demo/main/gallery/xaitk-classification-rise-4.jpg
    :width: 45%

.. |image2| image:: https://raw.githubusercontent.com/XAITK/xaitk-saliency-web-demo/main/gallery/xaitk-classification-sliding-window.jpg
    :width: 45%

.. |image3| image:: https://raw.githubusercontent.com/XAITK/xaitk-saliency-web-demo/main/gallery/xaitk-detection-retina.jpg
    :width: 45%

.. |image4| image:: https://raw.githubusercontent.com/XAITK/xaitk-saliency-web-demo/main/gallery/xaitk-similarity-1.jpg
    :width: 45%

|image1| |image2|

|image3| |image4|

Next Steps
----------

To learn more about xaitk-saliency, read the :doc:`Overview <./introduction>` or dive right into a
:doc:`Tutorial <./xaitk_tutorial>`.
