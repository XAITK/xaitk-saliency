Design Decisions
================

Image format
------------
We choose to use the `numpy.ndarray` data structure for our image
representation in this toolkit.
Earlier, we utilized the Pillow package's `PIL.Image.Image` data-structure but
encountered issues in certain use-cases regarding large images, images with
non-standard quantities of channels (e.g. > 3) or with imagery consisting of
12 or 16-bit valuation.
Additionally, other popular and highly utilized packages in the python
community, like OpenCV, Scikit-Image and PyTorch to name a few, utilize raw
`numpy.ndarray` matrices as the container for image data.
