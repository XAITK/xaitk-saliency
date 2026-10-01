Design Decisions
================

Concrete Dependencies and Updating
----------------------------------
This package, like many others, has dependencies that need to be fulfilled for
proper functionality to be satisfied.
While these general requirements, found in the :file:`pyproject.toml` file, are
known as "abstract" requirements, we also choose to codify "concrete"
requirements here via the :file:`poetry.lock` file.
This distinction is described well by [Stufft]_, using the ``setuptools``-based
resources which are parallel in use to the above respectively described files.

While xaitk-saliency is a library, we choose to retain concrete
dependencies via the :file:`poetry.lock` file to maintain consistency
of environment across developers as well as CI processes.
This falls in conceptual line with the "Developing Reusable Things or How Not
to Repeat Yourself" section from [Stufft]_.
However, we are still a "library" and desire to make sure that we work with the
"latest" versions of our listed abstract dependencies (within some reasonable
time window).
Currently, such concrete "version bumps" happen in the form of periodic update
branches that update the :file:`poetry.lock` file via a ``poetry update`` call.
These updates are submitted as PRs to the upstream repository and allow the
standard suite of CI checks to be performed to make sure the updated versions
do not break anything.
The timing of such updates are currently not concretely scheduled, nor are they
specifically tied to events, but more on an "every so often" cadence that is
relatively more frequent than versioned releases.

Image Format
------------
We choose to use the ``numpy.ndarray`` data structure for our image
representation in this toolkit.
Earlier, we utilized the Pillow package's ``PIL.Image.Image`` data structure
but encountered issues in certain use cases regarding large images, images with
non-standard quantities of channels (e.g. > 3) or with imagery consisting of
12 or 16-bit valuation.
Additionally, other popular and highly utilized packages in the Python
community, like OpenCV, Scikit-Image, and PyTorch to name a few, utilize raw
``numpy.ndarray`` matrices as the container for image data.

.. [Stufft] https://caremad.io/posts/2013/07/setup-vs-requirement/

Import Guards
-------------
Some xaitk-saliency implementations are conditional and need an optional dependency (an
"extra"). The class is published from a public module and reported under that path, never
under the private module that defines it. A generated default config (``make_default_config``) records

``xaitk_saliency.interop.maite.object_detection.MAITEDetector``

rather than

``xaitk_saliency.interop.maite.object_detection._model.MAITEDetector``

``xaitk_saliency._guard.guard()`` provides all of this functionality. Every module that exposes an
implementation (needing an extra) calls it exactly once; apart from the ``if TYPE_CHECKING:`` block and
the imports the call itself needs, the module does nothing else at import time.

.. code-block:: python

    from collections.abc import Callable
    from typing import TYPE_CHECKING, Any

    from xaitk_saliency._guard import Group, guard

    if TYPE_CHECKING:
        from xaitk_saliency.interop.maite.object_detection._model import MAITEDetector as MAITEDetector

    __getattr__: Callable[[str], Any]
    __dir__: Callable[[], list[str]]
    __all__: list[str]

    __getattr__, __dir__, __all__ = guard(
        namespace=globals(),
        groups=[
            Group(
                symbols={"MAITEDetector": "xaitk_saliency.interop.maite.object_detection._model"},
                extras=["maite"],
            ),
        ],
    )

Terminology
^^^^^^^^^^^

**package**
    A directory with an ``__init__.py``. A *public* package is one users import
    from; a *private* one starts with an underscore and exists only to hold leaf
    modules.

**leaf module**
    The ``.py`` file that actually defines a class, and what ``symbols`` points at.
    Never a package. xaitk-saliency's leaves are private modules inside a public
    package. Usually that package's ``__init__.py`` calls ``guard()``
    (``xaitk_saliency.interop.maite.image_classification`` fronts its ``_model``);
    the other layout is a public module fronting a sibling leaf, the way
    ``xaitk_saliency.utils.coco`` fronts ``xaitk_saliency.utils._coco``.

**submodule**
    A module named relative to the package containing it, which is why ``guard()``
    takes ``submodules=``. Use it only when the parent is the point; otherwise say
    module.

**group**
    One ``Group``: the symbols that share the same set of requirements. Add ``extras=``
    as the requirements need. Multiple extras is still one group —
    ``extras=["maite", "tools"]`` means both are required, not that two groups are
    needed. A module declares as many groups as it has distinct sets of
    requirements.

Adding an implementation
^^^^^^^^^^^^^^^^^^^^^^^^

Add the class to the ``symbols`` of the group with matching requirements or create
a new group, if existing ones do not satisfy requirements, and add it to the ``if TYPE_CHECKING:`` block.

``symbols`` maps the public name to the leaf module. Write that path out in full;
hoist a shared prefix into a ``_CONSTANT`` only when the line would otherwise run
past the 120-character limit.

The ``if TYPE_CHECKING:`` block is not decoration. ``__getattr__: Callable[[str], Any]`` makes pyright
treat every attribute of a guarded module as ``Any``, so a symbol missing from the
block still imports, still type-checks, and still scores 100% on
``--verifytypes`` — it just silently loses its signature, and with it: argument
checking at every call site, ``@override`` resolution, and editor hover.
``tests/test_guard_declarations.py::test_type_checking_block_matches_the_declaration``
is the only thing that catches it, in either direction: a declared symbol missing
from the block, or a leftover in the block that is no longer declared. The bare
annotations above the ``guard()`` call matter too:

.. code-block:: python

    __getattr__: Callable[[str], Any]
    __dir__: Callable[[], list[str]]
    __all__: list[str]

``--verifytypes`` requires a declared type for every exported module-level symbol.
Without these it reports ``__getattr__`` and ``__dir__`` as ambiguous rather than
known, which drops the score below 100% and fails. ``__all__`` is not counted
either way; it is declared alongside them so the three read as one unit.

Describing extras
^^^^^^^^^^^^^^^^^

``extras`` is a list of all dependencies required for a ``Group``; a nested tuple indicates a
choice between two or more alternatives, i.e., one of the tuple alternatives must be available at runtime.


.. code-block:: python

    extras=["maite"]           # maite
    extras=["maite", "tools"]  # maite and tools
    extras=[("a", "b")]        # a or b
    extras=["c", ("a", "b")]   # c, plus either a or b

The guard renders these requirements into the error that the user sees, with one ``pip install`` per
possibility of meeting the requirements. Requirements are specified as a list rather than a string because
there is no pip syntax for the "and either of these" situation that nested tuples represent.

Registering modules for discovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Being importable is not being discoverable. Register the module that calls
``guard()`` and advertises the class — the same public path the class is
published under — never the leaf module and never a parent that only
re-exports.

xaitk-saliency has no example of this rule yet.

Rules
^^^^^

Two things the guard cannot check for you, because both are decisions you make
when you declare a group.

- **Point** ``symbols`` **at a leaf module, never at anything that runs its own
  guard.**

  .. code-block:: python

      # Right: the leaf module that defines the class.
      "MAITEDetector": "xaitk_saliency.interop.maite.object_detection._model"

      # Wrong: a public package, which runs its own guard() call.
      "MAITEDetector": "xaitk_saliency.interop.maite.object_detection"

  Pointing at a public package would require that package's own ``guard()`` call
  to resolve first, reintroducing the import-order dependence the guard exists to
  remove.

- **A private package's** ``__init__.py`` **holds nothing but a docstring.**
  Re-exporting an extras-dependent leaf creates a second, unguarded path to the
  class and breaks the compliance walk
  (``tests/scripts/guard/packages_import_without_extras.py``, run by
  ``tests/test_import_guards_e2e.py``) that imports every package with no extras
  installed.

What the guard guarantees
^^^^^^^^^^^^^^^^^^^^^^^^^
There is nothing to do about the following — it is what ``guard()`` is for, and no
declaration you can write will break it. It is recorded here because it explains
why the guard behaves the way it does.

``__dir__`` never advertises a name that ``__getattr__`` would raise on. SMQTK's
plugin discovery calls ``getattr`` on every name ``dir()`` returns and catches
nothing, so a single raising name would abort the entire discovery pass — every
implementation after it going undiscovered, not just the one that raised.
``_Guard.dir`` returns ``__all__``, and ``__all__`` only ever gains symbols that
actually resolved, so that state cannot arise.

Holding that guarantee is why symbols resolve eagerly, at ``guard()`` time: whether
an extra is installed cannot change mid-process, so a group whose extra is missing
is never advertised in the first place. Only declared submodules are imported
lazily, on first access.
