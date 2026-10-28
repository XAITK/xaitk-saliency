* Ported the new import guard structure over from ``nrtk``.

* Migrated ``KWCocoUtils`` to use a new guard mechanism. Importing it without the ``tools`` extra now
  raises an ``ImportError`` naming the extra, and ``KWCocoUtils.is_usable()`` has been removed.

* Documented the import guard structure and conventions in ``docs/design.rst``.

* Added import guard tests.

* Added a ``tox -e optional`` environment that installs every extra and runs all non-notebook tests.
