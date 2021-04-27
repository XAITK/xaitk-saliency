import pkg_resources


# It is known that this will fail if SMQTK-Core is not "installed" in the
# current environment. Additional support is pending defined use-case-driven
# requirements.
__version__ = pkg_resources.get_distribution(__name__).version
