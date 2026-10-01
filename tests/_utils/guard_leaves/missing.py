"""A leaf that behaves as if its optional dependency were not installed."""

raise ImportError("No module named 'not_a_real_dependency'")
