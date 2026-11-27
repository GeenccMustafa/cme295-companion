"""Repository-wide pytest configuration.

Registers Hypothesis profiles: ``default`` keeps local runs fast, ``ci`` runs
many more examples. Select one with the ``HYPOTHESIS_PROFILE`` environment
variable, for example ``HYPOTHESIS_PROFILE=ci uv run pytest``.
"""

import os

from hypothesis import settings

settings.register_profile("default", max_examples=100, deadline=None)
settings.register_profile("ci", max_examples=1000, deadline=None, print_blob=True)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))
