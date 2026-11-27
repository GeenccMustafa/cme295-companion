"""Repository-wide pytest configuration.

Puts every lecture folder on ``sys.path``, so that each lecture package
(``l01``, ``l02``, ...) can be imported by its tests, by the doctests in its
modules, and by later lectures that build on it.

Registers Hypothesis profiles: ``default`` keeps local runs fast, ``ci`` runs
many more examples. Select one with the ``HYPOTHESIS_PROFILE`` environment
variable, for example ``HYPOTHESIS_PROFILE=ci uv run pytest``.
"""

import os
import sys
from pathlib import Path

from hypothesis import settings

for lecture_dir in sorted((Path(__file__).parent / "lectures").iterdir()):
    if lecture_dir.is_dir():
        sys.path.insert(0, str(lecture_dir))

settings.register_profile("default", max_examples=100, deadline=None)
settings.register_profile("ci", max_examples=1000, deadline=None, print_blob=True)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))
