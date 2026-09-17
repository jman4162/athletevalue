"""Version identifiers stamped on every estimate.

``MODEL_VERSION`` changes whenever a modelling choice changes an output, even if
the package version does not. Cached fits are keyed on it, so bumping it
invalidates stale results instead of silently reusing them.
"""

from __future__ import annotations

__version__ = "0.3.1"

MODEL_VERSION = "mbb-v0.3.1"
