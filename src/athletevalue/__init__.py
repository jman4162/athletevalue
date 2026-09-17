"""athletevalue: separate what a college athlete is worth from what the market pays."""

from athletevalue.runtime import apply_thread_environment
from athletevalue.versions import MODEL_VERSION, __version__

apply_thread_environment()

__all__ = ["MODEL_VERSION", "__version__"]
