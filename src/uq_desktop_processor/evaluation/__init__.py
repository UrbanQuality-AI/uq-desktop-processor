"""
Evaluation module:   init  .
"""

from .clip_common import print_results
from .clip_prefilter import prefilter_folder

__all__ = [
    "prefilter_folder",
    "print_results",
]
