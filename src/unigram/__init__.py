"""
Backward-compatibility shim: 'unigram' has been renamed to 'gramforge'.
This package re-exports everything from gramforge so existing code keeps working.
"""
import warnings
warnings.warn(
    "The 'unigram' package has been renamed to 'gramforge'. "
    "Please update your imports: pip install gramforge",
    DeprecationWarning, stacklevel=2
)

from gramforge import *  # noqa: F401,F403
from gramforge import unigram_to_nltk, gramforge_to_nltk  # noqa: F401