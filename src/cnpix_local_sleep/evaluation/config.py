"""Re-export of the shared evaluation configuration, which now lives in ``cnpix``.

Kept so that ``cnpix_local_sleep.evaluation.config`` keeps working for existing callers
and notebooks. New code should import from :mod:`cnpix.evaluation.config`.
"""

from cnpix.evaluation.config import (
    EVAL_CONFIGS,
    NREM_CONDITION,
    WAKE_CONDITION,
)

__all__ = [
    "EVAL_CONFIGS",
    "NREM_CONDITION",
    "WAKE_CONDITION",
]
