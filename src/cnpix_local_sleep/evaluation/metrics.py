"""Re-export of the shared metric kernels, which now live in ``cnpix``.

Kept so that ``cnpix_local_sleep.evaluation.metrics`` keeps working for existing callers
and notebooks. New code should import from :mod:`cnpix.evaluation.metrics`.
"""

from cnpix.evaluation.metrics import *  # noqa: F403
