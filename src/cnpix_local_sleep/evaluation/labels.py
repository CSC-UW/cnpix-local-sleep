"""Re-export of the shared manual-label machinery, which now lives in ``cnpix``.

Kept so that ``cnpix_local_sleep.evaluation.labels`` keeps working for existing callers
and notebooks. New code should import from :mod:`cnpix.evaluation.labels`.
"""

from cnpix.evaluation.labels import *  # noqa: F403

# Private helpers forwarded explicitly because ``import *`` skips
# underscore-prefixed names. In-tree callers: manual_validation, head_to_head,
# full48h_eval. Only ``_build_channel_maps`` and
# ``_get_subject_probe_pairs_with_labels`` still have any; the other five are
# kept as compatibility forwards for out-of-tree code.
from cnpix.evaluation.labels import (  # noqa: F401
    _STRUCT_8CONN,
    _VERSION_RE,
    _build_channel_maps,
    _get_manual_labels_path,
    _get_manual_labels_root,
    _get_subject_probe_pairs_with_labels,
    _manual_labels_filename,
)
