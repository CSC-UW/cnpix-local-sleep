"""Evaluation configurations: cnpix's shared ``NREM`` and ``Wake`` evaluations plus
this project's ``Match`` and ``SD``.

``Match`` scores the labeled images of the ``Early.REC.NREM.Match`` stack, like ``NREM``.
``SD`` scores the annotation of the whole sleep-deprivation stack (``NOD``): the images
with a label on the probe that the condition's hypnogram scores entirely Wake or partly
NREM. Its ``states`` make the banded detectors fit separately on the Wake and on the NREM
time of those images (:func:`cnpix_local_sleep.unit_based.banded_eval.fit_bouts`).
"""

from cnpix.evaluation.config import EVAL_CONFIGS as SHARED_EVAL_CONFIGS
from cnpix.evaluation.config import NREM_CONDITION, WAKE_CONDITION

MATCH_CONDITION = "Early.REC.NREM.Match"
SD_CONDITION = "NOD"

EVAL_CONFIGS: dict[str, dict] = {
    **SHARED_EVAL_CONFIGS,
    "Match": {
        "condition": MATCH_CONDITION,
        "chunks": "labeled",
        "filters": ("llas", "clas", "blas"),
    },
    "SD": {
        "condition": SD_CONDITION,
        "chunks": "labeled",
        "filters": ("llas", "clas"),
        "states": ("Wake", "NREM"),
    },
}

__all__ = [
    "EVAL_CONFIGS",
    "MATCH_CONDITION",
    "NREM_CONDITION",
    "SD_CONDITION",
    "WAKE_CONDITION",
]
