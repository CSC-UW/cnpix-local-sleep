# Only import lightweight configuration modules at package level.
# Heavy modules (trace_io, off_tables, channel_anatomy, unit_based, hyp,
# plots) must be imported directly to avoid slow import times:
#   from cnpix_local_sleep import trace_io
# Structure configuration lives in cnpix_local_sleep.sps_conf.
from . import const, files
from .const import (
    CONDITIONS,
    CONTRASTS,
    CORE_CONDITIONS,
    EXPERIMENT,
    Bands,
)

__all__ = [
    "const",
    "files",
    "CONDITIONS",
    "CONTRASTS",
    "CORE_CONDITIONS",
    "EXPERIMENT",
    "Bands",
]