from enum import Enum
from types import MappingProxyType
from typing import Final


class Bands(Enum):
    DELTA = (0.5, 4)
    ETA = (2, 6)
    THETA = (5, 10)
    SIGMA = (9, 16)

    @property
    def low(self) -> float:
        return self.value[0]

    @property
    def high(self) -> float:
        return self.value[1]

    def contains(self, freq: float) -> bool:
        return self.low <= freq <= self.high


EXPERIMENT: Final[str] = "novel_objects_deprivation"


CONDITIONS: Final[tuple[str, ...]] = (
    "Early.BSL.NREM",
    "Early.REC.NREM.Match",
    "Early.NOD.Wake",
    "Late.NOD.Wake",
    "Early.REC.NREM",
    "Late.REC.NREM",
    # "Early.EXT.Wake",
    # "Late.EXT.Wake",
    "Full.Conservative",
)

CORE_CONDITIONS: Final[tuple[str, ...]] = (
    "Early.BSL.NREM",
    "Early.REC.NREM.Match",
    "Early.NOD.Wake",
    "Late.NOD.Wake",
    "Early.REC.NREM",
    "Late.REC.NREM",
)

CONTRASTS: Final[MappingProxyType[str, tuple[str, str]]] = MappingProxyType(
    {
        # "EXT.Incline": ("Late.EXT", "Early.EXT"),
        "NOD.Incline": ("Late.NOD.Wake", "Early.NOD.Wake"),
        "NREM.Rebound": ("Early.REC.NREM", "Early.REC.NREM.Match"),
        "NREM.Surge": ("Early.REC.NREM", "Early.BSL.NREM"),
        "NREM.REC.Decline": ("Early.REC.NREM", "Late.REC.NREM"),
        "NREM.BSL.Decline": ("Early.BSL.NREM", "Early.REC.NREM.Match"),
    }
)
