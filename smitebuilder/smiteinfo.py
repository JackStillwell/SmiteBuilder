"""
Jack Stillwell
3 August 2020

The SmiteInfo module contains the information necessary for converting between readable language
and the data returned by the SMITE API and understood by the models.
"""

from enum import IntEnum
from typing import Dict, List, NamedTuple, Set

from dataclasses import dataclass


@dataclass
class SmiteBuild:
    core: Set[int]
    optional: Set[int]


class ReadableSmiteBuild(NamedTuple):
    core: List[str]
    optional: List[str]

    @staticmethod
    def from_SmiteBuild(build: SmiteBuild, item_map: Dict[int, str]):
        return ReadableSmiteBuild(
            core=[item_map[x] for x in build.core],
            optional=[item_map[x] for x in build.optional],
        )


class MainReturn(NamedTuple):
    build: ReadableSmiteBuild
    confidence: float


class RankTier(IntEnum):
    BRONZE5 = 0
    BRONZE4 = 1
    BRONZE3 = 2
    BRONZE2 = 3
    BRONZE1 = 4
    SILVER5 = 5
    SILVER4 = 6
    SILVER3 = 7
    SILVER2 = 8
    SILVER1 = 9
    GOLD5 = 10
    GOLD4 = 11
    GOLD3 = 12
    GOLD2 = 13
    GOLD1 = 14
    PLATINUM5 = 15
    PLATINUM4 = 16
    PLATINUM3 = 17
    PLATINUM2 = 18
    PLATINUM1 = 19
    DIAMOND5 = 20
    DIAMOND4 = 21
    DIAMOND3 = 22
    DIAMOND2 = 23
    DIAMOND1 = 24
    MASTER5 = 25
    MASTER4 = 26
    MASTER3 = 27
    MASTER2 = 28
    MASTER1 = 29
    GRANDMASTER = 30


class MatchmakingRank(IntEnum):
    BRONZE = 0
    SILVER = 1
    GOLD = 2
    PLATINUM = 3
    DIAMOND = 4
    MASTER = 5


def stat_to_rank(stat: float) -> MatchmakingRank:
    """Takes a "rank_stat" and returns the corresponding rank tier.

    Args:
        stat (float): The float "rank_stat" value in the raw match data.

    Returns:
        MatchmakingRank: An IntEnum correlating to the rank tier.
    """

    if stat < 640:
        return MatchmakingRank.BRONZE
    elif stat < 1205:
        return MatchmakingRank.SILVER
    elif stat < 1525:
        return MatchmakingRank.GOLD
    elif stat < 1840:
        return MatchmakingRank.PLATINUM
    elif stat < 2325:
        return MatchmakingRank.DIAMOND
    else:
        return MatchmakingRank.MASTER
