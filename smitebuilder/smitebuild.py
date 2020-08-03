"""
Jack Stillwell
3 August 2020

The SmiteBuild module performs all data manipulation unique to SMITE data. This includes filtering
match data by player skill level and converting model output into readable SMITE builds.
"""

from smitebuilder.smiteinfo import RankTier
from typing import List, Set
from dataclasses import dataclass

import numpy as np

from smitebuilder.etl import RawMatchData


@dataclass
class SmiteBuild:
    core: Set[int]
    optional: Set[int]


def filter_data_by_player_skill(
    raw_data: List[RawMatchData], conquest_tier_cutoff: RankTier
) -> List[bool]:
    """Takes a list of raw match data and returns a boolean list indicating which observations
    to include in the models.

    Args:
        raw_data (List[RawMatchData]): A list of dictionares containing the information required to
                                       run SmiteBuilder.
        conquest_tier_cutoff (RankTier): The minimum conquest rank tier required for a player to be
                                         included in the model training data.

    Returns:
        List[bool]: A list of booleans indicating which observations are to be kept.
    """

    # TODO improve this filtering
    return [x["conquest_tier"] >= conquest_tier_cutoff for x in raw_data]


def fuse_evolved_items(raw_data: List[RawMatchData]):
    """Combines 

    Args:
        raw_data (List[RawMatchData]): [description]
    """
    return None


def prune_item_data(item_data: np.ndarray) -> List[bool]:
    """Takes a matrix of item data and returns a boolean list indicating which items to include in
    the features.

    Args:
        item_data (np.ndarray): A matrix of item data, where each row is an observation and each
                                column is a feature representing a purchasable item.

    Returns:
        List[bool]: A list of booleans indicating which features are to be kept.
    """
    return []


def make_smitebuilds(builds: List[List[int]], num_core: int) -> List[SmiteBuild]:
    """
    NEEDS DOCSTRING
    """

    smitebuilds = []

    for i, build_i in enumerate(builds):
        for j, build_j in enumerate(builds):
            if i == j:
                continue

            potential_core = set(build_i) & set(build_j)
            if len(potential_core) >= num_core:
                optional = set(build_i) ^ set(build_j)
                try:
                    idx = [x.core for x in smitebuilds].index(potential_core)
                    smitebuilds[idx].optional |= optional
                except ValueError:
                    smitebuilds.append(SmiteBuild(potential_core, optional))

    return smitebuilds
