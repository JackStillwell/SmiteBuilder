"""
Jack Stillwell
3 August 2020

The SmiteBuild module performs all data manipulation unique to SMITE data. This includes filtering
match data by player skill level and converting model output into readable SMITE builds.
"""

from typing import Dict, NamedTuple, List, Set
from bidict import bidict

import numpy as np

from smitebuilder.etl import RawMatchData, ItemData
from smitebuilder.smiteinfo import RankTier


class SmiteBuild(NamedTuple):
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


def fuse_evolution_items(item_data: ItemData, itemmap: bidict(Dict[int, str])):
    """Fuses the "Evolved" and "Unevolved" versions of items into their unevolved version.
    NOTE: Works in-place.

    Args:
        item_data (ItemData): A tuple containing a (# of matches) by (# relevant items) matrix
                              containing item data from each match and a list mapping feature
                              (column) index to item id.
        itemmap (bidict(Dict[int, str])): A bidirectional map, with primary bindings from ID to
                                          Name.

    """

    evolved_ids = [k for k, v in itemmap.items() if "Evolved" in v]
    for idx, val in enumerate(item_data.feature_list):
        if val in evolved_ids:
            affected_rows = item_data.item_matrix[:, idx] == 1
            item_data.item_matrix[affected_rows, itemmap.inverse[itemmap[val][8:]]] = 1
            item_data.item_matrix[affected_rows, idx] = 0
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

    item_count = np.sum(item_data.item_matrix, axis=0)
    todelete = [
        idx
        for idx in range(len(item_count))
        if item_count[idx] < (item_data.item_matrix.shape[0] * 0.03)
    ]
    return [
        True if x in todelete else False for x in range(item_data.item_matrix.shape[1])
    ]


def make_smitebuilds(
    traces: List[List[int]], num_core: int, feature_list: List[int]
) -> List[SmiteBuild]:
    """Transform decision tree traces into SMITE item builds.

    Args:
        traces (List[List[int]]): A list of all the traces of the decision tree.
        num_core (int): The minimum number of items similar between multiple builds to be
                        considered a "core".
        feature_list (List[int]): A list mapping feature id (col idx) to item id.

    Returns:
        List[SmiteBuild]: A list of builds with "core" and "optional" items.
    """

    # first, convert all feature ids to item ids
    raw_builds = [[feature_list[x] for x in trace] for trace in traces]

    smitebuilds = []

    for i, build_i in enumerate(raw_builds):
        for j, build_j in enumerate(raw_builds):
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


def convert_build_to_observation(
    build: List[int], feature_list: List[int]
) -> np.ndarray:
    """Converts a list of SMITE ids into the corresponding observation.

    Args:
        build (List[int]): A list of SMITE item ids.
        feature_list (List[int]): A list where the index of an item_id corresponds to the feature
                                  index.

    Returns:
        np.ndarray: An observation row suitable for model input.
    """

    return np.array([1 if x in build else 0 for x in feature_list])


def rate_smitebuild(build: SmiteBuild, feature_list: List[int], dt, bnb) -> float:
    """Takes a SMITE build and returns a confidence rating.

    Args:
        build (SmiteBuild): A list of core and optional items for a build.
        feature_list (List[int]): A list where the index of an item_id corresponds to the feature
                                  index.
        dt: A trained sklearn DecisionTreeClassifier.
        bnb: sklearn BernoulliNB.

    Returns:
        float: A float between 0 and 1 representing the confidence the models show in the build.
    """

    # TODO update this to better correspond to overall confidence rather than just core build
    dt_proba = 0.0
    bnb_proba = 0.0

    observation = convert_build_to_observation(list(build.core), feature_list)

    dt_proba += dt.predict_proba(observation)
    bnb_proba += bnb.predict_proba(observation)

    return (dt_proba * 0.66) + (bnb_proba + 0.34)
