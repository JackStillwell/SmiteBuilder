"""
Jack Stillwell
3 August 2020

The SmiteBuild module performs all data manipulation unique to SMITE data. This includes filtering
match data by player skill level and converting model output into readable SMITE builds.
"""

from typing import cast, Dict, List, Set, Union
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from smitebuilder.etl import RawMatchData, ItemData
from smitebuilder.smiteinfo import RankTier


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
    tokeep = [x["conquest_tier"] >= conquest_tier_cutoff for x in raw_data]
    return tokeep


def fuse_evolution_items(item_data: ItemData, itemmap: Dict[int, str]):
    """Fuses the "Evolved" and "Unevolved" versions of items into their unevolved version.
    NOTE: Works in-place.

    Args:
        item_data (ItemData): A tuple containing a (# of matches) by (# relevant items) matrix
                              containing item data from each match and a list mapping feature
                              (column) index to item id.
        itemmap (Dict[int, str]): A bidirectional map, with primary bindings from ID to
                                          Name.

    """

    evolved_ids = [k for k, v in itemmap.items() if "Evolved" in v]
    for idx, val in enumerate(item_data.feature_list):
        if val in evolved_ids:
            affected_rows = item_data.item_matrix[:, idx] == 1
            item_data.item_matrix[
                affected_rows,
                item_data.feature_list.index(itemmap.inverse[itemmap[val][8:]]),
            ] = 1
            item_data.item_matrix[affected_rows, idx] = 0
    return None


def prune_item_data(item_matrix: np.ndarray, frequency_cutoff=0.03) -> List[bool]:
    """Takes a matrix of item data and returns a boolean list indicating which items to include in
    the features.

    Args:
        item_matrix (np.ndarray): A matrix of item data, where each row is an observation and each
                                column is a feature representing a purchasable item.
        frequency_cutoff (float): A percentage decimal between 0 (0%) and 1 (100%) indicating the
                                  percentage of observations a feature must be present in.
                                  Default: 0.03 (3%)

    Returns:
        List[bool]: A list of booleans indicating which features are to be kept.
    """

    item_count = np.sum(item_matrix, axis=0)
    tokeep = [
        idx
        for idx in range(len(item_count))
        if item_count[idx] > (item_matrix.shape[0] * frequency_cutoff)
    ]
    return [True if x in tokeep else False for x in range(item_matrix.shape[1])]


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


def _convert_build_to_observation(
    build: Union[List[int], Set[int]], feature_list: List[int]
) -> np.ndarray:
    """Converts a list of SMITE ids into the corresponding observation.

    Args:
        build (Union[List[int], Set[int]]): A list or set of SMITE item ids.
        feature_list (List[int]): A list where the index of an item_id corresponds to the feature
                                  index.

    Returns:
        np.ndarray: An observation row suitable for model input.
    """

    return np.array([1 if x in build else 0 for x in feature_list]).reshape(
        (1, len(feature_list))
    )


def rate_smitebuild(
    build: SmiteBuild,
    feature_list: List[int],
    dt,
    bnb,
    dt_percentage: float,
    bnb_percentage: float,
    percentile_cutoff: int,
) -> float:
    """Takes a SMITE build and returns a confidence rating.

    Args:
        build (SmiteBuild): A list of core and optional items for a build.
        feature_list (List[int]): A list where the index of an item_id corresponds to the feature
                                  index.
        dt: A trained sklearn DecisionTreeClassifier.
        bnb: A trained sklearn BernoulliNB.
        dt_score (float): The percentage of the final confidence based on the dt_score.
        bnb_score (float): The percentage of the final confidence based on the bnb_score.
        percentile_cutoff (int): The percentile of scores to consider as the "confidence" of a
                                 model.

    Returns:
        float: A float between 0 and 1 representing the confidence the models show in the build.
    """

    builds = gen_all_builds(build)
    observations = np.vstack(
        (_convert_build_to_observation(x, feature_list) for x in builds)
    )
    dt_raw_probas = dt.predict_proba(observations)
    dt_probas = [x[1] for x in dt_raw_probas]
    bnb_raw_probas = bnb.predict_proba(observations)
    bnb_probas = [x[1] for x in bnb_raw_probas]

    dt_70 = np.percentile(dt_probas, percentile_cutoff)
    bnb_70 = np.percentile(bnb_probas, percentile_cutoff)

    returnval = (dt_70 * dt_percentage) + (bnb_70 * bnb_percentage)

    return returnval


def gen_all_builds(build: SmiteBuild) -> List[Set[int]]:
    """Given a SmiteBuild, return all possible builds containing the core and the required number
    of optional items to complete a build.

    Args:
        build (SmiteBuild): A set of core and optional items for a build.

    Returns:
        List[Set[int]]: A list of sets containing the item ids for each complete build.
    """
    num_optional = 6 - len(build.core)

    if len(build.optional) > num_optional:
        optionals = [
            cast(Set[int], set(x))
            for x in combinations(build.optional, num_optional)
        ]
    else:
        optionals = [build.optional]
    
    return [build.core | x for x in optionals]