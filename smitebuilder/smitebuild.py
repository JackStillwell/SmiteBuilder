"""
Jack Stillwell
3 August 2020

The SmiteBuild module performs all data manipulation unique to SMITE data. This includes filtering
match data by player skill level and converting model output into readable SMITE builds.
"""

from typing import Callable, cast, Dict, List, Optional, Set, FrozenSet, Tuple, Union
from copy import deepcopy
from itertools import combinations

import numpy as np

from smitebuilder.etl import RawMatchData, ItemData
from smitebuilder.smiteinfo import RankTier, SmiteBuild, SmiteBuildPath


NUM_ITEMS_IN_BUILD = 6
NUM_ITEMS_IN_CORE = 4
NUM_OPTIONAL_ITEMS = NUM_ITEMS_IN_BUILD - NUM_ITEMS_IN_CORE
BUILD_SIMILARITY_CUTOFF = 0.25


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


def fuse_evolution_items(
    item_data: ItemData, itemmap: Dict[int, str], evolved_item_map: Dict[int, int]
):
    """Fuses the "Evolved" and "Unevolved" versions of items into their unevolved version.
    NOTE: Works in-place.

    Args:
        item_data (ItemData): A tuple containing a (# of matches) by (# relevant items) matrix
                              containing item data from each match and a list mapping feature
                              (column) index to item id.
        itemmap (Dict[int, str]): A bidirectional map, with primary bindings from ID to
                                  Name.
        evolved_item_map (Dict[int, int]): A map with keys from all t4 item ids to their t3 items.
    """

    for idx, val in enumerate(item_data.feature_list):
        if val in evolved_item_map.keys():
            affected_rows = item_data.item_matrix[:, idx] == 1
            item_data.item_matrix[
                affected_rows,
                item_data.feature_list.index(evolved_item_map[val]),
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


def feature_to_item(
    feature_ids: List[List[int]], feature_map: List[int]
) -> List[List[int]]:
    """Given a list of feature ids and a feature_to_item map, returns a list of item ids given a
    list of feature ids.

    Args:
        feature_ids (List[List[int]]): Lists of feature_ids
        feature_map (List[int]): A map linking feature_id (as idx) to item ids

    Returns:
        List[List[int]]: Lists of item ids
    """
    return [[feature_map[x] for x in ids] for ids in feature_ids]


def _convert_build_to_observation(
    build: Union[List[int], FrozenSet[int]], feature_list: List[int]
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


def rate_builds(
    builds: List[FrozenSet[int]],
    feature_list: List[int],
    dt,
    bnb,
    dt_percentage: float,
    bnb_percentage: float,
) -> List[float]:
    """NEEDS DOCSTRING"""
    if not builds:
        return []

    observations = np.vstack(
        [_convert_build_to_observation(x, feature_list) for x in builds]
    )
    dt_raw_probas = dt.predict_proba(observations)
    dt_probas = [x[1] for x in dt_raw_probas]
    bnb_raw_probas = bnb.predict_proba(observations)
    bnb_probas = [x[1] for x in bnb_raw_probas]

    return [
        (x * dt_percentage) + (y * bnb_percentage)
        for x, y in zip(dt_probas, bnb_probas)
    ]


def rate_smitebuildpath(
    build: SmiteBuildPath,
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
        dt_percentage (float): The percentage of the final confidence based on the dt_score.
        bnb_percentage (float): The percentage of the final confidence based on the bnb_score.
        percentile_cutoff (int): The percentile of scores to consider as the "confidence" of a
                                 model.

    Returns:
        float: A float between 0 and 1 representing the confidence the models show in the build.
    """

    builds = list(gen_all_builds(build))
    ratings = rate_builds(builds, feature_list, dt, bnb, dt_percentage, bnb_percentage)

    if not ratings:
        return 0.0

    return np.percentile(ratings, percentile_cutoff)


def gen_all_builds(build: SmiteBuildPath) -> Set[FrozenSet[int]]:
    """Given a SmiteBuild, return all possible builds containing the core and the required number
    of optional items to complete a build.

    Args:
        build (SmiteBuild): A set of core and optional items for a build.

    Returns:
        List[Set[int]]: A list of sets containing the item ids for each complete build.
    """
    optionals: Set[FrozenSet[int]] = set()

    for options in build.optionals:
        if len(options) > NUM_OPTIONAL_ITEMS:
            option_combos: Set[FrozenSet[int]] = {
                frozenset(x) for x in combinations(options, NUM_OPTIONAL_ITEMS)
            }
            optionals |= option_combos
        else:
            optionals.add(options)

    return {frozenset(build.core | x) for x in optionals}


def select_builds(
    builds: List[SmiteBuildPath], num_select: int, build_similarity_cutoff: float
) -> List[SmiteBuildPath]:
    """Given a list of builds and number of builds to select, return a list of distinct
    (a build similarity lower than "build_similarity_cutoff") SmiteBuildPaths. It goes through
    the "builds" and returns once the "num_select" is reached, so any builds after that will
    not be considered.

    Args:
        builds (List[SmiteBuildPath]): [description]
        num_select (int): [description]

    Returns:
        List[SmiteBuildPath]: [description]
    """

    ret_list = []
    i = 0
    while len(ret_list) < num_select and i < len(builds):
        curr_build = builds[i]
        add = True
        for build in ret_list:
            if build_similarity(build, curr_build) > build_similarity_cutoff:
                add = False

        if add:
            ret_list.append(curr_build)

        i += 1

    return ret_list


def build_similarity(build1: SmiteBuildPath, build2: SmiteBuildPath) -> float:
    """Given two SmiteBuildPath objects, return the percentage of possible builds which are
    identical between the two.

    Args:
        build1 (SmiteBuildPath): [description]
        build2 (SmiteBuildPath): [description]

    Returns:
        float: [description]
    """

    all_ones = set(gen_all_builds(build1))
    all_twos = set(gen_all_builds(build2))

    all_possible = all_ones | all_twos
    all_similar = all_ones & all_twos

    if len(all_possible) == 0:
        return 0
    else:
        similarity = len(all_similar) / len(all_possible)
        return similarity


def find_common_cores(
    traces: List[List[int]], core_length: int, num_cores: Optional[int]
) -> Set[FrozenSet[int]]:
    """Detects and returns up to "num cores" most frequently occurring cores in "traces".

    Args:
        builds (List[List[int]]): [description]
        core_length (int): [description]
        num_cores (int): [description]

    Returns:
        List[Set[int]]: [description]
    """

    all_cores: Set[FrozenSet[int]] = {
        frozenset(y) for x in traces for y in combinations(x, core_length)
    }

    if num_cores is None:
        return all_cores

    core_count = [
        (core, sum(1 if core <= set(x) else 0 for x in traces)) for core in all_cores
    ]

    core_count.sort(key=lambda x: x[1], reverse=True)

    return {x[0] for x in core_count[:num_cores]}


def get_options(traces: List[List[int]], core: FrozenSet[int]) -> Set[FrozenSet[int]]:
    """Given a list of items and a core, determines and returns a set of "optional" item sets.

    Args:
        builds (List[List[int]]): A list of list of items.
        core (FrozenSet[int]): A set representing the "core" found in those items.

    Returns:
        Set[FrozenSet[int]]: A set of frozensets of "optional" items found with the core.
    """
    return {frozenset(set(x) - core) for x in traces if core <= set(x)}


def prune_options(
    core: FrozenSet[int],
    optionals: Set[FrozenSet[int]],
    rank_builds: Callable[[List[FrozenSet[int]]], List[float]],
    rank_percentile_cutoff: int,
) -> Set[FrozenSet[int]]:
    """Remove any options which lower the score of the build below the percentile score of
    "rank_percentile_cutoff".

    Args:
        core (FrozenSet[int]): The core associated with the optional items.
        optionals (Set[FrozenSet[int]]): A set of possible optional items used to fill out the build.

    Returns:
        Set[FrozenSet[int]]: A set of possible optional items which score in the
                             "rank_percentile_cutoff" percentile of the optionals.
    """

    # ensure every item is a single option
    all_items = {y for x in optionals for y in x}

    # rank all combinations, and remove any that lower the rank of the core
    all_options: List[FrozenSet[int]] = [
        frozenset(x) for x in combinations(all_items, NUM_OPTIONAL_ITEMS)
    ]

    if not all_options:
        return set()

    all_builds: List[FrozenSet[int]] = [
        frozenset(core | options) for options in all_options
    ]
    build_ranks = rank_builds(all_builds)

    # rank percentile calculation
    rank_cutoff = np.percentile(build_ranks, rank_percentile_cutoff)

    pruned_options = {
        option for option, rank in zip(all_options, build_ranks) if rank >= rank_cutoff
    }

    return pruned_options


def consolidate_options(options: Set[FrozenSet[int]]) -> Set[FrozenSet[int]]:
    """Given a set of options, attempts to combine them so that the largest possible option sets
    are created.

    Args:
        options (Set[FrozenSet[int]]): Lists of items to fill out the core builds.

    Returns:
        Set[FrozenSet[int]]: A consolidated list of lists of items to fill out the core builds.
    """

    pruned_items = {y for x in options for y in x}

    # look through the combinations, and combine any which exist where all items are
    #    combined. [[x1, x2], [x2, x3], [x1, x3]] -> [[x1, x2, x3]]
    for i in reversed(range(NUM_OPTIONAL_ITEMS, len(pruned_items) + 1)):
        possible_consolidations: Set[FrozenSet[int]] = {
            frozenset(x) for x in combinations(pruned_items, i)
        }

        for consolidation in possible_consolidations:
            required_subcombos = {
                frozenset(x) for x in combinations(consolidation, NUM_OPTIONAL_ITEMS)
            }
            if required_subcombos <= options:
                options.add(consolidation)

    # remove any subsets
    deduped_options = deepcopy(options)
    for x in options:
        for y in options:
            if x < y:
                deduped_options.discard(x)

    return deduped_options
