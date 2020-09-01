"""
Jack Stillwell
3 August 2020

The SmiteBuild module performs all data manipulation unique to SMITE data. This includes filtering
match data by player skill level and converting model output into readable SMITE builds.
"""

from typing import Callable, cast, Dict, List, Optional, Set, FrozenSet, Tuple, Union
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from smitebuilder.etl import RawMatchData, ItemData
from smitebuilder.smiteinfo import RankTier, SmiteBuild


NUM_ITEMS_IN_BUILD = 6
NUM_ITEMS_IN_CORE = 4
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


def feature_to_item(
    feature_ids: List[List[int]], feature_map: List[int]
) -> List[List[int]]:
    """Needs Docstring
    """
    return [[feature_map[x] for x in ids] for ids in feature_ids]


def make_smitebuilds(raw_builds: List[List[int]], num_core: int,) -> List[SmiteBuild]:
    """Transform decision tree traces into SMITE item builds.

    Args:
        raw_builds (List[List[int]]): A list of item_ids from the traces of the decision tree.
        num_core (int): The minimum number of items similar between multiple builds to be
                        considered a "core".

    Returns:
        List[SmiteBuild]: A list of builds with "core" and "optional" items.
    """

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


def rate_builds(
    builds: List[FrozenSet[int]],
    feature_list: List[int],
    dt,
    bnb,
    dt_percentage: float,
    bnb_percentage: float,
) -> List[float]:
    """NEEDS DOCSTRING
    """
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
        dt_percentage (float): The percentage of the final confidence based on the dt_score.
        bnb_percentage (float): The percentage of the final confidence based on the bnb_score.
        percentile_cutoff (int): The percentile of scores to consider as the "confidence" of a
                                 model.

    Returns:
        float: A float between 0 and 1 representing the confidence the models show in the build.
    """

    builds = gen_all_builds(build)
    ratings = rate_builds(builds, feature_list, dt, bnb, dt_percentage, bnb_percentage)

    return np.percentile(ratings, percentile_cutoff)


def gen_all_builds(build: SmiteBuild) -> List[FrozenSet[int]]:
    """Given a SmiteBuild, return all possible builds containing the core and the required number
    of optional items to complete a build.

    Args:
        build (SmiteBuild): A set of core and optional items for a build.

    Returns:
        List[Set[int]]: A list of sets containing the item ids for each complete build.
    """
    num_optional = max(0, NUM_ITEMS_IN_BUILD - len(build.core))

    if len(build.optional) > num_optional:
        optionals = [
            cast(Set[int], set(x)) for x in combinations(build.optional, num_optional)
        ]
    else:
        optionals = [build.optional]

    return [frozenset(build.core | x) for x in optionals]


def consolidate(builds: Tuple[SmiteBuild, SmiteBuild]) -> Optional[SmiteBuild]:
    """NEEDS DOCSTRING
    """
    all_builds = [list(x.core) + list(x.optional) for x in builds]
    new_builds = make_smitebuilds(all_builds, NUM_ITEMS_IN_CORE)

    if len(new_builds) > 1 or not new_builds:
        return None

    return new_builds[0]


def consolidate_builds(builds: List[SmiteBuild]):
    """NEEDS DOCSTRING NOTE: Works in-place.
    """
    possible_consolidations = list(combinations(builds, 2))

    while possible_consolidations:
        c = possible_consolidations.pop()
        consolidation = consolidate(c)

        if consolidation:
            builds.remove(c[0])
            builds.remove(c[1])
            builds.append(consolidation)
            possible_consolidations = list(combinations(builds, 2))


def prune_and_split_build(
    build: SmiteBuild,
    rate_builds: Callable[[List[FrozenSet[int]]], List[float]],
    rating_cutoff: float,
) -> List[SmiteBuild]:
    """NEEDS DOCSTRING
    """

    # First, make all possible builds
    all_builds = gen_all_builds(build)

    # then rate all the builds
    build_ratings = rate_builds(all_builds)

    pruned_builds = [
        list(x) for x, y in zip(all_builds, build_ratings) if y > rating_cutoff
    ]

    smitebuilds = make_smitebuilds(pruned_builds, 4)

    return smitebuilds


def select_builds(builds: List[SmiteBuild], num_select: int) -> List[SmiteBuild]:
    """Given a list of builds and number of builds to select, return a list of distinct
    (containing no more than 4 similar items) SmiteBuilds.

    Args:
        builds (List[SmiteBuild]): [description]
        num_select (int): [description]

    Returns:
        List[SmiteBuild]: [description]
    """

    ret_list = []
    i = 0
    while len(ret_list) < num_select and i < len(builds):
        curr_build = builds[i]
        add = True
        for build in ret_list:
            if build_similarity(build, curr_build) > BUILD_SIMILARITY_CUTOFF:
                add = False

        if add:
            ret_list.append(curr_build)

        i += 1

    return ret_list


def build_similarity(build1: SmiteBuild, build2: SmiteBuild) -> float:
    """Given two SmiteBuild objects, return the percentage of possible builds which are identical
    between the two.

    Args:
        build1 (SmiteBuild): [description]
        build2 (SmiteBuild): [description]

    Returns:
        float: [description]
    """

    all_ones = set(gen_all_builds(build1))
    all_twos = set(gen_all_builds(build2))

    all_possible = all_ones | all_twos
    all_similar = all_ones & all_twos

    similarity = len(all_similar) / len(all_possible)

    return similarity


def find_common_cores(
    traces: List[List[int]], core_length: int, num_cores: int
) -> List[FrozenSet[int]]:
    """ Detects and returns up to "num cores" most frequently occurring cores in "traces".

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

    core_count = [
        (core, sum(1 if core <= set(x) else 0 for x in traces)) for core in all_cores
    ]

    core_count.sort(key=lambda x: x[1], reverse=True)

    return [x[0] for x in core_count[:num_cores]]


def get_options(traces: List[List[int]], core: FrozenSet[int]) -> Set[FrozenSet[int]]:
    """[summary]

    Args:
        builds (List[List[int]]): [description]
        core (FrozenSet[int]): [description]

    Returns:
        List[int]: [description]
    """
    return {frozenset(set(x) - core) for x in traces if core <= set(x)}


def prune_options(
    core: FrozenSet[int],
    optionals: Set[FrozenSet[int]],
    rank_builds: Callable[[List[FrozenSet[int]]], List[float]],
) -> Set[FrozenSet[int]]:
    """Remove any options which lower the score of the build.

    Args:
        core (FrozenSet[int]): [description]
        optionals (Set[FrozenSet[int]]): [description]

    Returns:
        Set[FrozenSet[int]]: [description]
    """

    # first, remove any single-item options which lower the chance of success
    single_option_list = [x for x in optionals if len(x) == 1]
    single_build_list = [core]
    single_build_list += [option | core for option in single_option_list]
    single_ranks = rank_builds(single_build_list)

    removed_singles = {
        option
        for option, rank in zip(single_option_list, single_ranks)
        if rank <= single_ranks[0]
    }

    multi_option_set = {x for x in optionals if len(x) > 1}
    removed_singles |= multi_option_set

    # next, look to combine optionals
    for optional in optionals:
        pass

    return set()


def consolidate_options(options: Set[FrozenSet[int]]) -> Set[FrozenSet[int]]:
    """[summary]

    Args:
        options (List[FrozenSet[int]]): [description]
    """

    # Get a list of every item which appears with each item
    all_items = {y for x in options for y in x}
    item_dicts = {x: {z for y in options for z in y - {x} if x in y} for x in all_items}
    item_groups = set()
    for key, val in item_dicts.items():
        new_set = set()
        new_set.add(key)
        new_set |= val
        new_frozen = frozenset(new_set)
        item_groups.add(new_frozen)

    # Goal: Include all items in the smallest amount of groupings
    # Go group by group, and include the largest group which contains the item
    pruned_groups = set()
    included_items = set()
    for item in all_items:
        if item in included_items:
            continue

        groups = list({x for x in item_groups if item in x})
        groups.sort(key=lambda x: len(x), reverse=True)

        included_items |= groups[0]
        pruned_groups.add(groups[0])

    return pruned_groups
