import json
import sys
import os

from argparse import ArgumentParser, Namespace
from typing import List, Set, Tuple, Optional
from copy import deepcopy
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB


def parse_args(args: List[str]) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--datapath", "-d", required=True,
    )
    parser.add_argument(
        "--queue", "-q", required=True, choices=["conquest", "joust", "duel"]
    )
    parser.add_argument("--god", "-g", required=True)
    parser.add_argument("--conquest_tier", "-ct", default=15)
    parser.add_argument("--probability_score_limit", "-psl", default=0.5)
    parser.add_argument("--probability_score_cutoff", "-psc", default=0.7)

    return parser.parse_known_args(args)[0]


@dataclass
class SmiteBuild:
    core: Set[int]
    optional: Set[int]


ReadableSmiteBuild = namedtuple("ReadableSmiteBuild", ["core", "optional"])
MainReturn = namedtuple("MainReturn", ["build", "dt_rank", "bnb_rank"])


def main(
    path_to_data: str,
    queue: str,
    target_god: str,
    conquest_tier_cutoff: int,
    probability_score_limit: float,
    probability_score_cutoff: float,
) -> Optional[List[MainReturn]]:
    # NOTE assumes laid out as in SmiteData repo
    with open(os.path.join(path_to_data, "gods.json"), "r") as infile:
        gods = json.loads(infile.readline())
        godname_to_id = {x["Name"]: x["id"] for x in gods}

    with open(os.path.join(path_to_data, "items.json"), "r") as infile:
        items = json.loads(infile.readline())
        id_to_itemname = {x["ItemId"]: x["DeviceName"] for x in items}
        item_ids = [x["ItemId"] for x in items]

    queue_path = queue + "_match_data"

    with open(
        os.path.join(
            path_to_data, queue_path, str(godname_to_id[target_god]) + ".json",
        ),
        "r",
    ) as infile:
        god_data = json.loads(infile.readline())

    # filter god data by conquest rank plat and above
    filtered_god_data = [
        x for x in god_data if x["conquest_tier"] > conquest_tier_cutoff
    ]
    while len(filtered_god_data) < 500:
        conquest_tier_cutoff -= 1
        filtered_god_data = [
            x for x in god_data if x["conquest_tier"] > conquest_tier_cutoff
        ]

        if conquest_tier_cutoff < 0:
            break

    god_data = filtered_god_data

    print(
        len(god_data),
        "matches found with a conquest_tier cutoff of",
        conquest_tier_cutoff,
    )

    npdata = np.array(
        [
            np.divide(
                np.array(
                    [
                        x["assists"],
                        x["damage_mitigated"],
                        x["damage_player"],
                        x["damage_taken"],
                        x["deaths"],
                        x["healing"],
                        x["healing_player_self"],
                        x["kills_player"],
                        x["structure_damage"],
                    ]
                ),
                x["match_time_minutes"],
            )
            for x in god_data
        ]
    )
    npdata_normalized = npdata / np.array(
        [x if x != 0 else 1 for x in npdata.max(axis=0)]
    )
    npdata_winlabel = np.array(
        [1 if x["win_status"] == "Winner" else 0 for x in god_data]
    )

    npdn_shape: Tuple[int, int] = npdata_normalized.shape
    print("pruning", npdn_shape[0], "rows for non-nan validity")

    valid_rows = [
        (x, y)
        for x, y in zip(npdata_normalized, npdata_winlabel)
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    ]

    print(len(valid_rows), "rows remain")

    if not valid_rows:
        print("No valid rows remain. Exiting...")
        return None

    temp = list(zip(*valid_rows))
    npdata_normalized = np.array(list(temp[0]))
    npdata_winlabel = np.array(list(temp[1]))

    sgd_classifier = SGDClassifier(max_iter=1000, random_state=0)
    sgd_classifier.fit(npdata_normalized, npdata_winlabel)

    print("sgd_score:", sgd_classifier.score(npdata_normalized, npdata_winlabel))

    # NOTE original design included a clustering step but after research
    #        I believe simply predicting based on SGD is more than sufficient

    new_winlabel = sgd_classifier.predict(npdata_normalized)

    item_data = np.array(
        [
            [1 if item_id in x["item_ids"] else 0 for item_id in item_ids]
            for x in god_data
        ]
    )

    item_max = np.max(item_data, axis=0)
    todelete = [idx for idx in range(len(item_max)) if item_max[idx] == 0]
    item_data = np.delete(item_data, todelete, axis=1)

    item_data_ids = [x for idx, x in enumerate(item_ids) if idx not in todelete]

    dt_classifier = DecisionTreeClassifier(
        criterion="entropy", max_features=1, random_state=0,
    )
    dt_classifier.fit(item_data, new_winlabel)

    print("dt_score:", dt_classifier.score(item_data, new_winlabel))

    bnb_classifier = BernoulliNB()
    bnb_classifier.fit(item_data, new_winlabel)

    print("bnb_score:", bnb_classifier.score(item_data, new_winlabel))

    print("Constructing builds from tree...")
    builds = []
    create_builds(dt_classifier.tree_, 0, [], builds)

    possible_builds = np.array(
        [[1 if idx in x else 0 for idx in range(len(item_data_ids))] for x in builds]
    )

    if not possible_builds.size > 0:
        print("No possible builds found. Exiting...")
        return None

    dt_predictions = dt_classifier.predict_proba(possible_builds)

    possible_builds = [
        x
        for idx, x in enumerate(possible_builds)
        if dt_predictions[idx][1] > probability_score_cutoff
    ]
    while not possible_builds:
        probability_score_cutoff -= 0.05
        possible_builds = [
            x
            for idx, x in enumerate(possible_builds)
            if dt_predictions[idx][1] > probability_score_cutoff
        ]

        if probability_score_cutoff < probability_score_limit:
            print(
                "Probability cutoff reached",
                probability_score_cutoff,
                ". No viable builds found. Exiting...",
            )
            return None

    print(
        len(possible_builds),
        "possible build paths found with a success probability cutoff of",
        probability_score_cutoff,
    )

    feature_builds = [
        [idx for idx, x in enumerate(build) if x == 1] for build in possible_builds
    ]

    core_size = 4
    smitebuilds = make_smitebuilds(feature_builds, core_size)
    while not smitebuilds:
        core_size -= 1
        smitebuilds = make_smitebuilds(feature_builds, core_size)

        if core_size < 1:
            print("No core found. Exiting...")
            return None

    print(len(smitebuilds), "possible builds found with a core_size of", core_size)

    possible_smitebuilds = np.array(
        [
            [1 if idx in x.core else 0 for idx in range(len(item_data_ids))]
            for x in smitebuilds
        ]
    )

    if not possible_smitebuilds.size > 0:
        print("Exiting...")
        return None

    bnb_ranking = bnb_classifier.predict_proba(possible_smitebuilds)
    dt_ranking = dt_classifier.predict_proba(possible_smitebuilds)

    smitebuild_ranks = [
        (x, bnb_ranking[idx][1], dt_ranking[idx][1])
        for idx, x in enumerate(smitebuilds)
    ]
    smitebuild_ranks.sort(key=lambda x: x[1], reverse=True)

    smitebuild_ranks_pruned = [
        x for x in smitebuild_ranks if x[2] > probability_score_cutoff
    ][:3]
    while len(smitebuild_ranks_pruned) < 3:
        probability_score_cutoff -= 0.05
        smitebuild_ranks_pruned = [
            x for x in smitebuild_ranks if x[2] > probability_score_cutoff
        ][:3]

        if probability_score_cutoff < 0:
            break

    returnval = []
    for smitebuild in smitebuild_ranks_pruned:
        elem = MainReturn(
            build=ReadableSmiteBuild(
                core=[id_to_itemname[item_data_ids[x]] for x in smitebuild[0].core],
                optional=[
                    id_to_itemname[item_data_ids[x]] for x in smitebuild[0].optional
                ],
            ),
            dt_rank=smitebuild[2],
            bnb_rank=smitebuild[1],
        )
        returnval.append(elem)
        print("core:", [id_to_itemname[item_data_ids[x]] for x in smitebuild[0].core])
        print(
            "optional:",
            [id_to_itemname[item_data_ids[x]] for x in smitebuild[0].optional],
        )
        print("dt_rank:", smitebuild[2])
        print("bnb_rank:", smitebuild[1])

    return returnval


def create_builds(tree, node: int, local_build: List[int], builds: List[List[int]]):
    # this stops if we hit a leaf node
    if tree.children_left[node] == tree.children_right[node]:
        return None

    # this stops if we flesh out a build
    if len(local_build) > 4:
        # print('appending build', local_build)
        builds.append(local_build)
        return None

    create_builds(tree, tree.children_left[node], deepcopy(local_build), builds)

    local_build += [tree.feature[node]]
    create_builds(tree, tree.children_right[node], deepcopy(local_build), builds)


def make_smitebuilds(builds: List[List[int]], num_core: int) -> List[SmiteBuild]:
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


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(
        args.datapath,
        args.queue,
        args.god,
        int(args.conquest_tier),
        float(args.probability_score_limit),
        float(args.probability_score_cutoff),
    )
