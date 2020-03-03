import json
import sys
import os

from argparse import ArgumentParser, Namespace
from typing import List, Set
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB


def parse_args(args: List[str]) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        '--datapath', '-d',
        required=True,
    )

    parser.add_argument(
        '--god', '-g',
        required=True
    )

    return parser.parse_known_args(args)[0]


@dataclass
class SmiteBuild():
    core: Set[int]
    optional: Set[int]


def main(path_to_data: str, target_god: str):
    # NOTE assumes laid out as in SmiteData repo
    with open(os.path.join(path_to_data, 'gods.json'), 'r') as infile:
        gods = json.loads(infile.readline())
        godname_to_id = {x['Name']: x['id'] for x in gods}

    with open(os.path.join(path_to_data, 'items.json'), 'r') as infile:
        items = json.loads(infile.readline())
        id_to_itemname = {x['ItemId']: x['DeviceName'] for x in items}
        item_ids = [x['ItemId'] for x in items]

    with open( os.path.join(
            path_to_data,
            'conquest_match_data',
            str(godname_to_id[target_god]) + '.json'
        ),
        'r') as infile:
        god_data = json.loads(infile.readline())

    npdata = np.array([
        np.divide(
            np.array([
                x['assists'],
                x['damage_mitigated'],
                x['damage_player'],
                x['damage_taken'],
                x['deaths'],
                x['healing'],
                x['healing_player_self'],
                x['kills_player'],
                x['structure_damage']
            ]),
            x['match_time_minutes']
        )
    for x in god_data])
    npdata_normalized = npdata / npdata.max(axis=0)
    npdata_winlabel = np.array([1 if x['win_status'] == 'Winner' else 0 for x in god_data])

    sgd_classifier = SGDClassifier(max_iter=1000)
    sgd_classifier.fit(npdata_normalized, npdata_winlabel)

    print('sgd_score:', sgd_classifier.score(npdata_normalized, npdata_winlabel))

    # NOTE original design included a clustering step but after research
    #        I believe simply predicting based on SGD is more than sufficient

    new_winlabel = sgd_classifier.predict(npdata_normalized)

    item_data = np.array([
        [
            1 if item_id in x['item_ids'] else 0
            for item_id in item_ids
        ] for x in god_data
    ])

    item_max = np.max(item_data, axis=0)
    todelete = [idx for idx in range(len(item_max)) if item_max[idx] == 0]
    item_data = np.delete(item_data, todelete, axis=1)

    item_data_ids = [x for idx, x in enumerate(item_ids) if idx not in todelete]

    dt_classifier = DecisionTreeClassifier(criterion='entropy', max_features=1)
    dt_classifier.fit(item_data, new_winlabel)

    print('dt_score:', dt_classifier.score(item_data, new_winlabel))

    bnb_classifier = BernoulliNB()
    bnb_classifier.fit(item_data, new_winlabel)

    print('bnb_score:', bnb_classifier.score(item_data, new_winlabel))

    builds = []
    create_builds(dt_classifier.tree_, 0, [], builds)

    possible_builds = np.array([
        [1 if idx in x else 0 for idx in range(len(item_data_ids))]
        for x in builds
    ])
    dt_predictions = dt_classifier.predict_proba(possible_builds)
    possible_builds = [x for idx, x in enumerate(possible_builds) if dt_predictions[idx][1] > 0.7]

    feature_builds = [
        [idx for idx, x in enumerate(build) if x == 1]
        for build in possible_builds
    ]

    smitebuilds = make_smitebuilds(feature_builds, 4)

    possible_smitebuilds = np.array([
        [1 if idx in x.core else 0 for idx in range(len(item_data_ids))]
        for x in smitebuilds
    ])

    bnb_ranking = bnb_classifier.predict_proba(possible_smitebuilds)
    dt_ranking = dt_classifier.predict_proba(possible_smitebuilds)

    smitebuild_ranks = [
        (x, bnb_ranking[idx][1], dt_ranking[idx][1])
        for idx, x in enumerate(smitebuilds)
    ]
    smitebuild_ranks.sort(key=lambda x: x[1], reverse=True)

    for smitebuild in [x for x in smitebuild_ranks if x[2] > 0.7][:3]:
        print('core:', [id_to_itemname[item_data_ids[x]] for x in smitebuild[0].core])
        print('optional:', [id_to_itemname[item_data_ids[x]] for x in smitebuild[0].optional])
        print('dt_rank:', smitebuild[2])
        print('bnb_rank:', smitebuild[1])


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


if __name__ == '__main__':
    args = parse_args(sys.argv)
    main(args.datapath, args.god)