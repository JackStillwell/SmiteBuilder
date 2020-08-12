"""
Jack Stillwell
4 August 2020

This is the main interface for SmiteBuilder, providing a complete pipeline for generation of
builds for deities given match information.
"""

import sys
import os

from argparse import ArgumentParser, Namespace
from typing import List, NamedTuple, Optional
from itertools import compress

from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB

from smitebuilder import etl, smiteinfo, dt_tracer
from smitebuilder.smitebuild import (
    rate_smitebuild,
    make_smitebuilds,
    fuse_evolution_items,
    prune_item_data,
    filter_data_by_player_skill
)

from smitebuilder.smiteinfo import MainReturn, ReadableSmiteBuild


def parse_args(args: List[str]) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--datapath", "-d", required=True, type=str)
    parser.add_argument(
        "--queue", "-q", required=True, choices=["conquest", "joust", "duel"], type=str
    )
    parser.add_argument("--god", "-g", required=True, type=str)
    parser.add_argument("--conquest_tier", "-ct", default=15, type=int)
    parser.add_argument("--store_build", "-s", default=None , type=str)

    return parser.parse_known_args(args)[0]


def main(
    path_to_data: str, queue: str, target_god: str, conquest_tier_cutoff: int, store_build: Optional[str]
) -> Optional[List[MainReturn]]:
    # NOTE assumes laid out as in SmiteData repo
    god_map = etl.get_godmap(os.path.join(path_to_data, "gods.json"))
    item_map = etl.get_itemmap(os.path.join(path_to_data, "items.json"))

    queue_path = queue + "_match_data"

    raw_match_data = etl.get_matchdata(
        os.path.join(
            path_to_data, queue_path, str(god_map.inverse[target_god]) + ".json",
        )
    )

    returnval = []

    performance_data = etl.extract_performance_data(raw_match_data)
    win_label = etl.extract_win_label(raw_match_data)
    item_data = etl.extract_item_data(raw_match_data, item_map)

    # prune and consolidate item data
    fuse_evolution_items(item_data, item_map)
    item_mask = prune_item_data(item_data.item_matrix)
    item_data.item_matrix = item_data.item_matrix[:, item_mask]
    item_data.feature_list = list(compress(item_data.feature_list, item_mask))

    while not returnval:

        # add mechanic to relax the filter
        skill_mask = filter_data_by_player_skill(
            raw_match_data, smiteinfo.RankTier(conquest_tier_cutoff)
        )

        performance_data = performance_data[skill_mask, :]
        win_label = win_label[skill_mask, :]
        item_data.item_matrix = item_data.item_matrix[skill_mask, :]

        print("Currently using", performance_data.shape[0], "matches")

        sgd_classifier = SGDClassifier(max_iter=1000, random_state=0)
        sgd_classifier.fit(performance_data, win_label.reshape((win_label.shape[0],)))

        print("sgd_score:", sgd_classifier.score(performance_data, win_label))

        new_winlabel = sgd_classifier.predict(performance_data)

        dt_classifier = DecisionTreeClassifier(
            criterion="entropy", max_features=1, random_state=0,
        )
        dt_classifier.fit(item_data.item_matrix, new_winlabel)

        print("dt_score:", dt_classifier.score(item_data.item_matrix, new_winlabel))

        bnb_classifier = BernoulliNB()
        bnb_classifier.fit(item_data.item_matrix, new_winlabel)

        print("bnb_score:", bnb_classifier.score(item_data.item_matrix, new_winlabel))

        traces = []
        dt_tracer.trace_decision(dt_classifier.tree_, 0, [], traces, 5)

        # turn the traces into smitebuilds
        smitebuilds = make_smitebuilds(traces, 4, item_data.feature_list)

        # rate the smitebuilds
        smitebuild_confidence = [
            (
                x,
                rate_smitebuild(
                    x, item_data.feature_list, dt_classifier, bnb_classifier
                ),
            )
            for x in smitebuilds
        ]

        smitebuild_confidence.sort(key=lambda x: x[1], reverse=True)

        for sb_c in smitebuild_confidence[:3]:
            elem = MainReturn(
                build=ReadableSmiteBuild(
                    core=[item_map[x] for x in sb_c[0].core],
                    optional=[item_map[x] for x in sb_c[0].optional],
                ),
                confidence=sb_c[1],
            )
            returnval.append(elem)
            print("core:", [item_map[x] for x in sb_c[0].core])
            print("optional:", [item_map[x] for x in sb_c[0].optional])
            print("confidence:", sb_c[1])

        if store_build:
            etl.store_build(returnval, os.path.join(store_build, target_god + ".json"))

        return returnval


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(
        args.datapath, args.queue, args.god, args.conquest_tier, args.store_build
    )
