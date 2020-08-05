"""
Jack Stillwell
4 August 2020

This is the main interface for SmiteBuilder, providing a complete pipeline for generation of
builds for deities given match information.
"""

import sys
import os

from argparse import ArgumentParser, Namespace
from typing import List, NamedTuple, Tuple, Optional
from itertools import compress

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB

from smitebuilder import etl, smiteinfo, dt_tracer
from smitebuilder.smitebuild import *


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


class ReadableSmiteBuild(NamedTuple):
    core: List[str]
    optional: List[str]


class MainReturn(NamedTuple):
    build: ReadableSmiteBuild
    confidence: float


def main(
    path_to_data: str,
    queue: str,
    target_god: str,
    conquest_tier_cutoff: int,
    probability_score_limit: float,
    probability_score_cutoff: float,
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
    item_mask = prune_item_data(item_data.item_matrix)
    preprocessed_item_data = ItemData(
        item_matrix=fuse_evolution_items(
            np.delete(item_data.item_matrix, item_mask, axis=1), item_map
        ),
        feature_list=list(compress(item_data.feature_list, item_mask)),
    )
    item_data = preprocessed_item_data

    while not returnval:

        # add mechanic to relax the filter
        skill_mask = filter_data_by_player_skill(
            raw_match_data, smiteinfo.RankTier(conquest_tier_cutoff)
        )

        item_mask = prune_item_data(item_data)

        sgd_classifier = SGDClassifier(max_iter=1000, random_state=0)
        sgd_classifier.fit(performance_data, win_label)

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
        ][:3]

        for sb_c in smitebuild_confidence:
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

        return returnval


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
