"""
Jack Stillwell
4 August 2020

This is the main interface for SmiteBuilder, providing a complete pipeline for generation of
builds for deities given match information.
"""

from smitebuilder.etl import load_build
import sys
import os

from argparse import ArgumentParser, Namespace
from typing import List, Optional
from itertools import compress

from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB

from smitebuilder import etl, smiteinfo, dt_tracer
from smitebuilder.smitebuild import (
    feature_to_item,
    rate_builds,
    fuse_evolution_items,
    prune_item_data,
    filter_data_by_player_skill,
    select_builds,
    find_common_cores,
    get_options,
    consolidate_options,
    prune_options,
    rate_smitebuildpath,
)
from smitebuilder.smiteinfo import (
    MainReturn,
    ReadableSmiteBuild,
    SmiteBuildPath,
    ReadableSmiteBuildPath,
)


def parse_args(args: List[str]) -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--datapath", "-d", required=True, type=str)
    parser.add_argument(
        "--queue", "-q", required=True, choices=["conquest", "joust", "duel"], type=str
    )
    parser.add_argument("--god", "-g", required=True, type=str)
    parser.add_argument("--conquest_tier", "-ct", default=15, type=int)
    parser.add_argument(
        "--store_build", "-s", default=False, choices=[True, False], type=bool
    )
    parser.add_argument("--silent", default=False, choices=[True, False], type=bool)

    return parser.parse_known_args(args)[0]


def main(
    path_to_data: str,
    queue: str,
    target_god: str,
    conquest_tier_cutoff: int,
    store_build: bool,
    silent: bool,
) -> Optional[List[MainReturn]]:
    # NOTE assumes laid out as in SmiteData repo
    god_map = etl.get_godmap(os.path.join(path_to_data, "gods.json"))
    item_map = etl.get_itemmap(os.path.join(path_to_data, "items.json"))

    queue_path = queue + "_match_data"

    match_data_path = os.path.join(
        path_to_data, queue_path, str(god_map.inverse[target_god]) + ".json",
    )

    # test if build exists or needs to be generated
    if store_build:
        build_path = os.path.join(path_to_data, queue + "_builds", target_god + ".json")
        if os.path.isfile(build_path):
            build_time = os.path.getmtime(build_path)
            data_time = os.path.getmtime(match_data_path)

            # if the build is newer than the data
            if (build_time - data_time) > 0:
                build = load_build(build_path)
                if not silent:
                    print(build)
                return build

    raw_match_data = etl.get_matchdata(match_data_path)
    returnval = []

    performance_data = etl.extract_performance_data(raw_match_data)
    win_label = etl.extract_win_label(raw_match_data)
    item_data = etl.extract_item_data(raw_match_data, item_map)

    # prune and consolidate item data
    fuse_evolution_items(item_data, item_map)
    item_mask = prune_item_data(item_data.item_matrix)
    item_data.item_matrix = item_data.item_matrix[:, item_mask]
    item_data.feature_list = list(compress(item_data.feature_list, item_mask))

    # add mechanic to relax the filter
    skill_mask = filter_data_by_player_skill(
        raw_match_data, smiteinfo.RankTier(conquest_tier_cutoff)
    )

    performance_data = performance_data[skill_mask, :]
    win_label = win_label[skill_mask, :]
    item_data.item_matrix = item_data.item_matrix[skill_mask, :]

    if not silent:
        print("Currently using", performance_data.shape[0], "matches")

    sgd_classifier = SGDClassifier(max_iter=1000, random_state=0)
    sgd_classifier.fit(performance_data, win_label.reshape((win_label.shape[0],)))

    if not silent:
        print("sgd_score:", sgd_classifier.score(performance_data, win_label))

    new_winlabel = sgd_classifier.predict(performance_data)

    dt_classifier = DecisionTreeClassifier(
        criterion="entropy", max_features=1, random_state=0,
    )
    dt_classifier.fit(item_data.item_matrix, new_winlabel)
    dt_score = dt_classifier.score(item_data.item_matrix, new_winlabel)

    if not silent:
        print("dt_score:", dt_score)

    bnb_classifier = BernoulliNB()
    bnb_classifier.fit(item_data.item_matrix, new_winlabel)
    bnb_score = bnb_classifier.score(item_data.item_matrix, new_winlabel)

    if not silent:
        print("bnb_score:", bnb_score)

    traces = []
    dt_tracer.trace_decision(dt_classifier.tree_, 0, [], traces, 5)

    item_ids = feature_to_item(traces, item_data.feature_list)

    # turn the traces into common cores
    cores = find_common_cores(item_ids, 4, None)

    dt_percentage = dt_score / (dt_score + bnb_score)
    bnb_percentage = 1.0 - dt_percentage
    rate_builds_lambda = lambda x: rate_builds(
        x,
        item_data.feature_list,
        dt_classifier,
        bnb_classifier,
        dt_percentage,
        bnb_percentage,
    )

    build_paths = [
        SmiteBuildPath(
            core=x,
            optionals=consolidate_options(
                prune_options(x, (get_options(item_ids, x)), rate_builds_lambda, 75)
            ),
        )
        for x in cores
    ]

    rate_smitebuild_lambda = lambda x: rate_smitebuildpath(
        x,
        item_data.feature_list,
        dt_classifier,
        bnb_classifier,
        dt_percentage,
        bnb_percentage,
        30,  # 70% of the scores must be above this number
    )

    # rate the smitebuilds
    smitebuild_confidence = [(x, rate_smitebuild_lambda(x)) for x in build_paths]

    smitebuild_confidence.sort(key=lambda x: x[1], reverse=True)

    final_builds = select_builds([x[0] for x in smitebuild_confidence], 3, 0.25)

    readable_paths = [
        (
            ReadableSmiteBuildPath.from_SmiteBuildPath(x, item_map),
            rate_smitebuild_lambda(x),
        )
        for x in final_builds
    ]

    for build, rating in readable_paths:
        elem = MainReturn(build=build, confidence=rating,)
        returnval.append(elem)

        if not silent:
            print(elem)

    if store_build:
        etl.store_build(
            returnval,
            os.path.join(path_to_data, queue + "_builds", target_god + ".json"),
        )

    return returnval


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(
        args.datapath,
        args.queue,
        args.god,
        args.conquest_tier,
        args.store_build,
        args.silent,
    )
