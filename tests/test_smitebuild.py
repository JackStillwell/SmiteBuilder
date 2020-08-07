from smitebuilder.smitebuild import *

from typing import cast
from unittest import mock

from bidict import bidict
import numpy as np


def test_filter_data_by_player_skill():
    data = cast(
        List[RawMatchData],
        [
            {"conquest_tier": 0,},
            {"conquest_tier": 11,},
            {"conquest_tier": 18,},
            {"conquest_tier": 26,},
        ],
    )

    conquest_tier_cutoff = 15
    expected = [0, 0, 1, 1]
    result = filter_data_by_player_skill(data, RankTier(conquest_tier_cutoff))

    assert expected == result


def test_fuse_evolution_items():
    item_data = ItemData(
        item_matrix=np.array(
            [[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 0], [0, 1, 0]]
        ),
        feature_list=[12, 34, 56],
    )

    item_map = bidict({12: "Evolved Thing", 34: "Thing", 56: "Thing2"})

    expected = ItemData(
        item_matrix=np.array(
            [[0, 1, 1], [0, 1, 1], [0, 0, 1], [0, 1, 0], [0, 0, 0], [0, 1, 0]]
        ),
        feature_list=[12, 34, 56],
    )

    # NOTE works in-place
    fuse_evolution_items(item_data, item_map)

    assert (
        np.array_equal(item_data.item_matrix, expected.item_matrix)
        and item_data.feature_list == expected.feature_list
    )


def test_prune_item_data():
    item_matrix = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1],])

    expected25 = [True, True, True]
    expected50 = [False, True, True]
    expected75 = [False, False, True]

    result25 = prune_item_data(item_matrix, frequency_cutoff=0.25)
    result50 = prune_item_data(item_matrix, frequency_cutoff=0.5)
    result75 = prune_item_data(item_matrix, frequency_cutoff=0.75)

    assert expected25 == result25 and expected50 == result50 and expected75 == result75


def test_make_smitebuilds():
    traces = [
        [0, 3, 4],
        [0, 3, 2],
        [0, 1, 4],
    ]
    feature_list = [12, 34, 56, 78, 90]

    result = make_smitebuilds(traces, 2, feature_list)
    expected = [
        SmiteBuild(core={12, 78}, optional={90, 56}),
        SmiteBuild(core={12, 90}, optional={34, 78}),
    ]

    assert expected == result


def test_rate_smitebuild():
    dt_mock = mock.MagicMock()
    dt_mock.predict_proba.return_value = [[0, 1]]
    bnb_mock = mock.MagicMock()
    bnb_mock.predict_proba.return_value = [[1, 0]]

    build = SmiteBuild(core={12, 78}, optional={90, 56})
    feature_list = [12, 34, 56, 78, 90]

    expected = 0.66
    result = rate_smitebuild(build, feature_list, dt_mock, bnb_mock)

    assert expected == result
