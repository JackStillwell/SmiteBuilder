from smitebuilder.smitebuild import (
    filter_data_by_player_skill,
    fuse_evolution_items,
    prune_item_data,
    make_smitebuilds,
    rate_smitebuild,
    RawMatchData,
    ItemData,
    SmiteBuild,
    gen_all_builds
)

from smitebuilder.smiteinfo import RankTier

from typing import List, cast
from unittest import mock

from bidict import bidict
import numpy as np
import pytest


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


@pytest.mark.parametrize(
    "expected,frequency_cutoff", 
    [
        ([True, True, True], 0.25),
        ([False, True, True], 0.5),
        ([False, False, True], 0.75)
    ]
)
def test_prune_item_data(expected, frequency_cutoff):
    item_matrix = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1],])

    result = prune_item_data(item_matrix, frequency_cutoff=frequency_cutoff)

    assert expected == result


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

@pytest.mark.parametrize(
    "expected,percentile_cutoff", 
    [
        (0.65, 30),
        (0.85, 70),
    ]
)
def test_rate_smitebuild(expected, percentile_cutoff):
    dt_mock = mock.MagicMock()
    dt_mock.predict_proba.return_value = [[0, 1], [1, 0], [0.5, 0.5]]
    bnb_mock = mock.MagicMock()
    bnb_mock.predict_proba.return_value = [[0, 1], [0, 1], [0, 1]]

    build = SmiteBuild(core=set(), optional=set())
    feature_list = []

    result = rate_smitebuild(build, feature_list, dt_mock, bnb_mock, 0.5, 0.5, percentile_cutoff)

    assert expected == result


def test_gen_all_builds():
    build = SmiteBuild(core={12, 78, 90, 56}, optional={11, 22, 33, 44})

    expected = [
        {12, 78, 90, 56, 11, 22},
        {12, 78, 90, 56, 11, 33},
        {12, 78, 90, 56, 11, 44},
        {12, 78, 90, 56, 22, 33},
        {12, 78, 90, 56, 22, 44},
        {12, 78, 90, 56, 33, 44},
    ]

    result = gen_all_builds(build)

    assert len(expected) == len(result) and all([x in expected for x in result])