from smitebuilder.smitebuild import (
    filter_data_by_player_skill,
    fuse_evolution_items,
    prune_item_data,
    rate_smitebuildpath,
    RawMatchData,
    ItemData,
    SmiteBuildPath,
    gen_all_builds,
    select_builds,
    build_similarity,
    find_common_cores,
    get_options,
    prune_options,
    consolidate_options,
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
        ([False, False, True], 0.75),
    ],
)
def test_prune_item_data(expected, frequency_cutoff):
    item_matrix = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1],])

    result = prune_item_data(item_matrix, frequency_cutoff=frequency_cutoff)

    assert expected == result


@pytest.mark.parametrize("expected,percentile_cutoff", [(0.65, 30), (0.85, 70),])
def test_rate_smitebuildpath(expected, percentile_cutoff):
    dt_mock = mock.MagicMock()
    dt_mock.predict_proba.return_value = [[0, 1], [1, 0], [0.5, 0.5]]
    bnb_mock = mock.MagicMock()
    bnb_mock.predict_proba.return_value = [[0, 1], [0, 1], [0, 1]]

    build = SmiteBuildPath(core=frozenset({1}), optionals={frozenset({2})})
    feature_list = []

    result = rate_smitebuildpath(
        build, feature_list, dt_mock, bnb_mock, 0.5, 0.5, percentile_cutoff
    )

    assert expected == pytest.approx(result)


def test_gen_all_builds():
    build = SmiteBuildPath(
        core=frozenset({12, 78, 90, 56}), optionals={frozenset({11, 22, 33, 44})}
    )

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


similarity_data = [
    SmiteBuildPath(core=frozenset({1, 2, 3, 4}), optionals={frozenset({5, 6, 7})}),
    SmiteBuildPath(core=frozenset({1, 6, 3, 5}), optionals={frozenset({2, 4, 7})}),
    SmiteBuildPath(core=frozenset({11, 12, 13, 14}), optionals={frozenset({15, 16})}),
    SmiteBuildPath(
        core=frozenset({1, 2, 3, 4}),
        optionals={frozenset({5, 6, 7}), frozenset({2, 4, 7})},
    ),
]


@pytest.mark.parametrize(
    "num,expected,similarity_cutoff",
    [
        (4, similarity_data, 1.0),
        (4, similarity_data[:3], 0.5),
        (4, [similarity_data[0], similarity_data[2]], 0.0),
        (2, similarity_data[:2], 1.0),
        (2, similarity_data[:2], 0.5),
        (2, [similarity_data[0], similarity_data[2]], 0.0),
    ],
)
def test_select_builds(num, expected, similarity_cutoff):
    result = select_builds(similarity_data, num, similarity_cutoff)

    assert result == expected


@pytest.mark.parametrize(
    "builds,expected",
    [
        ((similarity_data[0], similarity_data[1]), 0.2),
        ((similarity_data[1], similarity_data[2]), 0),
        ((similarity_data[0], similarity_data[0]), 1),
    ],
)
def test_build_similarity(builds, expected):
    result = build_similarity(builds[0], builds[1])

    assert expected == pytest.approx(result)


traces = [
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 7, 8],
    [1, 2, 3, 5, 6, 7],
    [1, 2, 3, 6, 7, 8],
]


@pytest.mark.parametrize(
    "core_length,num_cores,expected",
    [
        (4, 2, {frozenset({1, 2, 3, 7}), frozenset({1, 2, 3, 6})},),
        (4, None, 44),  # every possible core based on the items present
    ],
)
def test_find_common_cores(core_length, num_cores, expected):
    result = find_common_cores(traces, core_length, num_cores)

    if type(expected) is set:
        assert expected == result
    elif type(expected) is int:
        assert expected == len(result)


@pytest.mark.parametrize(
    "core,expected",
    [
        (frozenset({1, 2, 3, 4}), {frozenset({5, 6}), frozenset({7, 8})}),
        (
            frozenset({1, 2, 3, 6}),
            {frozenset({4, 5}), frozenset({5, 7}), frozenset({7, 8})},
        ),
    ],
)
def test_get_options(core, expected):
    result = get_options(traces, core)
    assert result == expected


@pytest.mark.parametrize(
    "build_ranks, cutoff, expected",
    [
        ([1.0, 2.0, 3.0], 70, {frozenset({6, 7})}),
        ([1.0, 2.0, 3.0], 30, {frozenset({5, 7}), frozenset({6, 7})}),
        (
            [1.0, 2.0, 3.0],
            0,
            {frozenset({5, 6}), frozenset({5, 7}), frozenset({6, 7})},
        ),
    ],
)
def test_prune_options(build_ranks, cutoff, expected):
    core = frozenset({1, 2, 3, 4})
    optionals = {frozenset({5, 6}), frozenset({5, 7}), frozenset({6, 7})}

    result = prune_options(core, optionals, lambda x: build_ranks, cutoff)

    assert result == expected


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, {}),
        (
            {frozenset({1, 2}), frozenset({1, 3}), frozenset({2, 3})},
            {frozenset({1, 2, 3})},
        ),
    ],
)
def test_consolidate_options(options, expected):
    result = consolidate_options(options)

    assert result == expected
