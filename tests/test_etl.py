from bidict import bidict
import numpy as np

from smitebuilder.etl import (
    get_godmap,
    get_itemmap,
    get_matchdata,
    extract_performance_data,
    extract_win_label,
    extract_item_data,
    store_build,
    load_build,
)
from smitebuilder.main import MainReturn, ReadableSmiteBuild 


def test_get_godmap():
    expected = {
        1: "one",
        2: "two",
    }

    actual = get_godmap("tests/test_godmap.json")

    assert expected == actual


def test_get_itemmap():
    expected = {
        1: "one",
        2: "two",
        3: "Evolved three",
        4: "Evolved four",
    }

    actual = get_itemmap("tests/test_itemmap.json")

    assert expected == actual


def test_get_matchdata():
    """Checking that fields are imported correctly and only the relevant information is included"""
    expected = [{"joust_tier": 1,}, {"deaths": 2, "damage_player": 16,}]

    actual = get_matchdata("tests/test_matchdata.json")

    assert expected == actual


raw_matchdata = [
    {
        "assists": 0,
        "damage_mitigated": 1,
        "damage_player": 0,
        "damage_taken": 1,
        "deaths": 0,
        "healing": 0,
        "healing_player_self": 1,
        "kills_player": 0,
        "structure_damage": 0,
        "match_time_minutes": 10,
        "win_status": "Winner",
        "item_ids": [1, 2, 4],
    },
    {
        "assists": 1,
        "damage_mitigated": 0,
        "damage_player": 1,
        "damage_taken": 1,
        "deaths": 0,
        "healing": 1,
        "healing_player_self": 0,
        "kills_player": 1,
        "structure_damage": 0,
        "match_time_minutes": 8,
        "win_status": "Loser",
        "item_ids": [5, 3, 5],
    },
]


def test_extract_performance_data():
    # NOTE to check, -1 means 0 and other had 1, 0 means both had same, 1 means 1 and other had 0
    #      don't forget that the first 1 is a bias column!
    expected = np.array(
        [[1, -1, 1, -1, -1, 0, -1, 1, -1, 0], [1, 1, -1, 1, 1, 0, 1, -1, 1, 0]]
    )

    result = extract_performance_data(raw_matchdata)

    assert np.array_equal(expected, result)


def test_extract_win_label():
    expected = np.array([[1], [0]])
    result = extract_win_label(raw_matchdata)

    assert np.array_equal(expected, result)


item_map = bidict({1: "one", 2: "two", 3: "three", 4: "four", 5: "five"})


def test_extract_item_data():
    expected_matrix = np.array([[1, 1, 0, 1, 0], [0, 0, 1, 0, 1]])
    expected_feature_list = [1, 2, 3, 4, 5]
    result = extract_item_data(raw_matchdata, item_map)

    assert (
        np.array_equal(expected_matrix, result.item_matrix)
        and expected_feature_list == result.feature_list
    )


def test_store_load_build():
    build_to_store = MainReturn(
        build=ReadableSmiteBuild(
            core=["item_one", "item_two"],
            optional=["item_three"]
        ),
        confidence=90.0,
    )

    store_build(build_to_store, "test_storeloadbuild.json")

    result = load_build("test_storeloadbuild.json")

    assert result == build_to_store