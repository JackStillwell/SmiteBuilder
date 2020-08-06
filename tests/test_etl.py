import json

from smitebuilder.etl import *


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
    }

    actual = get_itemmap("tests/test_itemmap.json")

    assert expected == actual


def test_get_matchdata():
    """Checking that fields are imported correctly and only the relevant information is included"""
    expected = [{"joust_tier": 1,}, {"deaths": 2, "damage_player": 16,}]

    actual = get_matchdata("tests/test_matchdata.json")

    assert expected == actual
