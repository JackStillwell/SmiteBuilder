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
