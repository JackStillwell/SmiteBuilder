from smitebuilder.smitebuild import *

from typing import cast

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
    pass
