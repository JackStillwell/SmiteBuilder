"""
Jack Stillwell
3 August 2020

The ETL module performs all disk-to-memory conversions required by SmiteBuilder.
This includes reformatting data as well as data pre-processing.
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass

import json

from bidict import bidict

import numpy as np
from sklearn.preprocessing import StandardScaler

from smitebuilder.smiteinfo import MainReturn, ReadableSmiteBuild


RawMatchData = Dict[str, Optional[Union[int, str, float, List[int]]]]


def get_godmap(path: str) -> Dict[int, str]:
    """Takes path to json and returns bidirectional map.

    Args:
        path (str): The full path to the return of the getgods (json) endpoint of the SMITE API.

    Returns:
        Dict[int, str]: A bidirectional map, with primary bindings from ID to Name.
    """
    with open(path, "r") as infile:
        gods = json.loads("".join(infile.readlines()))

    return bidict({x["id"]: x["Name"] for x in gods})


def get_itemmap(path: str) -> Dict[int, str]:
    """Takes path to json and returns bidirectional map.

    Args:
        path (str): The full path to the return of the getitems (json) endpoint of the SMITE API.

    Returns:
        Dict[int, str]: A bidirectional map, with primary bindings from ID to Name.
    """
    with open(path, "r") as infile:
        items = json.loads("".join(infile.readlines()))

    for item in items:
        if item["ItemTier"] == 4 and not item["DeviceName"].startswith("Evolved"):
            item["DeviceName"] = "Evolved " + item["DeviceName"]

    return bidict({x["ItemId"]: x["DeviceName"] for x in items})


def get_matchdata(path: str) -> List[RawMatchData]:
    """Given the path to a json file containing match information for a particular god, return a
    triplet of matrices containing the preprocessed information for that god's match data.

    Args:
        path (str): The full path to a json file containing match data for a particular god.

    Returns:
        List[RawMatchData]: A list of dictionares containing the information required to run
                            SmiteBuilder.
    """
    with open(path, "r") as infile:
        raw_data: List[RawMatchData] = json.loads("".join(infile.readlines()))

    relevant_information = [
        "conquest_tier",
        "duel_tier",
        "joust_tier",
        "rank_stat_conquest",
        "rank_stat_duel",
        "rank_stat_joust",
        "assists",
        "damage_mitigated",
        "damage_player",
        "damage_taken",
        "deaths",
        "healing",
        "healing_player_self",
        "kills_player",
        "structure_damage",
        "win_status",
        "item_ids",
        "match_time_minutes",
    ]

    return [{k: v for k, v in x.items() if k in relevant_information} for x in raw_data]


def extract_performance_data(raw_data: List[RawMatchData]) -> np.ndarray:
    """Takes raw match data and extracts the performance data required for SGDClassification.

    Args:
        raw_data (List[RawMatchData]): The raw dictionary-style information extracted from json.

    Returns:
        np.ndarray: A (# of matches) by 10 matrix containing performance data from each match.
    """
    # extract the required data
    performance_matrix = np.array(
        [
            np.divide(
                np.array(
                    [
                        x["assists"],
                        x["damage_mitigated"],
                        x["damage_player"],
                        x["damage_taken"],
                        x["deaths"],
                        x["healing"],
                        x["healing_player_self"],
                        x["kills_player"],
                        x["structure_damage"],
                    ]
                ),
                x["match_time_minutes"],
            )
            for x in raw_data
        ]
    )

    # normalize the data (improves SGD classification)
    scaler = StandardScaler(copy=False)
    scaler.fit_transform(performance_matrix)

    # add a bias feature (must be after scaling)
    performance_matrix = np.concatenate(
        (np.ones((performance_matrix.shape[0], 1)), performance_matrix), axis=1
    )

    return performance_matrix


def extract_win_label(raw_data: List[RawMatchData]) -> np.ndarray:
    """Takes raw match data and extracts the win label for each match.

    Args:
        raw_data (List[RawMatchData]): The raw dictionary-style information extracted from json.

    Returns:
        np.ndarray: A (# of matches) by 1 matrix containing the win label for each match.
    """

    return np.array(
        [1 if x["win_status"] == "Winner" else 0 for x in raw_data]
    ).reshape((len(raw_data), 1))


@dataclass
class ItemData:
    item_matrix: np.ndarray
    feature_list: List[int]


def extract_item_data(
    raw_data: List[RawMatchData], itemmap: Dict[int, str]
) -> ItemData:
    """Takes raw match data and extracts the item data required for DT and BNB classification.

    Args:
        raw_data (List[RawMatchData]): The raw dictionary-style information extracted from json.
        itemmap (Dict[int, str]): A bidirectional map, with primary bindings from ID to
                                          Name.

    Returns:
        ItemData: A tuple containing a (# of matches) by (# relevant items) matrix containing
                  item data from each match and a list mapping feature (column) index to item
                  id.
    """

    item_data = np.array(
        [
            [1 if item_id in x["item_ids"] else 0 for item_id in itemmap.keys()]
            for x in raw_data
        ]
    )

    return ItemData(item_matrix=item_data, feature_list=list(itemmap.keys()))


def store_build(build: List[MainReturn], path: str):
    with open(path, "w") as outfile:
        outfile.write(json.dumps(build))


def load_build(path: str) -> List[MainReturn]:
    with open(path, "r") as infile:
        raw_list = json.loads("".join(infile.readlines()))

    return [MainReturn(
        build=ReadableSmiteBuild(
            core=x[0][0],
            optional=x[0][1],
        ),
        confidence=x[1],
    ) for x in raw_list]
