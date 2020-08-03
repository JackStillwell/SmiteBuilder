"""
Jack Stillwell
3 August 2020

The ETL module performs all disk-to-memory conversions required by SmiteBuilder.
This includes reformatting data as well as data pre-processing.
"""

from typing import Dict, List, Optional, Tuple, Union

import json

from bidict import bidict

import numpy as np
from sklearn.preprocessing import StandardScaler


RawMatchData = Dict[str, Optional[Union[int, str, float, List[int]]]]


def get_godmap(path: str) -> bidict(Dict[int, str]):
    """Takes path to json and returns bidirectional map.

    Args:
        path (str): The full path to the return of the getgods (json) endpoint of the SMITE API.

    Returns:
        bidict(Dict[int, str]): A bidirectional map, with primary bindings from ID to Name.
    """
    with open(path, "r") as infile:
        gods = json.loads("".join(infile.readlines()))

    return bidict({x["Name"]: x["id"] for x in gods})


def get_itemmap(path: str) -> bidict(Dict[int, str]):
    """Takes path to json and returns bidirectional map.

    Args:
        path (str): The full path to the return of the getitems (json) endpoint of the SMITE API.

    Returns:
        bidict(Dict[int, str]): A bidirectional map, with primary bindings from ID to Name.
    """
    with open(path, "r") as infile:
        items = json.loads("".join(infile.readlines()))

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
                        1,  # bias feature
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
    scaler = StandardScaler()
    scaler.fit_transform(performance_matrix, copy=False)

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


def extract_item_data(
    raw_data: List[RawMatchData], itemmap: bidict(Dict[int, str])
) -> Tuple[np.ndarray, List[int]]:
    """Takes raw match data and extracts the item data required for DT and BNB classification.

    Args:
        raw_data (List[RawMatchData]): The raw dictionary-style information extracted from json.
        itemmap (bidict(Dict[int, str])): A bidirectional map, with primary bindings from ID to
                                          Name.

    Returns:
        Tuple[np.ndarray, List[int]]: A tuple containing a (# of matches) by (# relevant items) matrix containing
                                      item data from each match and a list mapping feature (column) index to item
                                      id.
    """

    return np.array(
        [1 if x["win_status"] == "Winner" else 0 for x in raw_data]
    ).reshape((len(raw_data), 1))