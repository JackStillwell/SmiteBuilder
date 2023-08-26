"""
Jack Stillwell
3 August 2020

The ETL module performs all disk-to-memory conversions required by SmiteBuilder.
This includes reformatting data as well as data pre-processing.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import json

from bidict import bidict

import numpy as np
from sklearn.preprocessing import StandardScaler

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from smitebuilder.smiteinfo import MainReturn, ReadableSmiteBuildPath

RawMatchData = Dict[str, Optional[Union[int, str, float, List[int]]]]


def get_godmap(path: str) -> Dict[int, str]:
    """Takes path to json and returns bidirectional map.

    Args:
        path (str): The full path to the return of the getgods (json) endpoint of the SMITE API.

    Returns:
        Dict[int, str]: A bidirectional map, with primary bindings from ID to Name.
    """
    with open(path, "r", encoding="utf-8") as infile:
        gods = json.load(infile)

    return bidict({x["id"]: x["Name"] for x in gods})


def get_itemmap(path: str) -> Tuple[Dict[int, str], Dict[int, int]]:
    """Takes path to json and returns bidirectional map.

    Args:
        path (str): The full path to the return of the getitems (json) endpoint of the SMITE API.

    Returns:
        Dict[int, str]: A bidirectional map, with primary bindings from ID to Name.
        Dict[int, int]: A map from all tier 4 item ids to their tier 3 ids.
    """
    with open(path, "r", encoding="utf-8") as infile:
        items = json.loads("".join(infile.readlines()))

    return (
        bidict(
            {
                x["ItemId"]: x["DeviceName"]
                for x in items
                if x["ActiveFlag"] == "y" and (x["ItemTier"] >= 3 or x["StartingItem"])
            }
        ),
        {x["ItemId"]: x["ChildItemId"] for x in items if x["ItemTier"] == 4},
    )


def get_matchdata(
    path: str,
    god_id: int,
    mmr_floor: int,
    conquest_tier_floor: int,
    role: str,
) -> List[RawMatchData]:
    """Given the path to a json file containing mongo auth information, return a
    triplet of matrices containing the preprocessed information for that god's match data.

    Args:
        path (str): The full path to a json file containing mongo auth information.

    Returns:
        List[RawMatchData]: A list of dictionares containing the information required to run
                            SmiteBuilder.
    """
    with open(path, "r") as infile:
        mongo_auth = json.loads("".join(infile.readlines()))

    uri = f"mongodb://{mongo_auth['username']}:{mongo_auth['password']}@{mongo_auth['clusterAddress']}/?{mongo_auth['uriArgs']}"
    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi("1"))
    # Send a ping to confirm a successful connection
    try:
        client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        exit(e)

    db = client.SmiteBuilds
    coll = db.MatchDetails

    raw_data = list(
        coll.find(
            {
                "GodId": god_id,
                "Role": role,
                "Final_Match_Level": {"$gte": 12},
                "Rank_Stat_Conquest": {"$gte": mmr_floor},
                "Conquest_Tier": {"$gte": conquest_tier_floor},
            }
        ).sort("Match", -1)
        # .limit(2000)
    )

    for x in raw_data:
        item_ids = [x["ItemId" + str(i)] for i in range(1, 7)]
        x["item_ids"] = item_ids

    relevant_information = {
        "Assists": "assists",
        "Damage_Mitigated": "damage_mitigated",
        "Damage_Player": "damage_player",
        "Damage_Taken": "damage_taken",
        "Deaths": "deaths",
        "Healing": "healing",
        "Healing_Player_Self": "healing_player_self",
        "Kills_Player": "kills_player",
        "Structure_Damage": "structure_damage",
        "Win_Status": "win_status",
        "item_ids": "item_ids",
        "Minutes": "match_time_minutes",
    }

    raw_data = [
        {
            relevant_information[k]: v
            for k, v in x.items()
            if k in relevant_information.keys()
        }
        for x in raw_data
    ]
    raw_data = [x for x in raw_data if x["match_time_minutes"] > 0]

    return raw_data


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

    return [
        MainReturn(
            build=ReadableSmiteBuildPath(
                core=x[0][0],
                optionals=x[0][1],
            ),
            confidence=x[1],
        )
        for x in raw_list
    ]
