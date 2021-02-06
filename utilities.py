from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from typing import Any, Dict, NamedTuple
import json

class ColPercentile(NamedTuple):
    ten: float
    ninety: float
        
raw_col_names = (
    'assists', 
    'damage_mitigated', 
    'damage_player', 
    'damage_taken', 
    'deaths', 
    'healing', 
    'healing_player_self', 
    'kills_player', 
    'structure_damage',
)

col_names = (
    'BIAS', 
    'assists', 
    'damage_mitigated', 
    'damage_player', 
    'damage_taken', 
    'deaths', 
    'healing', 
    'healing_player_self', 
    'kills_player', 
    'structure_damage', 
    'net_damage_taken',
    'win_status',
    'match_id',
)

def in_bounds(row: Dict[str, Any], col_perc_dict: Dict[str, ColPercentile]) -> bool:
    inbounds = [False] * len(raw_col_names)
    for idx, col in enumerate(raw_col_names):
        if col_perc_dict[col].ten <= (row[col] / row["match_time_minutes"]) <= col_perc_dict[col].ninety:
            inbounds[idx] = True

    return all(inbounds)

def data_to_training_df(path: str) -> (pd.DataFrame, StandardScaler):
    with open(path) as infile:
        raw_data = [
            x 
            for x in json.loads(''.join(infile.readlines()))
            if x["match_time_minutes"] >= 15
        ]
            
    col_perc_dict = {}
    for col in raw_col_names:
        col_list = [x[col] / x["match_time_minutes"] for x in raw_data]
        col_perc_dict[col] = ColPercentile(
            ten=np.percentile(col_list, 10),
            ninety=np.percentile(col_list, 90)
        )
    data = np.array(
        [
            np.append(
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
                            ((x["damage_mitigated"] + x["healing_player_self"] - x["damage_taken"]) / (x["damage_player"] if x["damage_player"] != 0 else 1)),
                        ]
                    ),
                    x["match_time_minutes"],
                ),
                [
                    1 if x["win_status"] == "Winner" else 0,
                    x["match_id"]
                ]
                
            )
            for x in raw_data if in_bounds(x, col_perc_dict)
        ]
    )
    scaler = StandardScaler(copy=False)
    scaler.fit_transform(data[:,:-2])
    bias = np.ones((data.shape[0], 1))
    data = np.hstack((bias, data))
    df = pd.DataFrame(data, columns=col_names)
    return df, scaler

def data_to_df(path: str, scaler: StandardScaler) -> pd.DataFrame:
    with open(path) as infile:
        raw_data = [
            x 
            for x in json.loads(''.join(infile.readlines()))
            if x["match_time_minutes"] >= 15
        ]
            
    col_perc_dict = {}
    for col in raw_col_names:
        col_list = [x[col] / x["match_time_minutes"] for x in raw_data]
        col_perc_dict[col] = ColPercentile(
            ten=np.percentile(col_list, 10),
            ninety=np.percentile(col_list, 90)
        )
    data = np.array(
        [
            np.append(
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
                            ((x["damage_mitigated"] + x["healing_player_self"] - x["damage_taken"]) / (x["damage_player"] if x["damage_player"] != 0 else 1)),
                        ]
                    ),
                    x["match_time_minutes"],
                ),
                [
                    1 if x["win_status"] == "Winner" else 0,
                    x["match_id"]
                ]
                
            )
            for x in raw_data # if in_bounds(x, col_perc_dict)
        ]
    )
    scaler.transform(data[:,:-2])
    bias = np.ones((data.shape[0], 1))
    data = np.hstack((bias, data))
    df = pd.DataFrame(data, columns=col_names)
    return df