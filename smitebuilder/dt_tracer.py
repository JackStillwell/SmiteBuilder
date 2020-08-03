"""
Jack Stillwell
3 August 2020

The Decision Tree Configuration Builder module contains the code necessary to generate
recommended configurations from a trained decision tree.
"""

from copy import deepcopy
from typing import List


def trace_decision(
    tree, node: int, local_trace: List[int], traces: List[List[int]], trace_length: int
):
    """
    NEEDS DOCSTRING
    """

    # this stops if we hit a leaf node
    if tree.children_left[node] == tree.children_right[node]:
        # if the trace is not long enough, discard it
        if len(local_trace) < trace_length:
            return None
        else:
            traces.append(local_trace)
            return None

    # this stops if we hit the trace length
    if len(local_trace) > 4:
        traces.append(local_trace)
        return None

    # the left node indicates the abscence of a feature
    trace_decision(
        tree, tree.children_left[node], deepcopy(local_trace), traces, trace_length
    )

    # the right node indicates the prescence of a feature
    local_trace.append(tree.feature[node])
    trace_decision(
        tree, tree.children_right[node], deepcopy(local_trace), traces, trace_length
    )
