from dataclasses import dataclass
from typing import List

from smitebuilder.dt_tracer import trace_decision


@dataclass
class Tree:
    feature: List[int]
    children_left: List[int]
    children_right: List[int]


tree = Tree(
    feature=list(range(15)),
    children_left=[1, 3, 5, 7, 9, 11, 13, -1, -1, -1, -1, -1, -1, -1, -1],
    children_right=[2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1],
)


def test_trace_decision_complete():
    """Ensure that traces are correct and that all are found given a great enough trace length."""
    traces = []
    trace_decision(tree, 0, [], traces, 4)

    expected_traces = [
        [0, 1, 2, 7],
        [0, 1, 3, 8],
        [0, 1, 4, 9],
        [0, 1, 4, 10],
        [0, 2, 5, 11],
        [0, 2, 5, 12],
        [0, 2, 6, 13],
        [0, 2, 6, 14],
    ]

    assert len(traces) == len(expected_traces)
    assert all([x in expected_traces for x in traces])


def test_trace_decision_length():
    """Ensure that traces are limited to the correct length"""
    traces = []
    trace_decision(tree, 0, [], traces, 3)

    expected_traces = [
        [0, 1, 3],
        [0, 1, 4],
        [0, 2, 5],
        [0, 2, 6],
    ]

    assert len(traces) == len(expected_traces)
    assert all([x in expected_traces for x in traces])

