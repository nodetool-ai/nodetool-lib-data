from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class ArgMaxArray(GraphNode):
    """
    Find indices of maximum values along a specified axis of a array.
    array, argmax, index, maximum

    Use cases:
    - Determine winning classes in classification tasks
    - Find peaks in signal processing
    - Locate best-performing items in datasets
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Input array",
    )
    axis: int | None | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="Axis along which to find maximum indices"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.statistics.ArgMaxArray"


class ArgMinArray(GraphNode):
    """
    Find indices of minimum values along a specified axis of a array.
    array, argmin, index, minimum

    Use cases:
    - Locate lowest-performing items in datasets
    - Find troughs in signal processing
    - Determine least likely classes in classification tasks
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Input array",
    )
    axis: int | None | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="Axis along which to find minimum indices"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.statistics.ArgMinArray"


class MaxArray(GraphNode):
    """
    Compute the maximum value along a specified axis of a array.
    array, maximum, reduction, statistics

    Use cases:
    - Find peak values in time series data
    - Implement max pooling in neural networks
    - Determine highest scores across multiple categories
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Input array",
    )
    axis: int | None | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="Axis along which to compute maximum"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.statistics.MaxArray"


class MeanArray(GraphNode):
    """
    Compute the mean value along a specified axis of a array.
    array, average, reduction, statistics

    Use cases:
    - Calculate average values in datasets
    - Implement mean pooling in neural networks
    - Compute centroids in clustering algorithms
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Input array",
    )
    axis: int | None | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="Axis along which to compute mean"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.statistics.MeanArray"


class MinArray(GraphNode):
    """
    Calculate the minimum value along a specified axis of a array.
    array, minimum, reduction, statistics

    Use cases:
    - Find lowest values in datasets
    - Implement min pooling in neural networks
    - Determine minimum thresholds across categories
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Input array",
    )
    axis: int | None | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="Axis along which to compute minimum"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.statistics.MinArray"


class SumArray(GraphNode):
    """
    Calculate the sum of values along a specified axis of a array.
    array, summation, reduction, statistics

    Use cases:
    - Compute total values across categories
    - Implement sum pooling in neural networks
    - Calculate cumulative metrics in time series data
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Input array",
    )
    axis: int | None | GraphNode | tuple[GraphNode, str] = Field(
        default=None, description="Axis along which to compute sum"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.statistics.SumArray"
