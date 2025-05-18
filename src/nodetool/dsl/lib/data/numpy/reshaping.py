from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class Reshape1D(GraphNode):
    """
    Reshape an array to a 1D shape without changing its data.
    array, reshape, vector, flatten

    Use cases:
    - Flatten multi-dimensional data for certain algorithms
    - Convert images to vector form for machine learning
    - Prepare data for 1D operations
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="The input array to reshape",
    )
    num_elements: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="The number of elements"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.reshaping.Reshape1D"


class Reshape2D(GraphNode):
    """
    Reshape an array to a new shape without changing its data.
    array, reshape, dimensions, structure

    Use cases:
    - Convert between different dimensional representations
    - Prepare data for specific model architectures
    - Flatten or unflatten arrays
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="The input array to reshape",
    )
    num_rows: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="The number of rows"
    )
    num_cols: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="The number of columns"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.reshaping.Reshape2D"


class Reshape3D(GraphNode):
    """
    Reshape an array to a 3D shape without changing its data.
    array, reshape, dimensions, volume

    Use cases:
    - Convert data for 3D visualization
    - Prepare image data with channels
    - Structure data for 3D convolutions
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="The input array to reshape",
    )
    num_rows: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="The number of rows"
    )
    num_cols: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="The number of columns"
    )
    num_depths: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="The number of depths"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.reshaping.Reshape3D"


class Reshape4D(GraphNode):
    """
    Reshape an array to a 4D shape without changing its data.
    array, reshape, dimensions, batch

    Use cases:
    - Prepare batch data for neural networks
    - Structure spatiotemporal data
    - Format data for 3D image processing with channels
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="The input array to reshape",
    )
    num_rows: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="The number of rows"
    )
    num_cols: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="The number of columns"
    )
    num_depths: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="The number of depths"
    )
    num_channels: int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="The number of channels"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.reshaping.Reshape4D"
