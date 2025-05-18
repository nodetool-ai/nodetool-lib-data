from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class AbsArray(GraphNode):
    """
    Compute the absolute value of each element in a array.
    array, absolute, magnitude

    Use cases:
    - Calculate magnitudes of complex numbers
    - Preprocess data for certain algorithms
    - Implement activation functions in neural networks
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="The input array to compute the absolute values from.",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.math.AbsArray"


class CosineArray(GraphNode):
    """
    Computes the cosine of input angles in radians.
    math, trigonometry, cosine, cos

    Use cases:
    - Calculating horizontal components in physics
    - Creating circular motions
    - Phase calculations in signal processing
    """

    angle_rad: (
        float
        | int
        | nodetool.metadata.types.NPArray
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(default=0.0, description=None)

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.math.CosineArray"


class ExpArray(GraphNode):
    """
    Calculate the exponential of each element in a array.
    array, exponential, math, activation

    Use cases:
    - Implement exponential activation functions
    - Calculate growth rates in scientific models
    - Transform data for certain statistical analyses
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Input array",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.math.ExpArray"


class LogArray(GraphNode):
    """
    Calculate the natural logarithm of each element in a array.
    array, logarithm, math, transformation

    Use cases:
    - Implement log transformations on data
    - Calculate entropy in information theory
    - Normalize data with large ranges
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Input array",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.math.LogArray"


class PowerArray(GraphNode):
    """
    Raises the base array to the power of the exponent element-wise.
    math, exponentiation, power, pow, **

    Use cases:
    - Calculating compound interest
    - Implementing polynomial functions
    - Applying non-linear transformations to data
    """

    base: (
        float
        | int
        | nodetool.metadata.types.NPArray
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(default=1.0, description=None)
    exponent: (
        float
        | int
        | nodetool.metadata.types.NPArray
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(default=2.0, description=None)

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.math.PowerArray"


class SineArray(GraphNode):
    """
    Computes the sine of input angles in radians.
    math, trigonometry, sine, sin

    Use cases:
    - Calculating vertical components in physics
    - Generating smooth periodic functions
    - Audio signal processing
    """

    angle_rad: (
        float
        | int
        | nodetool.metadata.types.NPArray
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(default=0.0, description=None)

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.math.SineArray"


class SqrtArray(GraphNode):
    """
    Calculates the square root of the input array element-wise.
    math, square root, sqrt, √

    Use cases:
    - Normalizing data
    - Calculating distances in Euclidean space
    - Finding intermediate values in binary search
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Input array",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.math.SqrtArray"
