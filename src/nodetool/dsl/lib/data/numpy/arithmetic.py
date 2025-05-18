from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class AddArray(GraphNode):
    """
    Performs addition on two arrays.
    math, plus, add, addition, sum, +
    """

    a: (
        int
        | float
        | nodetool.metadata.types.NPArray
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(default=0.0, description=None)
    b: (
        int
        | float
        | nodetool.metadata.types.NPArray
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(default=0.0, description=None)

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.arithmetic.AddArray"


class BinaryOperation(GraphNode):
    a: (
        int
        | float
        | nodetool.metadata.types.NPArray
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(default=0.0, description=None)
    b: (
        int
        | float
        | nodetool.metadata.types.NPArray
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(default=0.0, description=None)

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.utils.BinaryOperation"


class DivideArray(GraphNode):
    """
    Divides the first array by the second.
    math, division, arithmetic, quotient, /
    """

    a: (
        int
        | float
        | nodetool.metadata.types.NPArray
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(default=0.0, description=None)
    b: (
        int
        | float
        | nodetool.metadata.types.NPArray
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(default=0.0, description=None)

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.arithmetic.DivideArray"


class ModulusArray(GraphNode):
    """
    Calculates the element-wise remainder of division.
    math, modulo, remainder, mod, %

    Use cases:
    - Implementing cyclic behaviors
    - Checking for even/odd numbers
    - Limiting values to a specific range
    """

    a: (
        int
        | float
        | nodetool.metadata.types.NPArray
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(default=0.0, description=None)
    b: (
        int
        | float
        | nodetool.metadata.types.NPArray
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(default=0.0, description=None)

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.arithmetic.ModulusArray"


class MultiplyArray(GraphNode):
    """
    Multiplies two arrays.
    math, product, times, *
    """

    a: (
        int
        | float
        | nodetool.metadata.types.NPArray
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(default=0.0, description=None)
    b: (
        int
        | float
        | nodetool.metadata.types.NPArray
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(default=0.0, description=None)

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.arithmetic.MultiplyArray"


class SubtractArray(GraphNode):
    """
    Subtracts the second array from the first.
    math, minus, difference, -
    """

    a: (
        int
        | float
        | nodetool.metadata.types.NPArray
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(default=0.0, description=None)
    b: (
        int
        | float
        | nodetool.metadata.types.NPArray
        | GraphNode
        | tuple[GraphNode, str]
    ) = Field(default=0.0, description=None)

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.arithmetic.SubtractArray"
