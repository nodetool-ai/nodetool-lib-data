from pydantic import Field
import nodetool.metadata.types
from nodetool.dsl.graph import GraphNode


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
        return "lib.numpy.utils.BinaryOperation"
