from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode

import nodetool.nodes.lib.data.numpy.visualization


class PlotArray(GraphNode):
    """
    Create a plot visualization of array data.
    array, plot, visualization, graph

    Use cases:
    - Visualize trends in array data
    - Create charts for reports or dashboards
    - Debug array outputs in workflows
    """

    PlotType: typing.ClassVar[type] = (
        nodetool.nodes.lib.data.numpy.visualization.PlotArray.PlotType
    )
    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Array to plot",
    )
    plot_type: nodetool.nodes.lib.data.numpy.visualization.PlotArray.PlotType = Field(
        default=nodetool.nodes.lib.data.numpy.visualization.PlotArray.PlotType.LINE,
        description="Type of plot to create",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.visualization.PlotArray"
