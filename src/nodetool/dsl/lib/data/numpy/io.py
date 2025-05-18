from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class SaveArray(GraphNode):
    """
    Save a numpy array to a file in the specified folder.
    array, save, file, storage

    Use cases:
    - Store processed arrays for later use
    - Save analysis results
    - Create checkpoints in processing pipelines
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="The array to save.",
    )
    folder: types.FolderRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.FolderRef(type="folder", uri="", asset_id=None, data=None),
        description="The folder to save the array in.",
    )
    name: str | GraphNode | tuple[GraphNode, str] = Field(
        default="%Y-%m-%d_%H-%M-%S.npy",
        description="\n        The name of the asset to save.\n        You can use time and date variables to create unique names:\n        %Y - Year\n        %m - Month\n        %d - Day\n        %H - Hour\n        %M - Minute\n        %S - Second\n        ",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.io.SaveArray"
