from pydantic import BaseModel, Field
import typing
from typing import Any
import nodetool.metadata.types
import nodetool.metadata.types as types
from nodetool.dsl.graph import GraphNode


class ArrayToList(GraphNode):
    """
    Convert a array to a nested list structure.
    array, list, conversion, type

    Use cases:
    - Prepare array data for JSON serialization
    - Convert array outputs to Python data structures
    - Interface array data with non-array operations
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Array to convert to list",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.conversion.ArrayToList"


class ArrayToScalar(GraphNode):
    """
    Convert a single-element array to a scalar value.
    array, scalar, conversion, type

    Use cases:
    - Extract final results from array computations
    - Prepare values for non-array operations
    - Simplify output for human-readable results
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="Array to convert to scalar",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.conversion.ArrayToScalar"


class ConvertToArray(GraphNode):
    """
    Convert PIL Image to normalized tensor representation.
    image, tensor, conversion, normalization

    Use cases:
    - Prepare images for machine learning models
    - Convert between image formats for processing
    - Normalize image data for consistent calculations
    """

    image: types.ImageRef | GraphNode | tuple[GraphNode, str] = Field(
        default=types.ImageRef(type="image", uri="", asset_id=None, data=None),
        description="The input image to convert to a tensor. The image should have either 1 (grayscale), 3 (RGB), or 4 (RGBA) channels.",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.conversion.ConvertToArray"


class ConvertToAudio(GraphNode):
    """
    Converts a array object back to an audio file.
    audio, conversion, array

    Use cases:
    - Save processed audio data as a playable file
    - Convert generated or modified audio arrays to audio format
    - Output results of audio processing pipelinesr
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="The array to convert to an audio file.",
    )
    sample_rate: int | GraphNode | tuple[GraphNode, str] = Field(
        default=44100, description="The sample rate of the audio file."
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.conversion.ConvertToAudio"


class ConvertToImage(GraphNode):
    """
    Convert array data to PIL Image format.
    array, image, conversion, denormalization

    Use cases:
    - Visualize array data as images
    - Save processed array results as images
    - Convert model outputs back to viewable format
    """

    values: types.NPArray | GraphNode | tuple[GraphNode, str] = Field(
        default=types.NPArray(type="np_array", value=None, dtype="<i8", shape=(1,)),
        description="The input array to convert to an image. Should have either 1, 3, or 4 channels.",
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.conversion.ConvertToImage"


class ListToArray(GraphNode):
    """
    Convert a list of values to a array.
    list, array, conversion, type

    Use cases:
    - Prepare list data for array operations
    - Create arrays from Python data structures
    - Convert sequence data to array format
    """

    values: list[Any] | GraphNode | tuple[GraphNode, str] = Field(
        default=[], description="List of values to convert to array"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.conversion.ListToArray"


class ScalarToArray(GraphNode):
    """
    Convert a scalar value to a single-element array.
    scalar, array, conversion, type

    Use cases:
    - Prepare scalar inputs for array operations
    - Create constant arrays for computations
    - Initialize array values in workflows
    """

    value: float | int | GraphNode | tuple[GraphNode, str] = Field(
        default=0, description="Scalar value to convert to array"
    )

    @classmethod
    def get_node_type(cls):
        return "lib.data.numpy.conversion.ScalarToArray"
