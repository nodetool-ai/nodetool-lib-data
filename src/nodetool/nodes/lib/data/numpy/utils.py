from typing import Tuple
import numpy as np
from pydantic import Field
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode
from nodetool.metadata.types import to_numpy
from nodetool.metadata.types import NPArray
from pydub import AudioSegment


def numpy_to_audio_segment(arr: np.ndarray, sample_rate=44100) -> AudioSegment:
    """
    Convert a numpy array to an audio segment.

    Args:
        arr (np.ndarray): The numpy array to convert.
        sample_rate (int): The sample rate of the audio segment.

    Returns:
        AudioSegment: The audio segment.
    """
    # Convert the float array to int16 format, which is used by WAV files.
    arr_int16 = np.int16(arr * 32767.0).tobytes()

    # Create a pydub AudioSegment from raw data.
    return AudioSegment(arr_int16, sample_width=2, frame_rate=sample_rate, channels=1)


def pad_arrays(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    If one of the arguments is a scalar, both arguments are returned as is.
    Pads the smaller array with zeros so that both arrays are the same size.
    This is useful for operations like addition and subtraction.
    """
    if a.size == 1 or b.size == 1:
        return (a, b)
    if len(a) != len(b):
        if len(a) > len(b):
            b = np.pad(b, (0, (len(a) - len(b))), "constant")
        else:
            a = np.pad(a, (0, (len(b) - len(a))), "constant")
    return (a, b)


async def convert_output(
    context: ProcessingContext, output: np.ndarray
) -> float | int | NPArray:
    if output.size == 1:
        return output.item()
    else:
        return NPArray.from_numpy(output)


class BinaryOperation(BaseNode):
    _layout = "small"
    a: int | float | NPArray = Field(title="A", default=0.0)
    b: int | float | NPArray = Field(title="B", default=0.0)

    async def process(self, context: ProcessingContext) -> int | float | NPArray:
        a = to_numpy(self.a)
        b = to_numpy(self.b)
        if a.size > 1 and b.size > 1:
            (a, b) = pad_arrays(a, b)
        res = self.operation(a, b)
        return await convert_output(context, res)

    def operation(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
