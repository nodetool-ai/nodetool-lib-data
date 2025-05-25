import numpy as np
from pydub import AudioSegment
from nodetool.nodes.lib.numpy.utils import numpy_to_audio_segment


def test_numpy_to_audio_segment_basic():
    arr = np.array([0.0, 0.5, -0.5, 1.0])
    sample_rate = 22050
    segment = numpy_to_audio_segment(arr, sample_rate)
    assert isinstance(segment, AudioSegment)
    assert segment.frame_rate == sample_rate
    expected_duration_ms = int(len(arr) / sample_rate * 1000)
    assert len(segment) == expected_duration_ms
