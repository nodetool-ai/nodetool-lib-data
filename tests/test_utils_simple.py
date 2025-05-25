import numpy as np
import pytest
from nodetool.nodes.lib.numpy.utils import pad_arrays, convert_output
from nodetool.metadata.types import NPArray
from nodetool.workflows.processing_context import ProcessingContext

@pytest.fixture
def context():
    return ProcessingContext(user_id="test", auth_token="token")


def test_pad_arrays_extends_shorter():
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    pa, pb = pad_arrays(a, b)
    assert pa.shape == pb.shape
    assert np.array_equal(pa, np.array([1, 2, 3]))
    assert np.array_equal(pb, np.array([4, 5, 0]))


@pytest.mark.asyncio
async def test_convert_output_scalar(context):
    result = await convert_output(context, np.array([42]))
    assert result == 42


@pytest.mark.asyncio
async def test_convert_output_array(context):
    arr = np.array([1, 2, 3])
    result = await convert_output(context, arr)
    assert isinstance(result, NPArray)
    np.testing.assert_array_equal(result.to_numpy(), arr)
