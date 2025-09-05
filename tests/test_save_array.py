import numpy as np
import pytest
from types import SimpleNamespace

from nodetool.nodes.lib.numpy.io import SaveArray
from nodetool.metadata.types import NPArray, FolderRef
from nodetool.workflows.processing_context import ProcessingContext


@pytest.fixture
def context():
    return ProcessingContext(user_id="t", auth_token="token")


@pytest.mark.asyncio
async def test_save_array_process(context):
    arr = NPArray.from_numpy(np.array([1, 2, 3]))
    node = SaveArray(values=arr, folder=FolderRef(asset_id="folder"), name="test.npy")

    async def fake_create_asset(name, content_type, content, parent_id=None):
        return SimpleNamespace(id="asset", get_url="http://example.com/asset")

    context.create_asset = fake_create_asset

    result = await node.process(context)

    assert isinstance(result, NPArray)
    # NPArray does not expose asset metadata; ensure data round-trips correctly
    np.testing.assert_array_equal(result.to_numpy(), arr.to_numpy())
