import pytest
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock
from io import BytesIO
from datetime import datetime
from PIL import Image
import io

from nodetool.nodes.lib.data.numpy import (
    AddArray,
    SubtractArray,
    MultiplyArray,
    DivideArray,
    ModulusArray,
    SineArray,
    CosineArray,
    PowerArray,
    SqrtArray,
    SaveArray,
    ConvertToArray,
    ConvertToImage,
    ConvertToAudio,
    Stack,
    MatMul,
    TransposeArray,
    MaxArray,
    MinArray,
    MeanArray,
    SumArray,
    ArgMaxArray,
    ArgMinArray,
    AbsArray,
    ArrayToScalar,
    ScalarToArray,
    ListToArray,
    PlotArray,
    ArrayToList,
    ExpArray,
    LogArray,
    SliceArray,
    IndexArray,
    Reshape1D,
    Reshape2D,
    Reshape3D,
    Reshape4D,
    SplitArray,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import NPArray, ImageRef, AudioRef, FolderRef


@pytest.fixture
def processing_context():
    # Using test values for required parameters
    return ProcessingContext(user_id="test-user", auth_token="test-token")


@pytest.fixture
def sample_array():
    return NPArray.from_numpy(np.array([1, 2, 3, 4]))


@pytest.fixture
def sample_2d_array():
    return NPArray.from_numpy(np.array([[1, 2], [3, 4]]))


class TestBinaryOperations:
    @pytest.mark.asyncio
    async def test_add_array(self, processing_context, sample_array):
        # Setup
        node = AddArray(a=sample_array, b=2)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([3, 4, 5, 6]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_subtract_array(self, processing_context, sample_array):
        # Setup
        node = SubtractArray(a=sample_array, b=1)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([0, 1, 2, 3]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_multiply_array(self, processing_context, sample_array):
        # Setup
        node = MultiplyArray(a=sample_array, b=2)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([2, 4, 6, 8]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_divide_array(self, processing_context, sample_array):
        # Setup
        node = DivideArray(a=sample_array, b=2)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([0.5, 1, 1.5, 2]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_modulus_array(self, processing_context, sample_array):
        # Setup
        node = ModulusArray(a=sample_array, b=2)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([1, 0, 1, 0]), result.to_numpy())


class TestTrigonometricOperations:
    @pytest.mark.asyncio
    async def test_sine_array(self, processing_context):
        # Setup
        node = SineArray(angle_rad=np.pi / 2)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, float)
        assert pytest.approx(1.0, 0.0001) == result

    @pytest.mark.asyncio
    async def test_cosine_array(self, processing_context):
        # Setup
        node = CosineArray(angle_rad=0)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, float)
        assert pytest.approx(1.0, 0.0001) == result


class TestMathOperations:
    @pytest.mark.asyncio
    async def test_power_array(self, processing_context, sample_array):
        # Setup
        node = PowerArray(base=sample_array, exponent=2)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([1, 4, 9, 16]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_sqrt_array(self, processing_context):
        # Setup
        node = SqrtArray(values=NPArray.from_numpy(np.array([4, 9, 16, 25])))

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([2, 3, 4, 5]), result.to_numpy())


class TestConversionOperations:
    @pytest.mark.asyncio
    async def test_convert_to_array(self, processing_context):
        # Setup
        # Create a real PIL image
        test_image = Image.new("RGB", (10, 10), color=(255, 255, 255))
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()

        # Create ImageRef with actual image data
        image_ref = ImageRef(data=img_bytes)
        node = ConvertToArray(image=image_ref)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        assert result.to_numpy().shape == (10, 10, 3)
        assert np.all(result.to_numpy() == 1.0)  # Normalized to [0,1]

    @pytest.mark.asyncio
    async def test_convert_to_image(self, processing_context):
        # Setup
        array_data = NPArray.from_numpy(np.ones((10, 10, 3)))
        node = ConvertToImage(values=array_data)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, ImageRef)
        assert result.data is not None  # Should have binary data

    @pytest.mark.asyncio
    async def test_convert_to_audio(self, processing_context):
        # Setup
        array_data = NPArray.from_numpy(np.ones(1000))
        node = ConvertToAudio(values=array_data, sample_rate=44100)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, AudioRef)
        assert result.data is not None  # Should have binary data


class TestArrayManipulation:
    @pytest.mark.asyncio
    async def test_stack(self, processing_context):
        # Setup
        arrays = [
            NPArray.from_numpy(np.array([1, 2, 3])),
            NPArray.from_numpy(np.array([4, 5, 6])),
        ]
        node = Stack(arrays=arrays, axis=0)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        expected = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(expected, result.to_numpy())

    @pytest.mark.asyncio
    async def test_matmul(self, processing_context):
        # Setup
        a = NPArray.from_numpy(np.array([[1, 2], [3, 4]]))
        b = NPArray.from_numpy(np.array([[5, 6], [7, 8]]))
        node = MatMul(a=a, b=b)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(expected, result.to_numpy())

    @pytest.mark.asyncio
    async def test_transpose_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
        node = TransposeArray(values=array)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(expected, result.to_numpy())


class TestReductionOperations:
    @pytest.mark.asyncio
    async def test_max_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
        node = MaxArray(values=array, axis=1)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([3, 6]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_min_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
        node = MinArray(values=array, axis=1)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([1, 4]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_mean_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
        node = MeanArray(values=array, axis=1)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([2, 5]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_sum_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
        node = SumArray(values=array, axis=1)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([6, 15]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_argmax_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([[1, 2, 3], [6, 5, 4]]))
        node = ArgMaxArray(values=array, axis=1)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([2, 0]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_argmin_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([[1, 2, 3], [6, 5, 4]]))
        node = ArgMinArray(values=array, axis=1)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([0, 2]), result.to_numpy())


class TestElementWiseOperations:
    @pytest.mark.asyncio
    async def test_abs_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([-1, -2, 3, -4]))
        node = AbsArray(values=array)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([1, 2, 3, 4]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_exp_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([0, 1, 2]))
        node = ExpArray(values=array)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        expected = np.array([1, np.e, np.e**2])
        np.testing.assert_allclose(expected, result.to_numpy())

    @pytest.mark.asyncio
    async def test_log_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([1, np.e, np.e**2]))
        node = LogArray(values=array)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        expected = np.array([0, 1, 2])
        np.testing.assert_allclose(expected, result.to_numpy())


class TestTypeConversions:
    @pytest.mark.asyncio
    async def test_array_to_scalar(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([42]))
        node = ArrayToScalar(values=array)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, int)
        assert result == 42

    @pytest.mark.asyncio
    async def test_scalar_to_array(self, processing_context):
        # Setup
        node = ScalarToArray(value=42)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([42]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_list_to_array(self, processing_context):
        # Setup
        node = ListToArray(values=[1, 2, 3, 4])

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([1, 2, 3, 4]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_array_to_list(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([1, 2, 3, 4]))
        node = ArrayToList(values=array)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, list)
        assert result == [1, 2, 3, 4]


class TestVisualization:
    @pytest.mark.asyncio
    async def test_plot_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([1, 2, 3, 4]))
        node = PlotArray(values=array, plot_type=PlotArray.PlotType.LINE)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, ImageRef)
        assert result.data is not None  # Should have binary data
        # Optional: could verify it's a valid image by loading with PIL
        img = Image.open(io.BytesIO(result.data))
        assert img.format in ("PNG", "JPEG")


class TestSlicingAndIndexing:
    @pytest.mark.asyncio
    async def test_slice_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        node = SliceArray(values=array, start=2, stop=8, step=2, axis=0)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([2, 4, 6]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_index_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([10, 20, 30, 40, 50]))

        # Need to patch the map function since it's used in the implementation
        with patch("nodetool.nodes.lib.data.numpy.map", return_value=[1, 3]):
            node = IndexArray(values=array, indices="1,3", axis=0)

            # Execute
            result = await node.process(processing_context)

            # Assert
            assert isinstance(result, NPArray)
            np.testing.assert_array_equal(np.array([20, 40]), result.to_numpy())


class TestReshaping:
    @pytest.mark.asyncio
    async def test_reshape_1d(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([[1, 2], [3, 4]]))
        node = Reshape1D(values=array, num_elements=4)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        np.testing.assert_array_equal(np.array([1, 2, 3, 4]), result.to_numpy())

    @pytest.mark.asyncio
    async def test_reshape_2d(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([1, 2, 3, 4, 5, 6]))
        node = Reshape2D(values=array, num_rows=2, num_cols=3)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        expected = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(expected, result.to_numpy())

    @pytest.mark.asyncio
    async def test_reshape_3d(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array(range(24)))
        node = Reshape3D(values=array, num_rows=2, num_cols=3, num_depths=4)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        assert result.to_numpy().shape == (2, 3, 4)

    @pytest.mark.asyncio
    async def test_reshape_4d(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array(range(24)))
        node = Reshape4D(
            values=array, num_rows=2, num_cols=3, num_depths=2, num_channels=2
        )

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, NPArray)
        assert result.to_numpy().shape == (2, 3, 2, 2)


class TestSplitting:
    @pytest.mark.asyncio
    async def test_split_array(self, processing_context):
        # Setup
        array = NPArray.from_numpy(np.array([1, 2, 3, 4, 5, 6]))
        node = SplitArray(values=array, num_splits=3, axis=0)

        # Execute
        result = await node.process(processing_context)

        # Assert
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(r, NPArray) for r in result)
        np.testing.assert_array_equal(np.array([1, 2]), result[0].to_numpy())
        np.testing.assert_array_equal(np.array([3, 4]), result[1].to_numpy())
        np.testing.assert_array_equal(np.array([5, 6]), result[2].to_numpy())
