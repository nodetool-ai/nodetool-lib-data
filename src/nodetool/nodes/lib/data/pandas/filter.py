import pandas as pd
from pydantic import Field
from nodetool.metadata.types import DataframeRef
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode


class Filter(BaseNode):
    """
    Filter dataframe based on condition.
    filter, query, condition

    Example conditions:
    age > 30
    age > 30 and salary < 50000
    name == 'John Doe'
    100 <= price <= 200
    status in ['Active', 'Pending']
    not (age < 18)

    Use cases:
    - Extract subset of data meeting specific criteria
    - Remove outliers or invalid data points
    - Focus analysis on relevant data segments
    """

    df: DataframeRef = Field(
        default=DataframeRef(), description="The DataFrame to filter."
    )
    condition: str = Field(
        default="",
        description="The filtering condition to be applied to the DataFrame, e.g. column_name > 5.",
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = await context.dataframe_to_pandas(self.df)
        res = df.query(self.condition)
        return await context.dataframe_from_pandas(res)


class FindOneRow(BaseNode):
    """
    Find the first row in a dataframe that matches a given condition.
    filter, query, condition, single row

    Example conditions:
    age > 30
    age > 30 and salary < 50000
    name == 'John Doe'
    100 <= price <= 200
    status in ['Active', 'Pending']
    not (age < 18)

    Use cases:
    - Retrieve specific record based on criteria
    - Find first occurrence of a particular condition
    - Extract single data point for further analysis
    """

    df: DataframeRef = Field(
        default=DataframeRef(), description="The DataFrame to search."
    )
    condition: str = Field(
        default="",
        description="The condition to filter the DataFrame, e.g. 'column_name == value'.",
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = await context.dataframe_to_pandas(self.df)
        result = df.query(self.condition).head(1)
        return await context.dataframe_from_pandas(result)


class SortByColumn(BaseNode):
    """
    Sort dataframe by specified column.
    sort, order, column

    Use cases:
    - Arrange data in ascending or descending order
    - Identify top or bottom values in dataset
    - Prepare data for rank-based analysis
    """

    df: DataframeRef = Field(default=DataframeRef())
    column: str = Field(default="", description="The column to sort the DataFrame by.")

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = await context.dataframe_to_pandas(self.df)
        res = df.sort_values(self.column)
        return await context.dataframe_from_pandas(res)


class RemoveDuplicates(BaseNode):
    """
    Remove duplicate rows from dataframe.
    duplicates, unique, clean

    Use cases:
    - Clean dataset by removing redundant entries
    - Ensure data integrity in analysis
    - Prepare data for unique value operations
    """

    df: DataframeRef = Field(default=DataframeRef(), description="The input DataFrame.")

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = await context.dataframe_to_pandas(self.df)
        res = df.drop_duplicates()
        return await context.dataframe_from_pandas(res)


class RemoveIncompleteRows(BaseNode):
    """
    Remove rows with NA values from dataframe.
    na, missing, clean

    Use cases:
    - Clean dataset by removing incomplete entries
    - Prepare data for analysis requiring complete cases
    - Improve data quality for modeling
    """

    df: DataframeRef = Field(default=DataframeRef(), description="The input DataFrame.")

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = await context.dataframe_to_pandas(self.df)
        res = df.dropna()
        return await context.dataframe_from_pandas(res)


class Slice(BaseNode):
    """
    Slice a dataframe by rows using start and end indices.
    slice, subset, rows

    Use cases:
    - Extract a specific range of rows from a large dataset
    - Create training and testing subsets for machine learning
    - Analyze data in smaller chunks
    """

    dataframe: DataframeRef = Field(
        default=DataframeRef(), description="The input dataframe to be sliced."
    )
    start_index: int = Field(
        default=0, description="The starting index of the slice (inclusive)."
    )
    end_index: int = Field(
        default=-1,
        description="The ending index of the slice (exclusive). Use -1 for the last row.",
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = await context.dataframe_to_pandas(self.dataframe)

        if self.end_index == -1:
            self.end_index = len(df)

        sliced_df = df.iloc[self.start_index : self.end_index]
        return await context.dataframe_from_pandas(sliced_df)
