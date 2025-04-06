import pandas as pd
from pydantic import Field
from nodetool.metadata.types import DataframeRef
from nodetool.metadata.types import NPArray
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode
from typing import Any


class SelectColumn(BaseNode):
    """
    Select specific columns from dataframe.
    dataframe, columns, filter

    Use cases:
    - Extract relevant features for analysis
    - Reduce dataframe size by removing unnecessary columns
    - Prepare data for specific visualizations or models
    """

    dataframe: DataframeRef = Field(
        default=DataframeRef(),
        description="a dataframe from which columns are to be selected",
    )
    columns: str = Field("", description="comma separated list of column names")

    async def process(self, context: ProcessingContext) -> DataframeRef:
        columns = self.columns.split(",")
        df = await context.dataframe_to_pandas(self.dataframe)
        return await context.dataframe_from_pandas(df[columns])  # type: ignore


class ExtractColumn(BaseNode):
    """
    Convert dataframe column to list.
    dataframe, column, list

    Use cases:
    - Extract data for use in other processing steps
    - Prepare column data for plotting or analysis
    - Convert categorical data to list for encoding
    """

    dataframe: DataframeRef = Field(
        default=DataframeRef(), description="The input dataframe."
    )
    column_name: str = Field(
        default="", description="The name of the column to be converted to a list."
    )

    async def process(self, context: ProcessingContext) -> list[Any]:
        df = await context.dataframe_to_pandas(self.dataframe)
        return df[self.column_name].tolist()


class FormatAsText(BaseNode):
    """
    Convert dataframe rows to formatted strings.
    dataframe, string, format

    Use cases:
    - Generate text summaries from row data
    - Prepare data for natural language processing
    - Create custom string representations of rows
    """

    dataframe: DataframeRef = Field(
        default=DataframeRef(), description="The input dataframe."
    )
    template: str = Field(
        default="",
        description="The template for the string representation. Each column can be referenced by {column_name}.",
    )

    async def process(self, context: ProcessingContext) -> list[str]:
        df = await context.dataframe_to_pandas(self.dataframe)
        return [self.template.format(**row) for _, row in df.iterrows()]


class AddColumn(BaseNode):
    """
    Add list of values as new column to dataframe.
    dataframe, column, list

    Use cases:
    - Incorporate external data into existing dataframe
    - Add calculated results as new column
    - Augment dataframe with additional features
    """

    dataframe: DataframeRef = Field(
        default=DataframeRef(),
        description="Dataframe object to add a new column to.",
    )
    column_name: str = Field(
        default="",
        description="The name of the new column to be added to the dataframe.",
    )
    values: list[Any] = Field(
        default=[],
        description="A list of any type of elements which will be the new column's values.",
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df = await context.dataframe_to_pandas(self.dataframe)
        df[self.column_name] = self.values
        return await context.dataframe_from_pandas(df)


class MergeSideBySide(BaseNode):
    """
    Merge two dataframes along columns.
    merge, concat, columns

    Use cases:
    - Combine data from multiple sources
    - Add new features to existing dataframe
    - Merge time series data from different periods
    """

    dataframe_a: DataframeRef = Field(
        default=DataframeRef(), description="First DataFrame to be merged."
    )
    dataframe_b: DataframeRef = Field(
        default=DataframeRef(), description="Second DataFrame to be merged."
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df_a = await context.dataframe_to_pandas(self.dataframe_a)
        df_b = await context.dataframe_to_pandas(self.dataframe_b)
        df = pd.concat([df_a, df_b], axis=1)
        return await context.dataframe_from_pandas(df)


class CombineVertically(BaseNode):
    """
    Append two dataframes along rows.
    append, concat, rows

    Use cases:
    - Combine data from multiple time periods
    - Merge datasets with same structure
    - Aggregate data from different sources
    """

    dataframe_a: DataframeRef = Field(
        default=DataframeRef(), description="First DataFrame to be appended."
    )
    dataframe_b: DataframeRef = Field(
        default=DataframeRef(), description="Second DataFrame to be appended."
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df_a = await context.dataframe_to_pandas(self.dataframe_a)
        df_b = await context.dataframe_to_pandas(self.dataframe_b)

        # Handle empty dataframes
        if df_a.empty:
            return await context.dataframe_from_pandas(df_b)
        if df_b.empty:
            return await context.dataframe_from_pandas(df_a)

        # Check column compatibility only if both dataframes are non-empty
        if not df_a.columns.equals(df_b.columns):
            raise ValueError(
                f"Columns in dataframe A ({df_a.columns}) do not match columns in dataframe B ({df_b.columns})"
            )

        df = pd.concat([df_a, df_b], axis=0)
        return await context.dataframe_from_pandas(df)


class Join(BaseNode):
    """
    Join two dataframes on specified column.
    join, merge, column

    Use cases:
    - Combine data from related tables
    - Enrich dataset with additional information
    - Link data based on common identifiers
    """

    dataframe_a: DataframeRef = Field(
        default=DataframeRef(), description="First DataFrame to be merged."
    )
    dataframe_b: DataframeRef = Field(
        default=DataframeRef(), description="Second DataFrame to be merged."
    )
    join_on: str = Field(
        default="",
        description="The column name on which to join the two dataframes.",
    )

    async def process(self, context: ProcessingContext) -> DataframeRef:
        df_a = await context.dataframe_to_pandas(self.dataframe_a)
        df_b = await context.dataframe_to_pandas(self.dataframe_b)
        if not df_a.columns.equals(df_b.columns):
            raise ValueError(
                f"Columns in dataframe A ({df_a.columns}) do not match columns in dataframe B ({df_b.columns})"
            )
        df = pd.merge(df_a, df_b, on=self.join_on)
        return await context.dataframe_from_pandas(df)


class ConvertToTensor(BaseNode):
    """
    Convert dataframe to tensor.
    dataframe, tensor, convert

    Use cases:
    - Prepare data for deep learning models
    - Enable tensor operations on dataframe data
    - Convert tabular data to multidimensional format
    """

    dataframe: DataframeRef = Field(
        default=DataframeRef(), description="The input dataframe."
    )

    async def process(self, context: ProcessingContext) -> NPArray:
        df = await context.dataframe_to_pandas(self.dataframe)
        return NPArray.from_numpy(df.to_numpy())


class MapTemplate(BaseNode):
    """
    Maps a template string over dataframe rows using Jinja2 templating.
    dataframe, template, format, string

    Use cases:
    - Format each row into a custom string representation
    - Generate text summaries from structured data
    - Create formatted output from dataframe records

    Example:
    Template: "Name: {{ name }}, Age: {{ age|default('unknown') }}"
    Row: {"name": "Alice", "age": 30}
    Output: "Name: Alice, Age: 30"

    Available filters:
    - truncate(length): Truncates text to given length
    - upper: Converts text to uppercase
    - lower: Converts text to lowercase
    - title: Converts text to title case
    - trim: Removes whitespace from start/end
    - replace(old, new): Replaces substring
    - default(value): Sets default if value is undefined
    - first: Gets first character/item
    - last: Gets last character/item
    - length: Gets length of string/list
    - sort: Sorts list
    - join(delimiter): Joins list with delimiter
    """

    dataframe: DataframeRef = Field(
        default=DataframeRef(), description="The input dataframe."
    )
    template: str = Field(
        default="",
        description="""Template string with Jinja2 placeholders matching column names 
        (e.g., {{ column_name }}). Supports filters like {{ value|upper }}, {{ value|truncate(20) }}.""",
    )

    async def process(self, context: ProcessingContext) -> list[str]:
        from jinja2 import Environment, BaseLoader

        df = await context.dataframe_to_pandas(self.dataframe)

        # Create Jinja2 environment
        env = Environment(loader=BaseLoader())
        template = env.from_string(self.template)

        results = []
        for _, row in df.iterrows():
            try:
                results.append(template.render(**row.to_dict()))
            except Exception:
                # Skip rows that don't match the template
                continue

        return results
