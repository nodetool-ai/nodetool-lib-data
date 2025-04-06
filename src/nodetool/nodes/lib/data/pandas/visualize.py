import io
from enum import Enum
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import Field
from nodetool.metadata.types import DataframeRef
from nodetool.metadata.types import ImageRef
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.base_node import BaseNode


class Chart(BaseNode):
    """
    Create line, bar, or scatter plot from dataframe.
    plot, visualization, dataframe

    Use cases:
    - Visualize trends in time series data
    - Compare values across categories
    - Explore relationships between variables
    """

    class PlotType(str, Enum):
        LINE = "line"
        BAR = "bar"
        SCATTER = "scatter"

    dataframe: DataframeRef = Field(
        default=DataframeRef(), description="The input dataframe."
    )
    x_column: str = Field(
        default="",
        description="The name of the x column to be used in the plot.",
    )
    y_column: str = Field(
        default="",
        description="The name of the y column to be used in the plot.",
    )
    plot_type: PlotType = Field(
        default=PlotType.LINE,
        description="The type of plot to be created. Can be 'line', 'bar', or 'scatter'.",
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        df = await context.dataframe_to_pandas(self.dataframe)
        if self.x_column not in df.columns:  # type: ignore
            raise ValueError(f"Invalid x_column: {self.x_column}")
        if self.y_column not in df.columns:  # type: ignore
            raise ValueError(f"Invalid y_column: {self.y_column}")
        if self.plot_type == self.PlotType.LINE:
            plot = sns.lineplot(x=self.x_column, y=self.y_column, data=df)
        elif self.plot_type == self.PlotType.BAR:
            plot = sns.barplot(x=self.x_column, y=self.y_column, data=df)
        elif self.plot_type == self.PlotType.SCATTER:
            plot = sns.scatterplot(x=self.x_column, y=self.y_column, data=df)
        else:
            raise ValueError(f"Invalid plot type: {self.plot_type}")
        fig = plot.get_figure()
        if fig is None:
            raise ValueError("Invalid plot")
        img_bytes = io.BytesIO()
        fig.savefig(img_bytes, format="png")
        plt.close(fig)
        return await context.image_from_bytes(img_bytes.getvalue())


class Histogram(BaseNode):
    """
    Plot histogram of dataframe column.
    histogram, plot, distribution

    Use cases:
    - Visualize distribution of continuous data
    - Identify outliers and data patterns
    - Compare data distributions across categories
    """

    dataframe: DataframeRef = Field(
        default=DataframeRef(), description="The input dataframe."
    )
    column: str = Field(title="Column", default="", description="The column to plot.")

    async def process(self, context: ProcessingContext) -> ImageRef:
        df = await context.dataframe_to_pandas(self.dataframe)
        if self.column not in df.columns:
            raise ValueError(f"Invalid column: {self.column}")
        (fig, ax) = plt.subplots()
        sns.set_theme(style="darkgrid")
        sns.histplot(df[self.column], ax=ax)  # type: ignore
        img_bytes = io.BytesIO()
        fig.savefig(img_bytes, format="png")
        plt.close(fig)
        return await context.image_from_bytes(img_bytes.getvalue())


class Heatmap(BaseNode):
    """
    Create heatmap visualization of dataframe.
    heatmap, plot, correlation

    Use cases:
    - Visualize correlation between variables
    - Identify patterns in multi-dimensional data
    - Display intensity of values across categories
    """

    dataframe: DataframeRef = Field(
        default=DataframeRef(), description="The input dataframe."
    )

    async def process(self, context: ProcessingContext) -> ImageRef:
        df = await context.dataframe_to_pandas(self.dataframe)
        sns.set_theme(style="darkgrid")
        (fig, ax) = plt.subplots()
        sns.heatmap(df, ax=ax)
        img_bytes = io.BytesIO()
        fig.savefig(img_bytes, format="png")
        plt.close(fig)
        return await context.image_from_bytes(img_bytes.getvalue())
