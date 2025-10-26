import numpy as np
import pandas as pd 
from pydantic import BaseModel, field_validator
from typing import Literal
import operator 
from workflows import Context

OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le
}

class Filter(BaseModel):
    value: str | float | int 
    operator: Literal["==", "!=", ">", "<", ">=", "<="]

    @field_validator("operator")
    def validate_operator(cls, v, info):
        value = info.data.get("value")
        if isinstance(value, str):
            if v not in ["==", "!="]:
                raise ValueError("String values can only be compared with == or !=")
        elif isinstance(value, (int, float)):
            pass
        else:
            raise ValueError("Value must be a string, int, or float")
        return v

class FilterList(BaseModel):
    filters: list[Filter]
    

async def pearson_correlation(ctx: Context, col1_name: str, col2_name: str) -> str:
    "Useful for calculating the Pearson correlation coefficient between two column names of a dataframe"
    df = await ctx.store.get("df", default=None)

    if df is None:
        raise ValueError("df is not available in the context")

    valid_data = pd.concat([df[col1_name], df[col2_name]], axis=1).dropna()
    col1_clean, col2_clean = valid_data.iloc[:, 0], valid_data.iloc[:, 1]
    corr_matrix = np.corrcoef(col1_clean, col2_clean)
    return "Pearson correlation coefficient between {} and {} is {}".format(col1_name, col2_name, corr_matrix[0][1])

async def filter_column(ctx: Context, col_name: str, filter_values: FilterList):
    "Useful for filtering a column based on a list of filters"
    ">, <, >=, <= are for numeric columns, == and != are for string columns but can be used for numeric columns as well"
    parsed_filter_values = FilterList.model_validate(filter_values)

    df = await ctx.store.get("df", default=None)
    if df is None:
        raise ValueError("df is not available in the context")
    for filterValue in parsed_filter_values.filters:
        op_func = OPERATORS[filterValue.operator]  # get the actual operator function
        
        # Only apply comparison operators to numeric columns
        if filterValue.operator in [">", "<", ">=", "<="] and not pd.api.types.is_numeric_dtype(df[col_name]):
            raise ValueError(f"Operator '{filterValue.operator}' is only valid for numeric columns")
        
        # Apply the operator function
        df = df[op_func(df[col_name], filterValue.value)]
    await ctx.store.set("df", df)


async def aggregate(ctx: Context, col_name: str, agg_method: Literal["sum", "mean", "min", "max", "count"]) -> str:
    "Useful for aggregating a column based on an aggregation method, available methods are sum, mean, min, max, count"
    "You can only use mean, sum, min, max on numeric columns, count is for any column type"
    df = await ctx.store.get("df", default=None)
    if df is None:
        raise ValueError("df is not available in the context")

    if agg_method == "count":
        return str(df[col_name].value_counts())

    if not pd.api.types.is_numeric_dtype(df[col_name]):
        raise ValueError("You are applying an aggregation method on a non-numeric column, this is not valid")

    agg_func = None 
    if agg_method == "sum":
        agg_func = np.sum
    elif agg_method == "mean":
        agg_func = np.mean
    elif agg_method == "min":
        agg_func = np.min
    elif agg_method == "max":
        agg_func = np.max


    result = df[col_name].agg(agg_func)
    return str(result.item() if hasattr(result, "item") else result)
