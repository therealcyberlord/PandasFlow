import pandas as pd 
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_datetime64_any_dtype
import json 

def csv_to_metadata_json(df: pd.DataFrame, sample_rows=3):
    metadata = {
        "dataset_shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": {},
        "sample_rows": df.sample(min(sample_rows, len(df)), random_state=42).to_dict(orient="records")
    }

    for col in df.columns:
        col_data = df[col]
        col_info = {
            "dtype": str(col_data.dtype),
            "missing": int(col_data.isna().sum()),
            "unique": int(col_data.nunique())
        }

        # Handle numeric columns
        if is_numeric_dtype(col_data):
            desc = col_data.describe().to_dict()
            col_info["summary"] = {
                "min": desc.get("min"),
                "max": desc.get("max"),
                "mean": desc.get("mean"),
                "std": desc.get("std")
            }

        # Handle categorical/object columns
        elif is_object_dtype(col_data):
            top_values = col_data.value_counts().head(5).to_dict()
            col_info["summary"] = {
                "top_values": top_values
            }

        # Handle datetime columns
        elif is_datetime64_any_dtype(col_data):
            col_info["summary"] = {
                "min": str(col_data.min()),
                "max": str(col_data.max())
            }
        else:
            col_info["summary"] = {"note": "Unsupported dtype for detailed summary"}

        metadata["columns"][col] = col_info

    return json.dumps(metadata, indent=2, default=str)