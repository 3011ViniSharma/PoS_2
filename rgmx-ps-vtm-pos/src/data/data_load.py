# -*- coding: utf-8 -*-

import pandas as pd
from pyspark.sql import functions as F  # import used functions by name like in one below

from src.utils.logger import logger

function_name_to_expr_mapping = {
    "fahrenheit_to_celsius(@)": "(@ - 32) * (5/9)",
    "custom_formula(@)": "case when @ >= 0 then @ * 100 else @ * 200 end",
}


def convert_placeholder_to_function_expr(transformation_expr: str) -> str:
    """
    This function maps the transformation specified in data schema to the dataframe
    Args:
        :param transformation_expr:  str -> transformation to be applied

    Returns:
        transformation_expr -> str
    """
    for func_name, func_expr in function_name_to_expr_mapping.items():
        if func_name in transformation_expr:
            transformation_expr = transformation_expr.replace(func_name, func_expr)
    return transformation_expr


def transform(formatted_data, col_name, transformations):
    """
    This function applies the transformation specified in data schema to the dataframe
    Args:
        :param formatted_data: spark Dataframe
        :param col_name: str -> name of the column on which transformation needs to be applied
        :param transformations: transformation to be applied

    Returns:
        spark dataframe
    """
    for transformation_rule in transformations:
        transformation_rule = convert_placeholder_to_function_expr(transformation_rule)
        spark_expr = transformation_rule.replace("@", col_name)
        formatted_data = formatted_data.withColumn(col_name, F.expr(spark_expr))

    return formatted_data


def standardise_batch(formatted_data, standards):
    """
    This function applies standardisation to a batched dataframe.
    Args:
        formatted_data: Dataframe to apply standardisation on top of.
        standards: Dictionary of standards.

    Returns:
        Dataframes with standardisation rules applied to the columns.
    """
    start = pd.Timestamp.now()

    # loop through standardise input schema for each column
    for schema in standards["columns"]:

        if schema["name"] in formatted_data.columns:

            # convert date column to datetime format
            if "format" in schema["metadata"]:
                formatted_data = formatted_data.withColumn(
                    schema["name"], F.to_date(schema["name"], schema["metadata"]["format"])
                )

            # apply transformation, renaming and data type conversions
            if "standardized_name" in schema["metadata"]:
                new_name = schema["metadata"]["standardized_name"]
            else:
                new_name = schema["name"]

            if "transformations" in schema["metadata"]:
                formatted_data = formatted_data.withColumnRenamed(schema["name"], new_name)
                formatted_data = transform(
                    formatted_data, new_name, schema["metadata"]["transformations"]
                )
                formatted_data = formatted_data.withColumn(
                    new_name, F.col(new_name).cast(schema["metadata"]["type"])
                )
            else:
                formatted_data = formatted_data.withColumnRenamed(
                    schema["name"], new_name
                ).withColumn(new_name, F.col(new_name).cast(schema["metadata"]["type"]))

    duration = pd.Timestamp.now() - start
    logger.info(f"data loaded with given schema in {duration}s")

    return formatted_data
