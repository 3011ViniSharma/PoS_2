# -*- coding: utf-8 -*-
import re
from importlib import import_module
from typing import List

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame

from src.utils.config import *
from src.utils.logger import *

spark = SparkSession.builder.appName("app_name").getOrCreate()


def load_spark_data(catalog_entry: dict) -> DataFrame:
    """
    function to load a dataset given path
    Args:
        catalog_entry : dictionary for the catalog containing filepath, name and format of data to load
    Returns:
        dataframe : pyspark dataframe
    """
    return spark.read.csv(
        catalog_entry["filepath"] + catalog_entry["filename"] + "." + catalog_entry["format"],
        inferSchema="true",
        header="true",
    )


def write_data(df: DataFrame, file_path: str):
    """
    function to write a dataset to a given path
    Args:
        df: DataFrame
        file_path : output file path of the data file
    Returns:
        None
    """
    df.write.format("csv").mode("overwrite").options(header="true", sep=",").save(path=file_path)

    return None


def load_csv_data(file_path: str) -> DataFrame:
    """
    function to load a dataset given path
    Args:
        file_path : input path of the data file
    Returns:
        dataframe : pyspark dataframe
    """
    return pd.read_csv(file_path, header=0, infer_datetime_format=True)


def write_data_to_csv(df: DataFrame, file_path: str):

    """
    function to write a dataset to a given path
    Args:
        file_path : output file path of the data file
    Returns:
        None
    """
    df.to_csv(file_path, index=False)

    return None


def convert_sparkdf_to_pandasdf(sparkdf: DataFrame, catalog) -> pd.DataFrame:
    """
    Helper function to update report dictionary
    Args:
        sparkdf: spark dataframe
    Returns:
        pandas dataframe
    """
    write_data_parquet(sparkdf, catalog['temp_file']['filepath'] + catalog['temp_file']['filename'] + '.' + catalog['temp_file']['format'])
    return load_last_parquet_file(catalog['temp_file'])

def load_last_parquet_file(catalog_entry):
    path = catalog_entry['filepath'] + catalog_entry['filename'] + '.' + catalog_entry['format']
    path = path.replace('dbfs:/', '/dbfs/')
    files = os.listdir(path)
    files_to_load = [file_ for file_ in files if file_.startswith("part")]
    dfrs = []
    for file in files_to_load:
        dfrs.append(pd.read_parquet(path + '/' + file))
    return pd.concat(dfrs)

def write_data_parquet(df: DataFrame, file_path: str):
    """
    function to write a dataset to a given path
    Args:
        df: DataFrame
        file_path : output file path of the data file
    Returns:
        None
    """
    df.write.format("parquet").mode("overwrite").save(path=file_path)

    return None

def create_concatenated_column(
    input_df: DataFrame, column_list: List[str], name: str, seperator: str
) -> DataFrame:
    """
    This function creates a concatenated column from a given list of columns
    Args:
        input_df: spark dataframe
        column_list : list of colums to concatenate
        name : name of the new column
        seperator : separator value
    Returns:
        out_df : spark dataframe with a concatenated column
    """
    start = pd.Timestamp.now()

    input_df = input_df.withColumn(name, F.concat_ws(seperator, *column_list))

    duration = pd.Timestamp.now() - start
    logger.info(f"concatenated {column_list} as {name} in {duration}s")

    return input_df


def log_volume(vol_col: str) -> str:
    """
    Helper function to append "log_" to the volume column name
    Args:
        vol_col: string column name of column containing volume info
    Returns:
        string column name with "log_" appended
    """
    return f"log_{vol_col}"


def pantry_loading(num_pl_features):
    """
    Helper function to create colnames for pantry loading features
    Args:
        num_pl_features: int
            Number of pantry loading features
    Returns:
        pl_features_names: str
            List of pantry loading features names
    """
    pl_features_names = [f"pl{i}" for i in range(1, num_pl_features + 1)]
    return pl_features_names


def all_pantry_loading_list(num_pl_features):
    """
    Helper function to create a nested list of colnames for pantry loading features
    Args:
        num_pl_features: int
            Number of pantry loading features
    Returns:
        possible_pl_features_names: str
            List of list of possible pantry loading features names
    """
    possible_pl_features_names = []
    for i in range(3, num_pl_features + 1):
        possible_pl_features_names.append(pantry_loading(i))
    return possible_pl_features_names


def flatten_list(input_list):
    """
    Helper function to flatten a list
    Args:
        input_list: a nested list
    Returns:
    flat_list: a flattened list
    """
    flat_list = []
    # Iterate through the outer list
    for element in input_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


def process_pl_func(variable, pl_list):
    """
    Helper function to process a function inside the list and flatten it from a nested loop
    Args:
        variable: a variable to be loaded
        pl_list: list of pl variables
    Returns:
        variable: loaded variable
    """
    intermediate = variable.copy()
    intermediate[-1] = load_func(variable[-1])(len(pl_list))
    variable = flatten_list(intermediate)
    return variable


def load_func(dotpath: str):
    """
    Helper function to load a function in module.
    Function is right-most segment
    Args:
        dotpath: function path
    Returns:
        callable function
    """
    module_, func = dotpath.rsplit(".", maxsplit=1)
    m = import_module(module_)
    return getattr(m, func)


def get_effect_colnames(fixed_effect: str) -> List[str]:
    """
    Function to get effect column names from fixed effect part of modeling equation
    Args:
        fixed_effect: string from modeling equation
    Returns:
        list: list of columns for fixed effect
    """
    result = fixed_effect.split("+")
    result = [r for r in result if "&" in r]
    result = [re.sub("^(.*?)&", "", r) for r in result]
    result = [a.strip() for r in result for a in r.split("&")]
    return list(set(result))


def rename_cols_using_dict(df: pd.DataFrame, col_dict: dict) -> pd.DataFrame:
    """
    helper function to rename columns and assign column values using a dictionary
    Args:
        df : pandas dataframe
        col_dict : dictionary with column names and value mapping
    Returns:
        df : pandas dataframe with named columns
    """
    for key, value in col_dict.items():
        if bool(value):
            df = df.rename(columns={value: key.upper()})
        else:
            df[key.upper()] = value
    return df
