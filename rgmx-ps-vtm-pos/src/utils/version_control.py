# -*- coding: utf-8 -*-
import glob
import json
import os
import time as time

import pandas as pd


def create_file_path(file_path):
    """
    Helper function to check if directory exists and if not, create the directory
    Args:
        file_path: file path
    Returns:
        None
    """
    if os.path.isdir(file_path) == False:
        os.mkdir(file_path)
    return


def find_latest(path):
    """
    Helper function to find the latest written file in a directory
    Args:
        path: the folder to search for
    Returns:
        the latest written file
    """
    list_of_files = glob.glob("{0}/*".format(path))
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file


def write_obj(obj, file_path, file_name, format):
    """
    Helper function to write dataframes to files
    Args:
        obj: the obj to write (must be dict, spark or pandas)
        file_path: the folder name that will contain the data
        file_name: the file name of the data
        format: format of the data (must be txt, csv or parquet)
    Returns:
        None
    """
    create_file_path(file_path="{0}{1}".format(file_path, file_name))
    time_version = time.strftime("%Y%m%d-%H%M%S")
    if isinstance(obj, dict):
        if format == "txt":
            with open(
                "{0}{1}/{1}_{2}.{3}".format(file_path, file_name, time_version, format), "w"
            ) as file:
                file.write(json.dumps(obj))
        else:
            raise Exception(
                "File format {0} is not defined to be written using write_obj() for dictionary objects.".format(
                    format
                )
            )
    elif isinstance(obj, pd.DataFrame):
        # pandas
        if format == "csv":
            obj.to_csv(
                path_or_buf="{0}{1}/{1}_{2}.{3}".format(file_path, file_name, time_version, format),
                index=False,
            )
        else:
            raise Exception(
                "File format {0} is not defined to be written using write_obj() for pandas dataframes.".format(
                    format
                )
            )
    else:
        # spark
        if format == "csv":
            obj.write.format("csv").mode("overwrite").options(header="true", sep=",").save(
                path="{0}{1}/{1}_{2}.{3}".format(file_path, file_name, time_version, format)
            )
        elif format == "parquet":
            obj.write.format("parquet").mode("overwrite").save(
                path="{0}{1}/{1}_{2}.{3}".format(file_path, file_name, time_version, format)
            )
        else:
            raise Exception(
                "File format {0} is not defined to be written using write_obj() for spark dataframes.".format(
                    format
                )
            )


def read_obj(file_path, file_name, format):
    """
    Helper function to read a csv to pandas dataframe or txt file to dictionary
    Args:
        format:
        file_path: the folder name that will contain the data
        file_name: the file name of the data
        format: the format of the file (csv or txt supported only)
    Returns:
        contents of the file as a pandas dataframe or dictionary
    """
    latest_version = find_latest(
        path="{0}{1}".format(
            file_path,
            file_name,
        )
    )
    if format == "csv":
        return pd.read_csv(latest_version, header=0, infer_datetime_format=True)
    elif format == "txt":
        with open(latest_version, "r") as file:
            return json.loads(file.read())
    else:
        raise Exception("File format {0} is not supported by read_obj().".format(format))
