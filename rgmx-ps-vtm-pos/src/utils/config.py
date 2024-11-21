# -*- coding: utf-8 -*-
import os

import bios
from pyspark.sql.types import *


# get all file names that end with yml in a folder in python
def load_params(base: str):
    """
    Function to load parameters file in yml format into the environment
    Args:
        base : base location to project where the notebooks , src and the conf folders are present
    Returns:
        final_params: Dictionary of all parameter files present within the conf/parameters folder ending with .yml
    """
    final_params = {}
    parameters_list = os.listdir(base + "conf/parameters/")
    parameters_list = [i for i in parameters_list if i.endswith("yml")]
    for params in parameters_list:

        my_dict = bios.read(base + "conf/parameters/" + params)

        final_params.update(my_dict)

    return final_params


def load_catalog(base: str, dir_path_spark: str, dir_path: str):
    """
    Function to load catalog file in yml format into the environment
    Args:
        base : base location to project where the notebooks , src and the conf folders are present
        dir_path_spark: spark formatted path of the directory where inputs and outputs will be stored
        dir_path: path of the directory where inputs and outputs will be stored
    Returns:
        final_catalog: Dictionary of all catalog entries present within the conf/catalog folder ending with _catalog.yml
    """
    final_catalog = {}
    catalog_paths = os.listdir(base + "conf/catalog/")
    catalog_paths = [i for i in catalog_paths if i.endswith("_catalog.yml")]
    for path in catalog_paths:

        my_dict = bios.read(base + "conf/catalog/" + path)

        final_catalog.update(my_dict)

    # adding a common directory path to all filepaths
    for key in final_catalog.keys():
        if "type" in final_catalog[key].keys() and final_catalog[key]["type"] == "spark":
            final_catalog[key]["filepath"] = dir_path_spark + final_catalog[key]["filepath"]
        else:
            final_catalog[key]["filepath"] = dir_path + final_catalog[key]["filepath"]

    return final_catalog
