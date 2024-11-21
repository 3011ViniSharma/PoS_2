# -*- coding: utf-8 -*-
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import numpy as np


# config.yaml structure
# rgmx_ouput_config:
#     RGM_PS_VTM: saving_path/RGM_PS_VTM.csv
#     RGM_SIMULATION_CONFIG: saving_path/RGM_SIMULATION_CONFIG.csv
#     RGM_PS_STRUCTURE: saving_path/RGM_PS_STRUCTURE.csv
#     SOURCE_FEEDS_MANIFEST: saving_path/SOURCE_FEEDS_MANIFEST.csv

def rgmx_outputs_generator(
    sku_matrix: pd.DataFrame,
    point_of_sales_df: pd.DataFrame,
    rmgx_parameters: Dict,
    outputs_config: Dict[str, Dict[str, Any]],
) -> Dict[str, pd.DataFrame]:
    """Returns a dictionary with RGMx prepared ouputs.

    Args:
        sku_matrix (pd.DataFrame): SKU matrix
        point_of_sales_df (pd.DataFrame): dataframe with POS data
        source_manifest_files_order (List[str]) : order in which output files should occur in the SOURCE_FEED_MANIFEST
        pos_data_column_mapping (Dict[str, str]): mapping dataframe column names to the unified names
            needed to generate the outputs
        outputs_config (Dict[str, Dict[str, Any]]): outputs paths configuration with the provided structure:
            RGM_PS_VTM: {'path' : saving_path/RGM_PS_VTM.csv, 'kwargs' : ...}
            RGM_SIMULATION_CONFIG: {'path' : saving_path/RGM_SIMULATION_CONFIG.csv, 'kwargs' : ...}
            RGM_PS_STRUCTURE: {'path' : saving_path/RGM_PS_STRUCTURE.csv, 'kwargs' : ...}
            SOURCE_FEEDS_MANIFEST: {'path' : saving_path/SOURCE_FEEDS_MANIFEST.csv, 'kwargs' : ...}
            the ... represents the additional keyword arguments to be passed to the pandas to_csv() function

    Returns:
        Dict[str, pd.DataFrame]: dictionary with output dataframes, outputs are stored
            under keys: RGM_PS_VTM, RGM_SIMULATION_CONFIG, RGM_PS_STRUCTURE, SOURCE_FEEDS_MANIFEST
    """
    outputs = {}
    melted_sku_matrix = create_melted_sku_matrix(sku_matrix)
    # Note outputs keys should match the config keys, todo: add error if there is a mismatch
    outputs["RGM_PS_VTM"] = create_rgmx_ps_vtm_output(melted_sku_matrix)
    outputs["RGM_SIMULATION_CONFIG"] = create_rgmx_config(
        melted_sku_matrix, rmgx_parameters["RGM_SIMULATION_CONFIG"]
    )
    outputs["RGM_PS_STRUCTURE"] = create_ps_structure_output(
        point_of_sales_df, rmgx_parameters["RGM_PS_STRUCTURE"]["pos_data_column_mapping"]
    )
    outputs["SOURCE_FEEDS_MANIFEST"] = create_source_feeds_manifest(
       outputs, rmgx_parameters["SOURCE_FEEDS_MANIFEST"]["files_order"], outputs_config
    )
    outputs["RGM_SIMULATION_CONFIG"] = assure_same_index_between(
       outputs["RGM_PS_STRUCTURE"], outputs["RGM_SIMULATION_CONFIG"]
    )
    return outputs


def assure_same_index_between(
    ps_structure: pd.DataFrame, simulation_config: pd.DataFrame
) -> pd.DataFrame:
    """Reindexes Simulation config to be the same as PS Structure

    Args:
        ps_structure (pd.DataFrame): Rgmx output - 'RGM_PS_STRUCTURE'
        simulation_config (pd.DataFrame): Rgmx output - 'RGM_SIMULATION_CONFIG'

    Raises:
        pd.errors.InvalidIndexError: when ps_structure index and simulation_config index
            have different index values (order doesn't matter)

    Returns:
        pd.DataFrame: simulation_config with index aligned with the ps_structure
    """
    if ps_structure.index.sort_values().equals(simulation_config.index.sort_values()):
        return simulation_config.reindex(ps_structure.index)
    raise pd.errors.InvalidIndexError(
        "Simulation config cannot be reindex (not matching index values)"
    )


def create_melted_sku_matrix(sku_matrix: pd.DataFrame) -> pd.DataFrame:
    sku_mat = sku_matrix.copy()
    sku_mat[diagonal_df_bool_index(sku_mat)] = 0
    sku_mat.index.name = "index"
    melted_sku_matrix = sku_mat.melt(ignore_index=False, var_name="variable").reset_index()
    return melted_sku_matrix.rename(
        columns={"index": "UID_ROW", "variable": "UID_COLUMN", "value": "VTM_VALUE"}
    )


def create_rgmx_ps_vtm_output(melted_sku_matrix: pd.DataFrame) -> pd.DataFrame:
    """Returns RGM_PS_VTM dataframe

    Args:
        melted_sku_matrix (pd.DataFrame): dataframe with columns UID_ROW, UID_COLUMN and VTM VALUE

    Returns:
        pd.DataFrame: melted_sku_matrix with removed rows with 0 vtm value
    """
    vtm=melted_sku_matrix
    zero_counts = vtm.groupby('UID_ROW')['VTM_VALUE'].apply(lambda x: (x == 0).sum())

    K = vtm['UID_ROW'].nunique()
    filtered_uids = zero_counts[zero_counts == K].index.tolist()
    vtm['Zero_Transfer_UID'] = np.where((vtm['UID_ROW'].isin(filtered_uids)) & (vtm['UID_COLUMN']==vtm['UID_ROW']), 1, 0)
    vtm = vtm[(vtm['Zero_Transfer_UID'] == 1) | (vtm['VTM_VALUE'] != 0)]
    vtm.drop(columns=['Zero_Transfer_UID'], inplace=True)
    vtm=vtm.reset_index(drop=True)

    return vtm


def create_rgmx_config(melted_sku_matrix: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Returns RGM_SIMULATION_CONFIG

    Args:
        melted_sku_matrix (_type_): dataframe with columns UID_ROW, UID_COLUMN and VTM VALUE
        config (Dict): dictionary with keys: wr_avg_, wr_size_factor_, wr_index_factor_, wr_shape_factor_, sim_option

    Returns:
        pd.DataFrame: dataframe with UIDs and simulation parameters
    """
    params = dict(
        CHANNEL_NM=config["channel_nm"], #Chnage as per data team's requirement
        SIM_OPTION=config["sim_option"],
        SIM_PARAM_1=config["wr_avg_gbl_var"],
        SIM_PARAM_2=config["wr_size_gbl_var"],
        SIM_PARAM_3=config["wr_shape_gbl_var"],
        SIM_PARAM_4=config["wr_index_gbl_var"],
        SIM_PARAM_5=config["sim_param_5"],
        SIM_PARAM_6=config["sim_param_6"],
        SIM_PARAM_7=config["sim_param_7"],
        SIM_PARAM_8=config["sim_param_8"],
        SIM_PARAM_9=config["sim_param_9"],
        SIM_PARAM_10=config["sim_param_10"],
        SIM_BLOCK=config["sim_block"],
    )
    config_df = pd.DataFrame({"UID": melted_sku_matrix["UID_ROW"].unique()})
    for param_name, param_value in params.items():
        config_df[param_name] = param_value
    return config_df


def create_ps_structure_output(
    point_of_sales_df: pd.DataFrame, pos_data_column_mapping: Dict[str, str]
) -> pd.DataFrame:
    """Creates RGM_PS_STRUCTURE

    Args:
        point_of_sales_df (pd.DataFrame): point of sales dataframe
        pos_data_column_mapping (Dict[str, str]): mapping dataframe column names to the unified names
            needed to generate the outputs

    Returns:
        pd.DataFrame: Returns ps structure output dataframe
    """
    
    ps_structure_df = point_of_sales_df[list(pos_data_column_mapping.values())].copy()
    ps_structure_df = ps_structure_df.rename(
        columns=dict(zip(pos_data_column_mapping.values(), pos_data_column_mapping.keys()))
    )
    ps_structure_df["CHANNEL_NM"] = "ALL"
    ps_structure_df["PARENT_ID"] = ps_structure_df["END_NODE"]
    ps_structure_df["NODE_ID"] = ps_structure_df["END_NODE"]
    ps_structure_df["NODE_NAME"] = ps_structure_df["END_NODE"]
    ps_structure_df["DEPTH_LEVEL"] = [len([element for element in i if element != '-'])    
                                      for i in ps_structure_df["END_NODE"].str.split(" - ") ]
    
    ps_structure_df["LINEAGE"] = ps_structure_df["END_NODE"].copy()
    ps_structure_df["ITEM_ID"] = ps_structure_df["PPG_NM"].copy()
    ps_structure_df = ps_structure_df.drop(columns=["END_NODE", "PPG_NM"])
    return ps_structure_df.reset_index(drop=True)


def create_source_feeds_manifest(
    outputs: Dict[str, pd.DataFrame],
    source_manifest_files_order: List[str],
    outputs_config: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Creates SOURCE_FEEDS_MANIFEST, which is a dataframe with summary of the outputs

    Args:
        outputs (Dict[str, pd.DataFrame]): dictionary with key as a string output identifier and value being
            a pd.DataFrame with the output
        source_manifest_files_order (List[str]) : order in which output files should occur in the SOURCE_FEED_MANIFEST
        outputs_config (Dict[str, Dict[str, Any]]): dictionary with outputs IDS as keys and arguments to save file as
            as values. The values should contain the 'path' keyword.

    Raises:
        KeyError: if any of the output files does not have 'path' specified

    Returns:
        pd.DataFrame: Returns a dataframe with other outputs summary.
    """
    manifest: Dict[str, List] = defaultdict(list)
    for key, file_saving_spec in outputs_config.items():
        if("tag_" in outputs_config[key].keys()):
            print('key found', key)
            manifest["FEED_NM"].append(key)
            manifest["FILE_NM"].append(
                file_name_from_path(
                    file_saving_spec.get("filename", KeyError(f"File {key} does not have 'name' specified"))
                )
            )
            try:
                nrows = outputs[key].shape[0]
            except KeyError:
                nrows = 4
            manifest["ROW_COUNT"].append(nrows)
    manifest = pd.DataFrame(manifest)
    manifest = force_source_manifest_files_order(manifest, source_manifest_files_order)
    return add_feed_id_column_to_source_manifest(manifest)

def diagonal_df_bool_index(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a bool dataframe with True value on the matrix diagonal.

    Args:
        df (pd.DataFrame): dataframe with numeric values of a square shape (n, n)
    Raises:
        ValueError: Error when a dataframe has number of rows different from
            number of columns
    """
    nrows, ncols = df.shape[0], df.shape[1]
    if nrows != ncols:
        raise ValueError("Dataframe is not a square matrix")
    index = np.outer(np.repeat(False, nrows), np.repeat(False, nrows))
    for i in range(nrows):
        index[i, i] = True
    return pd.DataFrame(index, index=df.index, columns=df.columns)

def get_rmgx_output_generator_data_catalog(catalog):
    rgmx_outputs = {}
    for file_id in catalog:
        if catalog[file_id].get("tag_", "") == "rgmx_output":
            rgmx_outputs[file_id] = catalog[file_id]
            rgmx_outputs[file_id]
    return rgmx_outputs

def force_source_manifest_files_order(manifest, source_manifest_files_order):
    return manifest.set_index("FEED_NM").loc[source_manifest_files_order].reset_index()


def add_feed_id_column_to_source_manifest(manifest):
    manifest = manifest.reset_index().rename(columns={"index": "FEED_ID"})
    manifest["FEED_ID"] += 1  # to start from 1 instead of 0
    return manifest


def file_name_from_path(path: str) -> str:
    """Returns a file name from a path

    Args:
        path (str): a path to a file

    Returns:
        str: a name of the file
    """
    return Path(path).name
