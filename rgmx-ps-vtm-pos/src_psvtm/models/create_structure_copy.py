# -*- coding: utf-8 -*-
"""
author: Dotun Opasina
date: Dec 05 2022
"""
import os
import sys
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pathlib import Path
from pyspark.sql.types import *
from pyspark.sql.functions import *
from src.data.data_load import *
from src.features.data_prep import *
from src.utils.logger import logger
from src.utils.utils import *
from src.utils.version_control import *
from src.models.loess import run_loess
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import *
#####If error with dtype float(64) - run pip install rpy2==3.4.5

parameters = {}
CATALOG = {}

def set_params_catalog_ewb(params_ewb:dict, catalog_ewb:dict):
    """
    Function to set the parameters and catalog to be used with the econometrics module
    :param dict params_ewb: dictionary of parameters used for econometrics module
    :param dict catalog_ewb: dictionary of catalog used for econometrics module
    """
    global parameters
    global CATALOG
    parameters = params_ewb
    CATALOG = catalog_ewb
    logger.info("paramters and catalog for ewb instantiated")
    
def custom_group_columns_df (input_data: pd.DataFrame, params: dict,  case = 'lower') -> pd.DataFrame:
    """
    This function creates product attribute flags that the user may either define or use default parameters for 
    Args:
        input_data: input data given from initial POS data
        parameters: parameter dict
            These parameters are used to determine which attributes (values) in which columns to make dummy variables for
    Returns:
        input_data: dataframe with selected/computer-chosen variables as dummy variables 
    """
    if case.upper() == 'lower':
        input_data.columns = [x.lower() for x in input_data.columns]
    vars_selection = params["pos_purchase_structures"]['data']["variable_selection"]
    sales_col = params["default_vars"]['value_col']
    cust_vars = params["pos_purchase_structures"]['data']["variable_transformation_def"]
    for key in cust_vars.keys():
        for transforms in cust_vars[key].keys():
            col_name = key
            name_for_others = cust_vars[col_name][transforms]['name_to_use_for_others']
            column_name_to_use = cust_vars[col_name][transforms]['column_name_to_use']
            
            if cust_vars[col_name][transforms]['how'] not in ['', 'default']:
                if cust_vars[col_name][transforms]['how'] == 'top':
                    sort_vars = input_data.groupby(col_name).agg({
                         cust_vars[col_name][transforms]['sort_on']: 'sum'
                     }).reset_index()
                    sort_vars.columns = [col_name, 'to_order']
                    sort_vars = sort_vars.sort_values('to_order', ascending = False).reset_index(drop = True)
                    num_included = cust_vars[col_name][transforms]['top']
                    attrs_included = list(sort_vars[col_name][:num_included])        
                elif cust_vars[col_name][transforms]['how'] == 'predefined':
                    attrs_included = cust_vars[col_name][transforms]['what']
            else:
                sort_vars = input_data.groupby(col_name).agg({
                         sales_col: 'sum'
                     }).reset_index()
                sort_vars.columns = [col_name, 'to_order']
                sort_vars = sort_vars.sort_values('to_order', ascending = False).reset_index(drop = True)
                attrs_included = list(sort_vars[col_name][:params["default_vars"]['top']])
        
            input_data[column_name_to_use]=np.where(input_data[col_name].isin(attrs_included),input_data[col_name],name_for_others)
        
    return input_data

def flag_items(input_data:pd.DataFrame, params: dict):
    """
    Function to flag input data
    :param pd.DataFrame input_data: input containing skus and attribute information
    :param dict params_ewb: dictionary of parameters used for econometrics module
    :return: pd.DataFrame input_data: input containing flagged input data columns
    """
    vars_selection = params["pos_purchase_structures"]['data']["variable_selection"]
    sales_col = params["default_vars"]['value_col']
    cust_vars = params["pos_purchase_structures"]['data']["variable_transformation_def"]
    for item in vars_selection:
        for cust in cust_vars[item]["what"]:
            input_data[cust] = input_data[cust].apply(lambda x: cust if x == 1 else cust_vars[item]["name_to_use_for_others"])
    return input_data

def get_product_attr_flags(input_data: pd.DataFrame, dataframe: bool, params: dict,  case = 'lower') -> pd.DataFrame:
    """
    This function creates product attribute flags that the user may either define or use default parameters for 
    Args:
        input_data: input data given from initial POS data
        parameters: parameter dict
            These parameters are used to determine which attributes (values) in which columns to make dummy variables for
    Returns:
        input_data: dataframe with selected/computer-chosen variables as dummy variables 
    """
    if not dataframe:
        input_data = input_data.toPandas()
    if case.upper() == 'lower':
        input_data.columns = [x.lower() for x in input_data.columns]
    vars_selection = params["pos_purchase_structures"]['data']["variable_selection"]
    sales_col = params["default_vars"]['value_col']
    cust_vars = params["pos_purchase_structures"]['data']["variable_transformation_def"]

    for item in vars_selection:
        col_name = item
        if col_name in cust_vars and cust_vars[col_name]['how'] not in ['', 'default']:
            if cust_vars[col_name]['how'] == 'top':
                sort_vars = input_data.groupby(col_name).agg({
                     cust_vars[col_name]['on']: 'sum'
                 }).reset_index()
                sort_vars.columns = [col_name, 'to_order']
                sort_vars = sort_vars.sort_values('to_order', ascending = False).reset_index(drop = True)
                num_included = cust_vars[col_name]['top']
                attrs_included = list(sort_vars[col_name][:num_included])        
            elif cust_vars[col_name]['how'] == 'predefined':
                attrs_included = cust_vars[col_name]['what']
        else:
            sort_vars = input_data.groupby(col_name).agg({
                     sales_col: 'sum'
                 }).reset_index()
            sort_vars.columns = [col_name, 'to_order']
            sort_vars = sort_vars.sort_values('to_order', ascending = False).reset_index(drop = True)
            attrs_included = list(sort_vars[col_name][:params["default_vars"]['top']])
        
        dummies_df = pd.get_dummies(input_data[col_name], columns=[col_name])[attrs_included]
        other_list = ~input_data[col_name].isin(attrs_included)
        dummies_df['other'] = other_list
        dummies_df['other']  = dummies_df['other'].astype(int)
        dummies_df.columns = [col_name + "_" + col if 'other' in col else col for col in dummies_df.columns]
        input_data = pd.concat([input_data,dummies_df], axis=1)
    return input_data

def get_prod_attr_flags_df(input_data: pd.DataFrame, params: dict, dataframe=False) -> pd.DataFrame:
    """
    Function to flag attribute in input data based on the parameters passed in
    :param DataFrame input_data: input containing skus and attribute information
    :param dict params: used to specify variables to encode and how
    :param bool dataframe: boolean used to determine if input_Data is already in spark pandas format or not
    :return: DatFrame input_data_flags: data frame containing flagged information
    """
    input_data_flags = get_product_attr_flags(input_data=input_data, params=params, case='Lower',
                                              dataframe=dataframe)  # function from data prep in src features
    return input_data_flags


def get_pair_summary_drop(input_data_flags_df: pd.DataFrame, params:dict) -> pd.DataFrame:
    """
    Function to get pairs attribute to drop from the input dataframe
    :param DataFrame input_data_flags_df: flagged dataframe from get_prod_attr_flags_df
    :param dict params: dictionary of parameters
    :return: DataFrame pair_summary_drop: dataframe with attributes to drop
    """
    old_sales_col = params['default_vars']['value_col']  # ensure that the columns matches with R input
    sales_col = params['default_vars']['value_col']
    drop_thresh = params['default_vars']["drop_thresh"]
    grouping_cols = params["pos_purchase_structures"]['data']['grouping_cols']
    try:
        input_data_flags_df = input_data_flags_df.toPandas()
    except:
        pass
    # group by GOOD general flag based on parameters above
    input_df_good = input_data_flags_df[input_data_flags_df.general_flag == 'GOOD']
    input_df_good_grouped = input_df_good.groupby(grouping_cols).agg({old_sales_col: 'sum'})
    input_df_good_grouped = input_df_good_grouped.reset_index().rename(columns={old_sales_col: sales_col})
    
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    try:
        input_df_good_grouped = input_df_good_grouped.toPandas()
    except:
        pass

    input_df_good_grouped = input_df_good_grouped.sort_values(by=sales_col, ascending=False)
    total_sales = input_df_good_grouped[sales_col].sum()
    input_df_good_grouped['value_pct'] = (input_df_good_grouped[sales_col] / total_sales).astype(float)
    input_df_good_grouped['value_pct_cumm'] = input_df_good_grouped['value_pct'].cumsum()

    # should we filter value_pct_cumm by drop_thresh or fixed value of 1 as in R?
    input_df_good_grouped['cumm_flag'] = input_df_good_grouped['value_pct_cumm'].apply(
        lambda x: 1 if float(x) >= drop_thresh else 0)
    input_df_good_grouped['general_flag'] = input_df_good_grouped['cumm_flag'].apply(
        lambda x: "GOOD" if x == 1 else "DROP: " + str(drop_thresh) + " cut off")

    mask = (input_df_good_grouped.general_flag.str.contains("DROP"))
    mask = mask[mask==True]
    pair_summary_drop = input_df_good_grouped.loc[mask.index,:].reset_index(drop=True)

    final_pair_summary_drop = pair_summary_drop.loc[:, grouping_cols]
    return final_pair_summary_drop


def drop_pairs(input_data_flags_df: pd.DataFrame, pair_summary_drop: pd.DataFrame, params:dict) -> pd.DataFrame:
    """
    Function to drop from pair summary function within the input dataframe
    :param DataFrame input_data_flags_df: input dataframe
    :param DataFrame pair_summary_drop: pair summary to drop
    :param dict params: parameter dictionary
    :return: DataFrame data_for_PS: data input to be used within create structure
    """
    grouping_cols = params["pos_purchase_structures"]['data']['grouping_cols']

    try:
        input_data_flags_df = input_data_flags_df.toPandas()
    except:
        input_data_flags_df = input_data_flags_df

    try:
        pair_summary_drop = pair_summary_drop.toPandas()
    except:
        pair_summary_drop = pair_summary_drop

    #### go through each pair summary and exclude those that are part of the pairs
    to_exclude = pd.merge(input_data_flags_df, pair_summary_drop, how='right', on=grouping_cols)
    df3 = input_data_flags_df.merge(to_exclude, on=input_data_flags_df.columns.tolist(), how='left', indicator=True)
    data_for_PS = df3[df3['_merge'] == 'left_only'].drop(columns='_merge')
    return data_for_PS


def calc_opi_xpi(input_attr_df: pd.DataFrame, attr:str, params:dict, filt_index: pd.Series) -> pd.DataFrame:
    """
    Function to calculate the internal and external price index for the given attribute
    :param pd.DataFrame input_attr_df: input attribute dataframe
    :param str attr: attribute to calculate indexes for
    :param Dict params: parameter to be used in calc_opi_xpi
    :param pd.Series filt_index: boolean series for index to filter on
    :return: pd.DataFrame df_calcs: dataframe containing calculated opi and xpi information
    """
    logger.info("\n In calc opi xpi \n")
    try:
        input_attr_df = input_attr_df.toPandas()
    except:
        input_attr_df = input_attr_df
    time_period = params['default_vars']['time_period']
    item_col = params['default_vars']['item']
    vol_col = params['default_vars']['volume']
    value_col = params['default_vars']['value_col']
    channel_col = params['default_vars']['channel']
    wd_col = params['default_vars']['weight_distr']

    item_u = input_attr_df[item_col]
    per = input_attr_df[time_period]
    input_attr_df['combine'] = per.astype(str) + " " + item_u.astype(str) + " " + filt_index.astype(str)
    vol_own = input_attr_df.groupby('combine')[vol_col].transform('sum')
    price_own = (input_attr_df.groupby('combine')[value_col].transform('sum')) / vol_own
    # Calculate distribution for owner based on weight distribution and volume
    input_attr_df["dist_own"] = input_attr_df[wd_col] * input_attr_df[vol_col]
    dist_own = input_attr_df.groupby('combine')['dist_own'].transform('sum') / vol_own

    input_attr_df['combine_opi'] = per.astype(str) + " " + \
                                   input_attr_df[channel_col].astype(str) + " " + input_attr_df[attr].astype(
        str) + " " + filt_index.astype(str)

    # External production calculation
    vol_opiX = input_attr_df.groupby('combine_opi')[vol_col].transform('sum')
    price_opiX = input_attr_df.groupby('combine_opi')[value_col].transform('sum') / vol_opiX
    distr_opiX = input_attr_df.groupby('combine_opi')['dist_own'].transform('sum') / vol_opiX

    # Internal/ own production calculations
    p_opi = ((price_opiX * vol_opiX) - (price_own * vol_own)) / (vol_opiX - vol_own)
    d_opi = ((distr_opiX * vol_opiX) - (dist_own * vol_own)) / (vol_opiX - vol_own)
    v_opi = vol_opiX - vol_own

    input_attr_df["combine_tot"] = per.astype(str) + " " + input_attr_df[channel_col].astype(str) + " " + filt_index.astype(str)

    v_tot = input_attr_df.groupby('combine_tot')[vol_col].transform('sum')
    v_xpi = v_tot - vol_opiX

    p_tot = input_attr_df.groupby('combine_tot')[value_col].transform('sum') / v_tot
    d_tot = input_attr_df.groupby('combine_tot')['dist_own'].transform('sum') / v_tot

    p_xpi = ((p_tot * v_tot) - (price_opiX * vol_opiX)) / (v_xpi)
    d_xpi = ((d_tot * v_tot) - (distr_opiX * vol_opiX)) / (v_xpi)

    df_calcs = pd.DataFrame(list(zip(p_opi.values, p_xpi.values, d_opi.values, d_xpi.values, v_opi.values, v_xpi.values)))
    df_calcs = df_calcs.fillna(0)
    df_calcs = df_calcs.astype(float)
    
    input_attr_df = input_attr_df.drop(columns=['combine_opi','combine_tot','combine', 'dist_own' ], axis=1) # drop artificial columns for some reason df is updated
    df_calcs.columns = ['p_opi', 'p_xpi', 'd_opi', 'd_xpi', 'v_opi', 'v_xpi']  # own and external xpi outputs
    return df_calcs

def calc_ratio(model_cf: np.array, fairshare: np.array) -> np.array:
    """
    Function to calculate the model's ratio based on regression coefficient and fairshare
    :param np.array model_cf: array containing model coefficient to be calculated
    :param np.array fairshare: array containing fairshare values to be used for ration
    :return np.array ratio: np.array containing the returned calculated ratio
    """
    if np.nan in model_cf:
        ratio = np.array([1,1])
    elif model_cf.max() < 0:
        ratio = np.array([1,1])
    elif model_cf.min() > 0:
        sum_val_ratio = model_cf/model_cf.sum()
        ratio = sum_val_ratio/fairshare
    else:
        model_cf = np.where(model_cf <0, 0, model_cf)
        ratio = model_cf/model_cf.sum()
    return ratio


def get_res_df(target:pd.DataFrame, input_df:pd.DataFrame,
               loess_sh:pd.DataFrame, attr:str, params:dict,
               useshare:bool=True, filt_index:pd.Series=pd.Series(True)) -> pd.DataFrame:
    """
    Function to perform multiple regression for attribute price, volume, and distributions indexes
    :param pd.DataFrame target: coefficients gotten from calc_opi_xpi to be used for regression
    :param pd.DataFrame input_df: input attribute dataframe
    :param pd.DataFrame loess_sh: loess_sh dataframe to retrieve trend_term information
    :param str attr: attribute to
    :param dict params: dictionary of multiple parameters
    :param bool useshare:boolean to calculate own_vol ratio
    :param pd.Series filt_index: rows to be filtered from input_df
    :return: pd.DataFrame res_new: dataframe containing all the regression results
    """
    logger.info("\n Getting result dataframe\n")
    item_col = params['default_vars']['item']
    vol_col = params['default_vars']['volume']
    value_col = params['default_vars']['value_col']
    trend_col = params['default_vars']['trend_col']
    wd_col = params['default_vars']['weight_distr']
    seasons_var = params['default_vars']['seasons']
    add_pv_cols = params['default_vars']['add_pv_model_cols']
    add_dv_cols = params['default_vars']['add_dv_model_cols']
    
    try:
        input_df = input_df.toPandas()
    except:
        input_df = input_df
    
    seasons_col = input_df[seasons_var].unique().tolist() # get unique season columns
    filt_index = filt_index[filt_index == True]
    filt_items = input_df.loc[filt_index.index, item_col]
    unique_items = filt_items.drop_duplicates().values
    
    #get_loess_items = loess_sh[item_col].notnull() 
    #get_loess_items = get_loess_items[get_loess_items == True]

    res_rows = []
    res_rownames = []
    attr_res = []
    price = input_df[value_col] / input_df[vol_col]
    if useshare:
        own_vol = input_df[vol_col] / (target['v_opi'] + target['v_xpi'])
    else:
        own_vol = input_df[vol_col]

    own_vol_df = pd.DataFrame(own_vol)
    own_vol_df.columns = ['own_vol']
    
    ope_xpi_dynamic_master = pd.DataFrame()
    input_df_dynamic_master =  pd.DataFrame()
    
    
    for ui in unique_items:
        non_zero_idx = (filt_items == ui) \
                       & ((target['v_opi'] * price * target['v_xpi'] * input_df[vol_col]) != 0) & filt_index #& get_loess_items
        
        if non_zero_idx.sum() > 30:
            non_zero_idx = non_zero_idx[non_zero_idx == True]
            res_rownames.append(ui)
            attr_res.append(list(input_df.loc[non_zero_idx.index, attr])[0])
            fairshare = target.loc[non_zero_idx.index, ['v_opi', 'v_xpi']].sum(axis=0) / target.loc[
                non_zero_idx.index, ['v_opi', 'v_xpi']].sum().sum()
            fairshare = np.array(fairshare)
            price_col = price.loc[non_zero_idx.index,]

            # Perform Price Volume REGRESSION
            own_price_log = np.log(target.loc[non_zero_idx.index, 'p_opi'] / price_col)
            ext_price_log = np.log(target.loc[non_zero_idx.index, 'p_xpi'] / price_col)
            wd_log = np.log(input_df.loc[non_zero_idx.index, wd_col])
            loess_trend = loess_sh.loc[non_zero_idx.index, trend_col]
            season_cols = loess_sh.loc[non_zero_idx.index, seasons_col]
            log_own_vol = np.log(own_vol_df.loc[non_zero_idx.index, 'own_vol']).astype(float)
            
            # dynamic linear regression coefficients
            if len(add_pv_cols) > 0:
                added_pv_cols = loess_sh.loc[non_zero_idx.index, add_pv_cols] # included added pv columns in regression
                X_cf_pv = pd.concat([own_price_log, ext_price_log, wd_log, loess_trend, season_cols, added_pv_cols], axis=1)\
                .reindex(index=non_zero_idx.index).replace([np.inf, -np.inf], np.nan).fillna(0)
            else:
                X_cf_pv = pd.concat([own_price_log, ext_price_log, wd_log, loess_trend, season_cols], axis=1)\
                    .reindex(index=non_zero_idx.index).replace([np.inf, -np.inf], np.nan).fillna(0)
            X_cf_pv = X_cf_pv.values.astype(float)

            cf_pv = LinearRegression(normalize=True).fit(X_cf_pv, log_own_vol).coef_[0:2] # select own price, and ext price coefficients
            cf_pv = np.array(cf_pv)
            ratio_pv = calc_ratio(cf_pv, fairshare)

            # Perform Distribution Volume REGRESSION 
            own_dist_log = np.log(target.loc[non_zero_idx.index, 'd_opi'] / input_df.loc[non_zero_idx.index, wd_col])
            ext_dist_log = np.log(target.loc[non_zero_idx.index, 'd_xpi'] / input_df.loc[non_zero_idx.index, wd_col])
            
            # dynamic linear regression coefficients
            if len(add_dv_cols) > 0:
                added_dv_cols = loess_sh.loc[non_zero_idx.index, add_dv_cols] # included added dv columns in regression
                X_cf_dv = pd.concat([own_dist_log, ext_dist_log, price_col,loess_trend, season_cols, added_dv_cols], axis=1)\
                .reindex(index=non_zero_idx.index).replace([np.inf, -np.inf], np.nan).fillna(0)
            else:
                X_cf_dv = pd.concat([own_dist_log, ext_dist_log, price_col,loess_trend, season_cols], axis=1)\
                .reindex(index=non_zero_idx.index).replace([np.inf, -np.inf], np.nan).fillna(0)

            
            X_cf_dv = pd.concat([own_dist_log, ext_dist_log, price_col,loess_trend, season_cols], axis=1)\
                .reindex(index=non_zero_idx.index).replace([np.inf, -np.inf], np.nan).fillna(0)
            cf_dv = LinearRegression(normalize=True).fit(X_cf_dv, log_own_vol).coef_[0:2] # select own distribution and ext distribution coefficients
            cf_neg = np.array([-x for x in cf_dv])
            ratio_dv = calc_ratio(cf_neg, fairshare)

            price_gap = np.nanmean((price_col - target.loc[non_zero_idx.index, 'p_opi']).abs() < (
                    price_col - target.loc[non_zero_idx.index, 'p_xpi']).abs())

            # Own Trend and Season to Volume regression
            X_cf_ts_own = pd.concat([loess_trend, season_cols], axis=1)\
                        .reindex(index=non_zero_idx.index).replace([np.inf, -np.inf], np.nan).fillna(0)
            y_cf_ts_own = np.log(input_df.loc[non_zero_idx.index, vol_col])
            cf_ts_own = np.array(LinearRegression(normalize=True).fit(X_cf_ts_own, y_cf_ts_own).coef_)

            # cf_ts_opi Trend & Season to Volume Own Price Index regression 
            X_cf_ts_opi = pd.concat([loess_trend, season_cols], axis=1)\
                                        .reindex(index=non_zero_idx.index).replace([np.inf, -np.inf],  np.nan).fillna(0)
            y_cf_ts_opi = np.log(target.loc[non_zero_idx.index, 'v_opi']).replace([np.inf, -np.inf], np.nan).fillna(0)

            X_cf_ts_opi = X_cf_ts_opi.values.astype(float)
            y_cf_ts_opi = y_cf_ts_opi.values.astype(float)
            cf_ts_opi = np.array(LinearRegression(normalize=True).fit(X_cf_ts_opi, y_cf_ts_opi).coef_)
            
            # cf_ts_opi Trend & Season to Volume External Price Index regression 
            X_cf_ts_xpi = pd.concat([loess_trend, season_cols], axis=1)\
                                .reindex(index=non_zero_idx.index).replace([np.inf, -np.inf],np.nan).fillna(0)
            y_cf_ts_xpi = np.log(target.loc[non_zero_idx.index, 'v_xpi'])
            X_cf_ts_xpi = X_cf_ts_xpi.values.astype(float)
            y_cf_ts_xpi = y_cf_ts_xpi.values.astype(float)
            cf_ts_xpi = np.array(LinearRegression(normalize=True).fit(X_cf_ts_xpi, y_cf_ts_xpi).coef_)

            trend_fit = np.abs(cf_ts_own[0] - cf_ts_opi[0]) < np.abs(cf_ts_own[0] - cf_ts_xpi[0])
            season_fit = (np.abs(cf_ts_own[1:] - cf_ts_opi[1:])).sum() < (np.abs(cf_ts_own[1:] - cf_ts_xpi[1:])).sum()

            ####Gross elasticity regression
            X_gross_elas = pd.concat([np.log(price_col), wd_log, loess_trend, season_cols], axis=1)\
                .reindex(index=non_zero_idx.index).replace([np.inf, -np.inf], np.nan).fillna(0)
            y_gross_elas = np.log(own_vol.loc[non_zero_idx.index])

            X_gross_elas = X_gross_elas.values.astype(float)
            y_gross_elas = y_gross_elas.values.astype(float)
            gross_elas = np.array(LinearRegression(normalize=True).fit(X_gross_elas, y_gross_elas).coef_[1])

            row = [ratio_pv[0], ratio_pv[1], ratio_dv[0], ratio_dv[1], price_gap, int(trend_fit), int(season_fit),
                   gross_elas, input_df.loc[non_zero_idx.index, vol_col].sum()]
            res_rows.append(row)
            
            #loess_season = loess_sh.loc[non_zero_idx.index,].reset_index(drop=True)
            #loess_season ['sku_modelled'] = ui
            #loess_season ['attr_modelled'] = attr
            #loess_season_master = loess_season_master.append(loess_season)
            
            ope_xpi_dynamic = target.loc[non_zero_idx.index,].reset_index(drop=True)
            ope_xpi_dynamic ['sku_modelled'] = ui
            ope_xpi_dynamic ['attr_modelled'] = attr
            ope_xpi_dynamic_master = ope_xpi_dynamic_master.append(ope_xpi_dynamic)
            
            input_df_dynamic = input_df.loc[non_zero_idx.index,].reset_index(drop=True)
            input_df_dynamic ['sku_modelled'] = ui
            input_df_dynamic ['attr_modelled'] = attr
            input_df_dynamic_master = input_df_dynamic_master.append(input_df_dynamic)
            
    # rename res_new to att imp df
    res_new = pd.DataFrame(res_rows)
    res_new[item_col] = res_rownames
    res_new = res_new.set_index(res_new[item_col])
    res_new = res_new.drop([item_col], axis=1)
    res_new['attr'] = attr_res

    if res_new.empty == True:
        logger.info("ERROR: Empty df for %s in get_res_df ", attr)
        return [None,ope_xpi_dynamic_master,input_df_dynamic_master]
    res_new.columns = ["index_opi", "index_xpi", "index_odi", "index_xdi", "price_gap", "trend", "season",
                       "elasticity_calc", "tot_vol", 'attr']
    return [res_new,ope_xpi_dynamic_master,input_df_dynamic_master]


def _weighted(dataframe: pd.DataFrame, cols:list, wt:list=[]) -> pd.Series:
    """
    helper function to calculate weighted sum
    :param pd.DataFrame dataframe: input dataframe
    :param list cols: list columns to use for values
    :param list wt: weighted values array
    :return pd.Series val: calculated weight average values
    """
    val = dataframe[cols]
    if is_numeric_dtype(val):
        return (val * wt).sum() / wt.sum()
    return val


def get_agg_elasticities(res_df: pd.DataFrame, attr:str, capp:int=2, params=None):
    """
    Function to calculate summary aggregated elasticities for results
    :param pd.DataFrame res_df: result dataframe gotten from get_res_df
    :param str attr: current attribute
    :param int capp:int used to filter capp amount
    :return pd.DataFrame summary_res: summary dataframe
    """

    logger.info("creating aggregate elasticities summaries")
    try:
        res_df = res_df.toPandas()
    except:
        res_df = res_df

    resdf2 = res_df.copy()
    resdf2['tot_times_e'] = res_df['tot_vol'] * res_df['elasticity_calc']
    resdf3 = resdf2.groupby('attr').agg({
        'tot_vol': 'sum',
        'tot_times_e': 'sum'
    })
    attrelasticity = resdf3['tot_times_e'] / resdf3['tot_vol']
    attrelasticity = pd.DataFrame(attrelasticity.values, index=attrelasticity.index, columns=["attrelasticity"])
    avgelasticity = (resdf2['tot_vol'] * resdf2['elasticity_calc']).sum() / (resdf2['tot_vol']).sum()
    resdf2 = resdf2.merge(attrelasticity, how="left", on=['attr'])
    # determine if less than average elasticity
    resdf2['elasticity'] = np.where(
        np.abs(np.array(resdf2['elasticity_calc']) - np.array(resdf2['attrelasticity'])) < np.abs(
            np.array(resdf2['elasticity_calc']) - np.array(avgelasticity)), 1, 0)

    if capp != 0:
        # update based on values greater than capp
        f12 = resdf2.iloc[:, 1] / resdf2.iloc[:, 0]  # index_xpi / index_opi
        mask = f12 > capp
        resdf2.iloc[mask, 0] = 1 / np.sqrt(capp)
        resdf2.iloc[mask, 1] = np.sqrt(capp)

        # update based on values less than capp
        mask = f12 < 1 / capp
        resdf2.iloc[mask, 1] = 1 / np.sqrt(capp)
        resdf2.iloc[mask, 0] = np.sqrt(capp)

        f34 = resdf2.iloc[:, 3] / resdf2.iloc[:, 2]
        mask = f34 > capp
        resdf2.iloc[mask, 2] = 1 / np.sqrt(capp)
        resdf2.iloc[mask, 3] = np.sqrt(capp)

        mask = f34 < 1 / capp
        resdf2.iloc[mask, 3] = 1 / np.sqrt(capp)
        resdf2.iloc[mask, 2] = np.sqrt(capp)

    resdf3 = resdf2[['index_opi', 'index_xpi', 'index_odi', 'index_xdi', 'price_gap',
                     'trend', 'season','elasticity', 'tot_vol', 'attr']]

    summary_res1 = resdf3[
        ['index_opi', 'index_xpi', 'index_odi', 'index_xdi', 'price_gap', 'trend', 'season', 'elasticity']].apply(
        lambda x: _weighted(x, cols=x.index, wt=resdf3["tot_vol"]))
    summary_res1 = pd.DataFrame(summary_res1).T
    summary_res1[attr] = 'total'
    summary_res1 = summary_res1.set_index(attr)

    summary_res2_df = resdf3.copy()
    summary_res2_df.loc[:, ['index_opi', 'index_xpi', 'index_odi', 'index_xdi', 'price_gap', 'trend', 'season',
                            'elasticity']] = summary_res2_df.loc[:,
                                             ['index_opi', 'index_xpi', 'index_odi', 'index_xdi', 'price_gap', 'trend',
                                              'season', 'elasticity']].multiply(summary_res2_df.loc[:, 'tot_vol'],
                                                                                axis="index")

    total_aggr = summary_res2_df.groupby('attr')['tot_vol'].sum().to_frame()
    all_sum = summary_res2_df[
        ['index_opi', 'index_xpi', 'index_odi', 'index_xdi', 'price_gap', 'trend', 'season', 'elasticity',
         'attr']].groupby('attr').sum()
    summary_res2_df = pd.DataFrame(all_sum.values / total_aggr.values, total_aggr.index,
                                   columns=['index_opi', 'index_xpi', 'index_odi', 'index_xdi', 'price_gap', 'trend',
                                            'season', 'elasticity'])
    summary_vol = resdf3.groupby('attr')['tot_vol'].sum().to_frame() / resdf3['tot_vol'].sum()
    summary_res = pd.concat([summary_res1, summary_res2_df], axis=0)
    summary_res['ratio_price'] = summary_res.iloc[:, 0].values / summary_res.iloc[:, 1].values
    summary_res['ratio_dist'] = summary_res.iloc[:, 2].values / summary_res.iloc[:, 3].values
    summary_res['share'] = 1
    summary_res.loc[summary_vol.index, 'share'] = summary_vol.values

    summary_res = summary_res.round(2)

    return summary_res


def attr_imp(input_df: pd.DataFrame, attr:str, input_filt: pd.Series, params:dict = None, useshare:bool=True):
    """
    Function to calculate attribute importance using input dataframe and calc_opi_xpi results
    :param pd.DataFrame input_df: input attribute dataframe
    :param str attr: attribute inputs
    :param pd.Series input_filt: indexes to filter in the input dataframe
    :param dict params: universal dictionary
    :param bool useshare: boolean to know when to use share or not for calculating own volume
    :return list[pd.DataFrame] [summary_df, res_df]: list used to return summary and result df(for interim results)
    """
    logger.info("In attribute importance for %s ", attr)
    time_period = params['default_vars']['time_period']
    vol_col = params['default_vars']['volume']
    value_col = params['default_vars']['value_col']
    attrimp_quant = params['default_vars']['attrimp_quantile'] #0.9
    seasons_var = params['default_vars']['seasons']
    loess_target_col = "log_" + vol_col
    
    
    res_calc_df = calc_opi_xpi(input_attr_df=input_df, attr=attr, params=params, filt_index=input_filt)
    res_calc_df['per'] = input_df[time_period]
    res_calc_df['sales'] = res_calc_df.v_opi + res_calc_df.v_xpi
    total_sales = res_calc_df.groupby('per')['sales'].transform('sum')

    # filter out sales greater than specific quantile
    outlier = total_sales > (1.5 * total_sales.quantile(q=attrimp_quant))
    outlier_filt = input_filt & (res_calc_df["v_opi"] > 0) & (res_calc_df["v_xpi"] > 0) & (input_df[vol_col] > 0) \
         & (input_df[value_col] > 0) & (outlier == False)

    outlier_filt = outlier_filt[outlier_filt == True]

    if useshare:
        ownvol = input_df[vol_col] / (res_calc_df.v_opi + res_calc_df.v_xpi)
    else:
        ownvol = input_df[vol_col]
    data_for_loess_sh = input_df.copy()
    data_for_loess_sh = data_for_loess_sh.replace([np.inf, -np.inf], np.nan)#.fillna(0)
    data_for_loess_sh[loess_target_col] = ownvol#.fillna(0)
    data_for_loess_sh = pd.get_dummies(data_for_loess_sh, columns=[seasons_var], prefix='', prefix_sep='') #Get dummies for seasons
    data_for_loess_sh = data_for_loess_sh.replace([np.inf, -np.inf], np.nan).fillna(0)
    catalog = CATALOG
    # run_loess is from src/models
    
    data_for_loess_sh['ownvol'] =ownvol
    data_for_loess_sh ['attr_modelled'] = attr
    
    loess_sh = run_loess(data_for_loess_raw=data_for_loess_sh, params=parameters, catalog=catalog)
    get_res_extended_list = get_res_df(target=res_calc_df, input_df=input_df, loess_sh=loess_sh, attr=attr, params=params, useshare=True, filt_index=outlier_filt)
    
    res_df = get_res_extended_list[0]
    ope_xpi_dynamic_master = get_res_extended_list[1]
    input_df_dynamic_master =  get_res_extended_list[2]
    loess_season_master = data_for_loess_sh#get_res_extended_list[3]
    
    #res_df = get_res_df(target=res_calc_df, input_df=input_df, loess_sh=loess_sh, attr=attr, params=params, useshare=True, filt_index=outlier_filt)
    if res_df is None:
        return [None, None, ope_xpi_dynamic_master , input_df_dynamic_master , loess_season_master]
    summary_res_df = get_agg_elasticities(res_df, attr=attr, capp=2, params=params)

    return [summary_res_df, res_df , ope_xpi_dynamic_master , input_df_dynamic_master , loess_season_master]


def attr_imp_batch(input_df: pd.DataFrame, attr_list: list, input_filt: pd.Series,
                    params: dict= None, useshare: bool =True):
    """
    Function to calculate the attribute importance in a batch
    :param pd.DataFrame input_df: input attribute dataframe
    :param attr_list:
    :param pd.Series input_filt: indexes to filter in the input dataframe
    :param dict params: universal dictionary
    :param bool useshare: boolean to know when to use share or not for calculating own volume
    :return: pd.DataFrame resbatch: result batch returned with calculated prodshare & ratio values
    """
    logger.info("SUCCESS: In attr importance batch for %s ", attr_list)
    try:
        input_df = input_df.toPandas()
    except:
        input_df = input_df

    filter_df = input_df.loc[input_filt.index, attr_list]
    filter1 = filter_df.apply(lambda x: x.nunique() > 1, axis=0)
    filt_list = np.array(attr_list)
    filt_list = filt_list[filter1 == True]

    if len(filt_list) == 0:
        return None
    resbatch_rows = []
    resbatch_row_names = []
    
    loess_season_master = pd.DataFrame()
    res_df_master = pd.DataFrame()
    ope_xpi_dynamic_master = pd.DataFrame()
    input_df_dynamic_master = pd.DataFrame()
    
    
    for attr in filt_list:
        #perform attribute imputation and select summary result
        attr_imp_result_list = attr_imp(input_df=input_df, attr = attr, input_filt = input_filt,
                          params = params, useshare= useshare)
        att_df = attr_imp_result_list[0]
        res_df = attr_imp_result_list[1]
        res_df_master = res_df_master.append(res_df)
        ope_xpi_dynamic = attr_imp_result_list[2]
        ope_xpi_dynamic_master = ope_xpi_dynamic_master.append(ope_xpi_dynamic)
        input_df_dynamic = attr_imp_result_list[3]
        input_df_dynamic_master = input_df_dynamic_master.append(input_df_dynamic)
        loess_season = attr_imp_result_list[4]
        loess_season_master = loess_season_master.append(loess_season)
        
        #att_df = attr_imp(input_df=input_df, attr = attr, input_filt = input_filt,
        #                  params = params, useshare= useshare)[0]
        if att_df is None:
            logger.info("WARNING: No attribute importance for this attribute %s", attr)
        else:
            sum_att_df = att_df.isnull().sum().sum()
            if (sum_att_df == 0 and len(att_df) > 2):
                res_val = att_df.drop('total')
                prodshare = res_val['share'].sort_values(ascending=False)[1]
                ratio_price_min = res_val['ratio_price'].min()
                ratio_dist_min = res_val['ratio_dist'].min()
                att_df = att_df.loc['total', :].to_frame().T.reset_index(drop=True)
                att_df['ratio_price_min'] = ratio_price_min
                att_df['ratio_dist_min'] = ratio_dist_min
                att_df['Prodshare'] = prodshare
                att_df = att_df[
                    ['index_opi', 'index_xpi', 'index_odi', 'index_xdi', 'price_gap', 'trend', 'season', 'elasticity',
                     'ratio_price', 'ratio_dist', 'ratio_price_min', 'ratio_dist_min', 'Prodshare']]
                resbatch_row_names.append(attr)
                resbatch_rows.append(att_df)
    if len(resbatch_rows) == 0:
        logger.info("WARNING: No attribute importance for this batch %s ", str(filt_list))
        return [None, loess_season_master,input_df_dynamic_master,ope_xpi_dynamic_master,res_df_master]
    elif len(resbatch_rows) == 1:
        resbatch = pd.DataFrame(resbatch_rows[0])
    else:
        resbatch = pd.concat(resbatch_rows)

    resbatch['to_index'] = resbatch_row_names
    resbatch = resbatch.set_index('to_index')
    resbatch.columns = ['index_opi', 'index_xpi', 'index_odi', 'index_xdi', 'price_gap', 'trend', 'season',
                        'elasticity', 'ratio_price', 'ratio_dist', 'ratio_price_min', 'ratio_dist_min', 'ProdShare']
    resbatch = resbatch[resbatch['ProdShare'].notna()]
    # log final imputation batch
    logger.info("SUCCESS: returning results in attr imputation batch ")
    logger.info(resbatch)
    return [resbatch,loess_season_master,input_df_dynamic_master,ope_xpi_dynamic_master,res_df_master]

def get_attr_summary(input_df: pd.DataFrame, attrlist: list, input_filt:pd.Series=True,
                     size_validated:float =0.075, useshare:bool =True, params:dict =None):
    """
    Function to get attribute importance summary
    :param pd.DataFrame input_df: input attributes dataframe
    :param list attrlist: list of selected attributes
    :param pd.Series input_filt: selected indexes for attributes to filter
    :param float size_validated: float to filter out specific share amount
    :param bool useshare:boolean to use share or not in the attribute importance batch
    :param dict params: parameters dictionary
    :return: list[pd.DataFrame] [score_table,batch_attr_imp]: resulting scoretable & attr importance for batch
    """
    logger.info("\n Getting Attributes Importance Summary \n")
    weight_arr = list(params['default_vars']['attrsumm_wt']) #default [10, 3, 1, 2, 2, 2, 0.75])
    weight = np.array(weight_arr)
    # Get attribute importance batch results
    batch_attr_imp_list = attr_imp_batch(input_df = input_df, attr_list = attrlist,
                        input_filt = input_filt, params=params, useshare=useshare)
    batch_attr_imp = batch_attr_imp_list[0]
    ope_xpi_dynamic_master = batch_attr_imp_list[3]
    input_df_dynamic_master =  batch_attr_imp_list[2]
    loess_season_master = batch_attr_imp_list[1]
    res_df_master = batch_attr_imp_list[4]
    
    #batch_attr_imp = attr_imp_batch(input_df = input_df, attr_list = attrlist,
    #                    input_filt = input_filt, params=params, useshare=useshare)
    
    #return batch_attr_imp
    if batch_attr_imp is None:
        return [[None] , ope_xpi_dynamic_master, input_df_dynamic_master , loess_season_master , res_df_master]
    if len(batch_attr_imp) == 0:
        return [[None] , ope_xpi_dynamic_master, input_df_dynamic_master , loess_season_master , res_df_master]

    share = batch_attr_imp['ProdShare']
    elasticity = batch_attr_imp['elasticity']
    pricelevels = batch_attr_imp['price_gap']
    trend = batch_attr_imp['trend']
    season = batch_attr_imp['season']
    btch_ratio_price = batch_attr_imp['ratio_price']
    btch_ratio_price_min = batch_attr_imp['ratio_price_min']
    btch_ratio_dist = batch_attr_imp['ratio_dist']
    btch_ratio_dist_min = batch_attr_imp['ratio_dist_min']

    filtered_size_share = share > size_validated  # size_validated
    pricevolume = np.exp(pd.concat([2 * np.log(batch_attr_imp['ratio_price']),
                                    np.log(batch_attr_imp['ratio_price_min'])], axis=1).mean(axis=1))
    distvolume = np.exp(pd.concat([2 * np.log(batch_attr_imp['ratio_dist']),
                                   np.log(batch_attr_imp['ratio_dist_min'])], axis=1).mean(axis=1))
    distvolume[~np.isfinite(distvolume)] = 0
    score = pd.concat([pricevolume, distvolume, elasticity, pricelevels, trend, season, share], axis=1)
    score.columns = ['pricevolume', 'distvolume', 'elasticity', 'pricelevels', 'trend', 'season', 'share']

    if filtered_size_share.sum() <= 1:
        logger.info("WARNING: Index not possible, not enough attributes with sufficient share based on size validated")
        for col in score.columns:
            score[col] = 100
        return [[score],ope_xpi_dynamic_master, input_df_dynamic_master , loess_season_master , res_df_master]
    
    score_norm =  score.apply(lambda x: (x-x.mean())/x.std(), axis=0)
    score_norm = score_norm.multiply(100).round(0).astype(float) #changed from int to float
    score_index = score_norm.apply(lambda x: np.sum(weight * x) / np.sum(weight), axis=1)
    score_index = score_index.round(0)
    score_index[~filtered_size_share] = -999
    score_index_sorted = score_index.sort_values(ascending=False)
    score_table = pd.concat([score_norm,btch_ratio_price, btch_ratio_price_min, btch_ratio_dist, btch_ratio_dist_min, score_index], axis=1).reindex(score_index_sorted.index)
    score_table.columns = ['pricevolume', 'distvolume', 'elasticity', 'price_gap', 'trend', 'season', 'share','ratio_price', 'ratio_price_min', 'ratio_dist', 'ratio_dist_min','TOT_INDEX']
    score_table = score_table.fillna(0)
    if len(filtered_size_share[filtered_size_share == False]) > 0:
        logger.info("WARNING Attributes with insufficient share: %s", batch_attr_imp[~filtered_size_share].index)

    return [[score_table, batch_attr_imp] , ope_xpi_dynamic_master, input_df_dynamic_master , loess_season_master , res_df_master]


def level_weight(interim_output: pd.DataFrame, winner:str, input_df:pd.DataFrame, params:dict= None):
    """
    Function to calculate the weighted ratio in a specific level
    :param pd.DataFrame interim_output: attribute summary all attributes results
    :param str winner: current selected level winner
    :param pd.DataFrame input_df: attribute input dataframe
    :param dict params: parameters dictionary
    :return: float weighted_ratio: average weighted ratio
    """
    logger.info("\n Calculating level weight for price structure  \n")
    lvl_wt_arr = list(params['default_vars']['lvl_wt'])
    if winner in input_df.columns and winner in interim_output.index:
        ratio_pd = [interim_output.loc[winner,'ratio_price'], interim_output.loc[winner,'ratio_dist']]
    else:
        new_winner = input_df[winner].drop_duplicates()
        logger.info("INFO: no winner in index new winner is: %s", new_winner)
        interim_winner = interim_output[interim_output.index.isin(list(new_winner))]
        interim_winner = interim_winner[interim_winner == True]
        subset_interim_output = interim_output.loc[interim_winner.index,]
        ratio_pd = [np.mean(subset_interim_output['ratio_price']), np.mean(subset_interim_output['ratio_dist'])]
    ratio_pd = [1.2 if x < 1.2 else x for x in ratio_pd]
    weighted_ratio = np.average(ratio_pd, weights = lvl_wt_arr)
    weighted_ratio = weighted_ratio.round(2)
    return weighted_ratio


def create_binary(input_df: pd.DataFrame,attrlist: list, filter_index: pd.DataFrame, top_n: int=3, params: dict=None) -> list:
    """
    Function to create binary values for selected attributes list
    :param pd.DataFrame input_df: attribute input dataframe
    :param list attrlist: list of selected attributes
    :param pd.DataFrame filter_index: series with filtered index
    :param int top_n: number of attributes to select based on values
    :param dict params: input dictionary
    :return: list result: list of attributes, updated input dataframe and map
    """
    logger.info("\n Creating binary values \n")
    vol_col = params['default_vars']['volume']
    attrlist = np.array(attrlist)
    filter_df = input_df.loc[filter_index.index, attrlist]
    filter1 = filter_df.apply(lambda x: x.nunique() == 1, axis=0)
    filter2 = filter_df.apply(lambda x: x.nunique() > 2, axis=0)
    mask = (filter1 != True) & (filter2 != True)
    attrlistx = attrlist[mask].tolist()
    df_lists = {}
    col_names = []
    for att in attrlist[filter2 == True]:
        grouped_df = input_df.loc[filter_index.index, :].groupby(att).agg({
            vol_col: 'sum'
        })
        grouped_df = grouped_df.sort_values(vol_col, ascending=False)
        big_a = list(grouped_df.index[:top_n])
        df_lists[att] = list(big_a)
        attrlistx.extend(list(big_a))

        for b in big_a:
            input_df.loc[:, b] = np.where(input_df[att] == b, b, "not" + "_" + str(b))
        col_names.append(att)

    att_map = pd.DataFrame(df_lists)
    att_map.columns = col_names
    result = [list(dict.fromkeys(attrlistx)), input_df, att_map]
    return result


def ps_break(input_df: pd.DataFrame, attrlist:List,filter_index: pd.Series, purchase_structure: pd.DataFrame, current_level: int, params:Dict) -> List:
    """
    Function to create the price structure dataframe
    :param pd.DataFrame input_df: attribute input dataframe
    :param list attrlist: list of selected attributes
    :param pd.Series filter_index: series containing filtered index
    :param pd.DataFrame purchase_structure: price structure dataframe containing weights
    :param int current_level: current level in decision tree
    :param dict params: params ditionary
    :return: list(pd.DataFrame, pd.DataFrame) purchase_structure, interim: price structure & interim results dataframe
    """
    logger.info("\n In price structure break \n")
    unq_ps_value = purchase_structure.iloc[:, 0:current_level+1].loc[filter_index.index, :].drop_duplicates()
    unq_ps_level = len(unq_ps_value)
 
    if unq_ps_level > 1:
        logger.info("\n not a unique purchase_structure level\n")
        logger.info(unq_ps_value)
        return [purchase_structure, "not a unique purchase_structure level"]

    if current_level == 0:  # starts at level zero in python (for loop)
        purchase_structure.loc[filter_index.index, "W" + str(current_level + 1)] = 1

    current_level = current_level + 1
    logger.info("curr level is: %d ", current_level + 1)
    fx_tflag = filter_index 
    bin_output = create_binary(input_df=input_df, attrlist = attrlist, filter_index=fx_tflag, top_n = 3, params=params)
    attrlistx = bin_output[0]
    input_df = bin_output[1]
    attr_map = bin_output[2]
    input_filt_df = input_df.loc[filter_index.index, :]
    filt_flag_index = filter_index 
    
    logger.info("potential attributes are: %s", attr_map)
    logger.info("no of rows post filter passed to Attrsummary: %s", len(filt_flag_index))
    attr_summ_list = get_attr_summary(input_df = input_filt_df, attrlist=attrlistx,
                                 input_filt=filt_flag_index, size_validated=0.075,
                                 useshare=True, params=params)
    print(len(attr_summ_list))
    print(attr_summ_list)
    
    attr_summ = attr_summ_list[0]
    ope_xpi_dynamic_master = attr_summ_list[1]
    input_df_dynamic_master = attr_summ_list[2]
    loess_season_master = attr_summ_list[3]
    res_df_master = attr_summ_list[4]
    
    #attr_summ = get_attr_summary(input_df = input_filt_df, attrlist=attrlistx,
    #                             input_filt=filt_flag_index, size_validated=0.075,
    #                             useshare=True, params=params)
    
    if attr_summ is None:
        logger.info("No valid attributes for this break")
        return [[purchase_structure, "No valid attributes for this break"], ope_xpi_dynamic_master, input_df_dynamic_master , loess_season_master , res_df_master]

    if len(attr_summ) == 1:
        logger.info("No valid attributes for this break")
        return [[purchase_structure, "No valid attributes for this break"], ope_xpi_dynamic_master, input_df_dynamic_master , loess_season_master , res_df_master]

    # Select top 2 candidates
    if len(attr_map) > 0:
        top2_candidate = list(attr_summ[0].index)[:2]
        attr_map = pd.DataFrame(attr_map)
        attr_chk = attr_map.apply(lambda x: (x.isin(top2_candidate)).sum() == len(top2_candidate), axis=0)
        num_trues = np.array(attr_chk).sum()

        if num_trues > 0:
            new_attr = top2_candidate[0] + "_" + top2_candidate[1]
            attr_chk = attr_chk[attr_chk == True]
            attr_chk_name = list(attr_chk.index)
            logger.info("SUCCESS: top 2_candidate is: %s", top2_candidate)
            logger.info("INFO: attribute to check: %s", str(attr_chk_name))

            for name in attr_chk_name:
                logger.info("INFO: This is the atttr_chk_name: %s", name)
                input_df[new_attr] = np.where(input_df[name].isin(top2_candidate), input_df[name], "Not_" + new_attr)
            winner = new_attr
            logger.info("INFO: winner is: %s", winner)
            logger.info("INFO: attr_sum is: %s", input_df[name])
        else:
            winner = attr_summ[0].index[0]
            logger.info("INFO: no winner in num_trues, new winner is %s ", winner)
            logger.info("INFO: attr_sum is: %s ", attr_summ)
    else:
        winner = attr_summ[0].index[0]
        logger.info("INFO: no winner in map, new winner is %s ", winner)
        logger.info("INFO: attr_sum is: %s ", attr_summ)

    if len(winner) > 0:
        if winner in input_df.columns:
            purchase_structure.iloc[filter_index.index, current_level] = input_df.loc[filter_index.index, winner]
            purchase_structure = purchase_structure.fillna("-")
        else:
            logger.info("INFO: no further break possible for current level: %d ", current_level)
            purchase_structure.iloc[filter_index.index, current_level] = "**"

        weight = level_weight(interim_output=attr_summ[1], winner=winner, input_df = input_df, params = params)
        purchase_structure.loc[filter_index.index, "W" + str(current_level + 1)] = weight

    interim_obj3 = pd.DataFrame(attr_summ[0]) 
    winner_df = pd.DataFrame([winner for i in range(len(interim_obj3))]).set_index(interim_obj3.index)
    winner_df.columns = ['winner']
    L_df = pd.DataFrame([current_level + 1 for i in range(len(interim_obj3))]).set_index(interim_obj3.index)
    L_df.columns = ['L']
    interim_scoretable = pd.concat([L_df, winner_df, interim_obj3], axis=1)
    interim_scoretable['attributes'] = interim_scoretable.index
   
    return [[purchase_structure, interim_scoretable] ,  ope_xpi_dynamic_master, input_df_dynamic_master , loess_season_master , res_df_master]


def _concat(a) -> str:
    """
    Function to concat values in array
    :param a: array values to concat
    :return: str ca concated string
    """
    ca = a[0]
    if len(a) > 1:
        for i in a[1:]:
            ca = ca + "-" + i
    return ca

def create_structure(attr_df: pd.DataFrame, attrlist:list, filt_idx: pd.Series=[True],
                     nlevels:int = 25, params:dict = None) -> list:
    """
    Function to create purchase structure for the attributes and attribut list
    :param pd.DataFrame attr_df: input attr dataframe
    :param list attrlist: list of selected attributes
    :param pd.Series filt_idx: series containing filtered index
    :param int nlevels: number of levels
    :param dict params: dictionary of parameters
    :return list [purchase_structure, interim_result]:
    """
    logger.info("\n Starting create structure \n")
    thresh = params['default_vars']['create_struct_thresh']
    vol_col = params['default_vars']['volume']
    # Create dummy result df
    fixed_cols = ["X" + str(i + 1) for i in range(nlevels)]
    purchase_struct_df = pd.DataFrame(data="-", columns=fixed_cols, index=attr_df.index)
    purchase_struct_df.iloc[:, 0] = "Total"  # first column is Total
    if len(filt_idx) <= 1:
        filt_idx = pd.Series([True for i in range(len(attr_df))])

    interim_res = pd.DataFrame()
    ope_xpi_dynamic_master = pd.DataFrame()
    input_df_dynamic_master =  pd.DataFrame()
    loess_season_master = pd.DataFrame()
    res_df_master =  pd.DataFrame()
    
    for curr_lvl in range(nlevels - 1):  # Python is not inclusive
        phrase = purchase_struct_df.iloc[:, 0:curr_lvl + 1].apply(lambda x: _concat(x), axis=1).to_frame()
        # check whether the share of level are above the threshold
        phrase.loc[phrase.index, "lowshare_filt"] = attr_df.loc[phrase.index, vol_col].fillna(0).sum() / \
                                                    attr_df[vol_col].fillna(0).sum()
        phrase.loc[phrase.index, "lowshare_filt_filter"] = phrase.loc[phrase.index, "lowshare_filt"] <= thresh
        phrase = phrase[phrase["lowshare_filt_filter"] != True]
        phrase = phrase.drop(columns=["lowshare_filt", "lowshare_filt_filter"])

        unique_phrases = phrase.drop_duplicates().values

        logger.info("\n In create structure loop\n")
        logger.info("INFO: level number: %d ", curr_lvl+1)

        dashes_sum = (purchase_struct_df.iloc[:, curr_lvl - 1] == "-").sum()
        if curr_lvl >= 2:
            print(dashes_sum) 
            print(len(purchase_struct_df))

        if curr_lvl >= 2 and dashes_sum == len(purchase_struct_df):
            logger.info("Only running once")
            return [purchase_struct_df, interim_res]

        for unique_phrase in unique_phrases:
            logger.info("INFO: Unique phrase is: %s", unique_phrase[0])
            val = unique_phrase[0]            
            sltd_filt_idx = (phrase == val)[0]
            sltd_filt_idx = sltd_filt_idx[sltd_filt_idx == True]
            logger.info("length of attribute dataframe passed to ps_break is: %d", len(attr_df.loc[sltd_filt_idx.index,]))
            attr_filter = attr_df.loc[sltd_filt_idx.index, attrlist]
            filter1 = attr_filter.apply(lambda x: x.nunique() > 1, axis=0)
            mask = (filter1 == True)
            attrlistx = attrlist[mask]
            # Get the unique list
            if len(attrlistx) > 0:
                struc = ps_break(input_df=attr_df, attrlist=attrlistx,filter_index=sltd_filt_idx, purchase_structure=purchase_struct_df,current_level=curr_lvl, params=params)
                
                if len(struc[0]) > 1:
                    purchase_struct_df = struc[0][0]
                    interim = struc[0][1]
                    res_df = struc[4]
                    res_df['phrase'] = val
                    res_df['level'] = curr_lvl
                    res_df_master = res_df_master.append(res_df)
                    
                    ope_xpi_dynamic = struc[1]
                    ope_xpi_dynamic['phrase'] = val
                    ope_xpi_dynamic['level'] = curr_lvl
                    ope_xpi_dynamic_master = ope_xpi_dynamic_master.append(ope_xpi_dynamic)
                    
                    input_df_dynamic = struc[2]
                    input_df_dynamic['phrase'] = val
                    input_df_dynamic['level'] = curr_lvl
                    input_df_dynamic_master = input_df_dynamic_master.append(input_df_dynamic)
                    
                    loess_season = struc[3]
                    loess_season['phrase'] = val
                    loess_season['level'] = curr_lvl
                    loess_season_master = loess_season_master.append(loess_season)
                    
                    if isinstance(interim, pd.DataFrame):
                        interim.loc[:, "_id"] = str(unique_phrase[0])
                        interim_res = interim_res.append(interim)
                else:
                    purchase_struct_df = struc
            else:
                logger.info("WARNING: No attributes to test")

    interim_result = interim_res#pd.concat(interim_res, ignore_index=True)
    return [purchase_struct_df, interim_result , ope_xpi_dynamic_master, input_df_dynamic_master , loess_season_master , res_df_master]

def ps_break_between(input_df: pd.DataFrame, attrlist:List,filter_index: pd.Series, purchase_structure: pd.DataFrame, current_level: int, params:Dict) -> List:
    """
    Function to create the price structure between dataframe
    :param pd.DataFrame input_df: attribute input dataframe
    :param list attrlist: list of selected attributes
    :param pd.Series filter_index: series containing filtered index
    :param pd.DataFrame purchase_structure: price structure dataframe containing weights
    :param int current_level: current level in decision tree
    :param dict params: params ditionary
    :return: list(pd.DataFrame, pd.DataFrame) purchase_structure, interim: price structure & interim results dataframe
    """
    logger.info("\n In price structure break \n")
    unq_ps_value = purchase_structure.iloc[:, 0:current_level+1].loc[filter_index.index, :].drop_duplicates()
    unq_ps_level = len(unq_ps_value)
 
    if unq_ps_level > 1:
        logger.info("\n not a unique purchase_structure level\n")
        logger.info(unq_ps_value)
        return [purchase_structure, "not a unique purchase_structure level"]

    if current_level == 0:  # starts at level zero in python (for loop)
        purchase_structure.loc[filter_index.index, "W" + str(current_level + 1)] = 1
    
    #not incremented by 1
    current_level = current_level 
    logger.info("curr level is: %d ", current_level + 1)
    fx_tflag = filter_index 
    bin_output = create_binary(input_df=input_df, attrlist = attrlist, filter_index=fx_tflag, top_n = 3, params=params)
    attrlistx = bin_output[0]
    input_df = bin_output[1]
    attr_map = bin_output[2]
    input_filt_df = input_df.loc[filter_index.index, :]
    filt_flag_index = filter_index 
    
    logger.info("potential attributes are: %s", attr_map)
    logger.info("no of rows post filter passed to Attrsummary: %s", len(filt_flag_index))
    attr_summ = get_attr_summary(input_df = input_filt_df, attrlist=attrlistx,
                                 input_filt=filt_flag_index, size_validated=0.075,
                                 useshare=True, params=params)
    if attr_summ is None:
        logger.info("No valid attributes for this break")
        return [purchase_structure, "No valid attributes for this break"]

    if len(attr_summ) == 1:
        logger.info("No valid attributes for this break")
        return [purchase_structure, "No valid attributes for this break"]

    # Select top 2 candidates
    if len(attr_map) > 0:
        top2_candidate = list(attr_summ[0].index)[:2]
        attr_map = pd.DataFrame(attr_map)
        attr_chk = attr_map.apply(lambda x: (x.isin(top2_candidate)).sum() == len(top2_candidate), axis=0)
        num_trues = np.array(attr_chk).sum()

        if num_trues > 0:
            new_attr = top2_candidate[0] + "_" + top2_candidate[1]
            attr_chk = attr_chk[attr_chk == True]
            attr_chk_name = list(attr_chk.index)
            logger.info("SUCCESS: top 2_candidate is: %s", top2_candidate)
            logger.info("INFO: attribute to check: %s", str(attr_chk_name))

            for name in attr_chk_name:
                logger.info("INFO: This is the atttr_chk_name: %s", name)
                input_df[new_attr] = np.where(input_df[name].isin(top2_candidate), input_df[name], "Not_" + new_attr)
            winner = new_attr
            logger.info("INFO: winner is: %s", winner)
            logger.info("INFO: attr_sum is: %s", input_df[name])
        else:
            winner = attr_summ[0].index[0]
            logger.info("INFO: no winner in num_trues, new winner is %s ", winner)
            logger.info("INFO: attr_sum is: %s ", attr_summ)
    else:
        winner = attr_summ[0].index[0]
        logger.info("INFO: no winner in map, new winner is %s ", winner)
        logger.info("INFO: attr_sum is: %s ", attr_summ)

    if len(winner) > 0:
        if winner in input_df.columns:
            purchase_structure.iloc[filter_index.index, current_level] = input_df.loc[filter_index.index, winner]
            purchase_structure = purchase_structure.fillna("-")
        else:
            logger.info("INFO: no further break possible for current level: %d ", current_level)
            purchase_structure.iloc[filter_index.index, current_level] = "**"

        weight = level_weight(interim_output=attr_summ[1], winner=winner, input_df = input_df, params = params)
        purchase_structure.loc[filter_index.index, "W" + str(current_level + 1)] = weight

    interim_obj3 = pd.DataFrame(attr_summ[0]) 
    winner_df = pd.DataFrame([winner for i in range(len(interim_obj3))]).set_index(interim_obj3.index)
    winner_df.columns = ['winner']
    L_df = pd.DataFrame([current_level + 1 for i in range(len(interim_obj3))]).set_index(interim_obj3.index)
    L_df.columns = ['L']
    interim_scoretable = pd.concat([L_df, winner_df, interim_obj3], axis=1)
    interim_scoretable['attributes'] = interim_scoretable.index
   
    return [purchase_structure, interim_scoretable]



def create_structure_between(attr_df: pd.DataFrame, purchase_struct_df: pd.DataFrame, interim_result: pd.DataFrame,attrlist:list, filt_idx: pd.Series=[True],lvl:int=0,
                     nlevels:int = 25, params:dict = None) -> list:
    
    """
    Function to force create purchase structure between two attributes from atributes list
    :param pd.DataFrame attr_df: input attr dataframe
    :param pd.DataFrame purchase_struct_df: input price structure dataframe to be force
    :param pd.DataFrame interim_result: interim result containing list of winners
    :param list attrlist: list of selected attributes to force between
    :param pd.Series filt_idx: series containing filtered index
    :param int lvl: number of level to create new structure from
    :param int nlevels: number of levels
    :param dict params: dictionary of parameters
    :return list [purchase_structure, interim_result]:
    """
    logger.info("\n Starting create structure \n")
    thresh = params['default_vars']['create_struct_thresh']
    vol_col = params['default_vars']['volume']
    master_interim_result=interim_result.copy()
    if len(filt_idx) <= 1:
        filt_idx = pd.Series([True for i in range(len(attr_df))])

    int_res = []
    assert len(attr_df) == len(purchase_struct_df) # ensure data frame and price structure are the same
    lvl = lvl # subtract to move index up
    f = pd.Series([ True for i in range(len(attr_df))])
    if purchase_struct_df is None or lvl <= 0: 
        # Create dummy result df
        fixed_cols = ["X" + str(i + 1) for i in range(nlevels)]
        purchase_struct_df= pd.DataFrame(data="-", columns=fixed_cols, index=attr_df.index)
        purchase_struct_df.iloc[:, 0] = "Total"  # first column is Total

    else:
        last_value =int(list(purchase_struct_df.columns[-1])[-1]) # get the last value and subtract our current value from it
        purchase_struct_df.iloc[filt_idx,lvl:last_value] = "-" # update everything in current level and below        
        weight_cols = ["W" + str(i+1) for i in range(lvl, nlevels)] # create clean column
        purchase_struct_df.loc[filt_idx, weight_cols] = "-"

        # Get new weight for current level
        filt_lvl = lvl
        interim_result =interim_result[interim_result['L'] == filt_lvl]

        id_filter=''
        for i in  range(1,filt_lvl):

            val = purchase_struct_df[filt_idx]['X'+str(i)].unique()[0]
            id_filter=id_filter + "-" + val
            id_filter=id_filter.lstrip("-")

        temp_result = interim_result[interim_result['L']==filt_lvl][interim_result['_id']==id_filter]
        forced_vals = purchase_struct_df[filt_idx]["X"+str(filt_lvl)].unique().tolist()
        forced_column_name = attr_df.columns[attr_df.isin(forced_vals).any()][0]
        print(forced_column_name)
        temp_result[temp_result['attributes'] == forced_column_name]

        logger.info("Here is the forced attribute %s",forced_column_name)

        temp_result = temp_result.set_index('attributes')
        new_weight = level_weight(interim_output = temp_result, winner=forced_column_name, input_df=attr_df, params= params)

        #update weight for forced attribute in new PS
        purchase_struct_df.loc[filt_idx,"W"+str(lvl)] = new_weight

    for curr_lvl in range(lvl,nlevels):  # Python is not inclusive

        phrase = purchase_struct_df.iloc[:, 0:curr_lvl + 1].apply(lambda x: _concat(x), axis=1).to_frame()
        # check whether the share of level are above the threshold
        phrase.loc[phrase.index, "lowshare_filt"] = attr_df.loc[phrase.index, vol_col].fillna(0).sum() / \
                                                    attr_df[vol_col].fillna(0).sum()
        phrase.loc[phrase.index, "lowshare_filt_filter"] = phrase.loc[phrase.index, "lowshare_filt"] <= thresh
        phrase = phrase[phrase["lowshare_filt_filter"] != True]
        phrase = phrase.drop(columns=["lowshare_filt", "lowshare_filt_filter"])
        unique_phrases = phrase[filt_idx].drop_duplicates().values
        logger.info("\n In create structure loop\n")
        logger.info("INFO: level number: %d ", curr_lvl + 1)

        dashes_sum = (purchase_struct_df.iloc[:, curr_lvl - 1] == "-").sum()

        for unique_phrase in unique_phrases:
            logger.info("INFO: Unique phrase is: %s", unique_phrase[0])
            val = unique_phrase[0]
            sltd_filt_idx = (phrase == val)[0]
            sltd_filt_idx = sltd_filt_idx[sltd_filt_idx == True]
            logger.info("length of attribute dataframe passed to ps_break is: %d", len(attr_df.loc[sltd_filt_idx.index,]))
            attr_filter = attr_df.loc[sltd_filt_idx.index, attrlist]
            filter1 = attr_filter.apply(lambda x: x.nunique() > 1, axis=0)
            mask = (filter1 == True)
            attrlistx = attrlist[mask]
            # Get the unique list
            if len(attrlistx) > 0:
                struc = ps_break_between(input_df=attr_df, attrlist=attrlistx,filter_index=sltd_filt_idx, purchase_structure=purchase_struct_df,current_level=curr_lvl, params=params)
                if len(struc) > 1:
                    purchase_struct_df = struc[0]
                    interim = struc[1]
                    print("INTERIM")
                    print(interim)
                    if isinstance(interim, pd.DataFrame):
                        interim.loc[:, "_id"] = str(unique_phrase[0])
                        master_interim_result = master_interim_result.append(interim)
                else:
                    purchase_struct_df = struc
            else:
                logger.info("WARNING: No attributes to test")

    return [purchase_struct_df,master_interim_result]


def create_structure_unmapped_data (unmapped_data_for_ps:pd.DataFrame,purchase_struct: pd.DataFrame, nlevels:int ) ->pd.DataFrame:
    """
    Function to create purchase structure for unmapped data
    :param pd.DataFrame unmapped_data_for_ps: input attr unmapped data dataframe
    :param pd.DataFrame purchase_struct_df: input purchase structure dataframe to be used as referece for mapping
    :param int nlevels: number of levels in the PS
    :return pd.DataFrame ps_unmapped: purhcase structure for unmapped data
    """

    #create empty dataframe for storing PS unmapped results . Shape (rows same as unmapped data , colums same as purchase strucutre from previous step)
    ps_unmapped =pd.DataFrame(np.full((unmapped_data_for_ps.shape[0],2*len([i for i in purchase_struct.columns if i.startswith("X")])), "-"), 
                              columns=[i for i in purchase_struct.columns if i.startswith("X")] + [i for i in purchase_struct.columns if i.startswith("W")])

    for curr_lvl in range(1,nlevels+1):
        next_level = curr_lvl+1

        if curr_lvl ==1:
            ps_unmapped["X"+str(curr_lvl)] = "Total"
            ps_unmapped["W"+str(curr_lvl)] = 1

        else:
            selected_structure = purchase_struct[["X"+str(i) for i in range(1,curr_lvl+1)]+["W"+str(i) for i in range(1,curr_lvl+1)]].drop_duplicates()
            selected_structure['phrase'] = selected_structure[["X"+str(i) for i in range(1,curr_lvl)] ].apply(lambda x: _concat(x), axis=1)
            unique_phrases = selected_structure['phrase'].unique()
            ps_unmapped['phrase'] = ps_unmapped[["X"+str(i) for i in range(1,curr_lvl)] ].apply(lambda x: _concat(x), axis=1)

            for iter_phrase in unique_phrases:
                
                #filter rows to naviagte through the tree
                filter_rows = ps_unmapped['phrase'] == iter_phrase

                #filter the purhcase strucutre to match the phrase to get the real attributes that were split
                sub_ps = selected_structure [selected_structure.phrase == iter_phrase]
                
                #find attribute and weight values used in the split for the level in a given branch of tree 
                break_vals = sub_ps["X" + str(curr_lvl)].unique().tolist()
                weight_vals = sub_ps["W" + str(curr_lvl)].unique().tolist()
                
                #dont consider break values that have not or Not as they are generally not present in the data and are created internally by the create binary function 
                break_vals=[i for i in [i for i in break_vals if not i.startswith("not")] if not i.startswith("Not") ] 
                
                #check if there is a valid break value or not . if it is "-" indicates tree doesnt split further
                if break_vals[0] !="-":


                    #find attribute name column that has break_vals
                    attr_name =unmapped_data_for_ps.eq(str(break_vals[0])).any().idxmax()
                    
                    #refine break values list to only have values that are present in the data, everything else get clasiified as not_xxxx
                    break_vals = [i for i in break_vals if i in  unmapped_data_for_ps[attr_name].unique()]
                    break_vals.sort()

                    #define value for others 
                    value_for_other = "not_"+"_".join([str(item) for item in break_vals])
                    if value_for_other == "not_-":
                        value_for_other = "-"

                    #update the structure
                    ps_unmapped.loc[filter_rows,'X'+str(curr_lvl)] = np.where(unmapped_data_for_ps[filter_rows][attr_name].isin(break_vals),
                                                                           unmapped_data_for_ps[filter_rows][attr_name],
                                                                           value_for_other).tolist()
                    #update weights
                    ps_unmapped.loc[filter_rows,'W'+str(curr_lvl)] = weight_vals[0]
                    
                else:
                    ps_unmapped.loc[filter_rows,'X'+str(curr_lvl)] = "-"
                    ps_unmapped.loc[filter_rows,'W'+str(curr_lvl)] = "-"
    
    ps_unmapped=ps_unmapped.drop(columns=['phrase'],axis=1)

    return(ps_unmapped)
  