# -*- coding: utf-8 -*-

from typing import Dict

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from pandas import DataFrame
from pygam import LinearGAM, f, l, s
from pyspark.sql.types import *
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

from src.utils.logger import logger
from src.utils.utils import *
from src.utils.version_control import *


def create_month_id(data: DataFrame, params: Dict):
    """
    Function to create month identifier from data
    Args:
        data:data for loess as pandas dataframe
        params: parameter dictionary
    Returns:
        data: Input dataframe with month column added
    """

    column_with_months = params["ewb"]["data_management"]["column_with_date"]
    data[column_with_months] = pd.to_datetime(data[column_with_months])
    data["month_id"] = data[column_with_months].dt.month

    return data


def loess_regression_models(data_for_loess_sub: DataFrame, params: Dict) -> DataFrame:
    """
    Function to develop regression model and pass the output to mixed models
    Args:
        data_for_loess_sub:data for loess as pandas dataframe
        params: parameter dictionary
    Returns:
        data: data for loess as pandas dataframe with regression outputs added
    """

    data_for_loess_sub = create_month_id(data_for_loess_sub, params)

    if len(data_for_loess_sub) > 10:
        volume_term = load_func(params["ewb"]["modeling"]["common"]["target_var"])(
            params["ewb"]["data_management"]["volume"]
        )
        # price_term = params["ewb"]["modeling"]["common"]["price_term"]
        # acv_term = params["ewb"]["modeling"]["loess"]["acv_term"]
        seasonality_term = params["ewb"]["modeling"]["common"]["seasonality_term"]
        independent_features = params["ewb"]["modeling"]["loess"]["independent_features_linear"]
        intercept = params["ewb"]["modeling"]["loess"]["fit_intercept"]
        independent_features_non_seasonal = [
            i for i in independent_features if i not in [seasonality_term]
        ]

        if seasonality_term in independent_features:
            seasonal_dummies = pd.get_dummies(data_for_loess_sub["month_id"])
        #             print("seasonal values exist and dummies are computed")
        else:
            seasonal_dummies = pd.DataFrame()
        trend_term = params["ewb"]["data_management"]["granularity"] + "_number"
        data_for_loess_sub["trend_variable"] = data_for_loess_sub[[trend_term]]

        indep_data = pd.concat(
            [
                data_for_loess_sub[independent_features_non_seasonal + ["trend_variable"]],
                seasonal_dummies,
            ],
            axis=1,
        )

        X = indep_data.values
        y = data_for_loess_sub[volume_term].values

        regr = LinearRegression(fit_intercept=intercept)
        regr.fit(X, y)
        # print(i,j,"reg score -",regr.score(X, y))

        y_pred = regr.predict(X)

        data_for_loess_sub["error"] = data_for_loess_sub[volume_term] - y_pred

        coeff_df = pd.DataFrame(
            {
                "features": indep_data.columns,
                "coeffs": regr.coef_,
            }
        )

        coeff_linear = coeff_df[coeff_df.features == "trend_variable"]["coeffs"].values.tolist()[0]
        data_for_loess_sub["linear_term"] = data_for_loess_sub["trend_variable"] * coeff_linear

    return data_for_loess_sub


def loess_gam_models(data_for_loess_sub: DataFrame, params: Dict) -> list:
    """
    Function to gam model to estimate trend term
    Args:
        data_for_loess_sub:data for loess as pandas dataframe
        params: parameter dictionary
    Returns:
        data: Gam prediction for every input record
    """

    time_period = params["ewb"]["data_management"]["granularity"] + "_"
    target_gam = params["ewb"]["modeling"]["loess"]["target_gam"]
    intercept = params["ewb"]["modeling"]["loess"]["fit_intercept"]
    n_splines = params["ewb"]["modeling"]["loess"]["n_splines"]

    X = data_for_loess_sub[[time_period]].values
    y = data_for_loess_sub[target_gam].values

    gam = LinearGAM(s(0), fit_intercept=intercept)
    gam.gridsearch(X, y, lam=np.logspace(-3, 3, 10), n_splines=n_splines, progress=False)
    preds = gam.predict(X)

    return preds


def loess_custom_models(data_for_loess_sub: DataFrame, params: Dict) -> list:
    """
    Function to gam model to estimate trend term
    Args:
        data_for_loess_sub:data for loess as pandas dataframe
        params: parameter dictionary
    Returns:
        data: loess prediction for every input record
    """

    time_period = params["ewb"]["data_management"]["granularity"] + "_number"
    target_gam = params["ewb"]["modeling"]["loess"]["target_gam"]
    span = params["ewb"]["modeling"]["loess"]["span"]

    X = data_for_loess_sub[[time_period]].values
    y = data_for_loess_sub[target_gam].values

    x_data = [item for sublist in X.tolist() for item in sublist]
    y_data = [item for sublist in y.tolist() for item in sublist]
    preds = lowess(y_data, x_data, frac=span)

    return preds


def loess_prediction(data_for_loess: DataFrame, params: Dict) -> list:
    """
    Function to gam model to estimate trend term
    Args:
        data_for_loess:data for loess as pandas dataframe
        params: parameter dictionary
    Returns:
        data: Gam prediction for every input record
    """

    volume_term = load_func(params["ewb"]["modeling"]["common"]["target_var"])(
        params["ewb"]["data_management"]["volume"]
    )
    time_period = params["ewb"]["data_management"]["granularity"] + "_number"
    use_pygam = params["ewb"]["modeling"]["loess"]["use_pygam"]

    data_for_loess_sub = loess_regression_models(data_for_loess, params)

    # y = data_for_loess_sub[target_gam].values #changed in client testing

    preds_loess = loess_custom_models(data_for_loess_sub, params)
    preds_loess_final = pd.DataFrame(preds_loess)[1].tolist()
    if use_pygam:

        preds_pygam = loess_gam_models(data_for_loess_sub, params)

        dict_df = pd.DataFrame(
            {
                time_period: [
                    item for sublist in data_for_loess_sub[[time_period]].values for item in sublist
                ],
                "trend_pygam": preds_pygam,
                "linear_term": data_for_loess_sub["linear_term"].values,
            }
        )
        dict_df["trend_term"] = dict_df["trend_pygam"] + dict_df["linear_term"]
    else:
        dict_df = pd.DataFrame(
            {
                time_period: [
                    item for sublist in data_for_loess_sub[[time_period]].values for item in sublist
                ],
                "trend_loess": preds_loess_final,
                "linear_term": data_for_loess_sub["linear_term"].values,
            }
        )
        dict_df["trend_term"] = dict_df["trend_loess"] + dict_df["linear_term"]

    return dict_df


def run_loess(data_for_loess_raw: DataFrame, params: Dict, catalog: Dict) -> DataFrame:
    """
    Main function to run loess- calls all other functions
    Args:
        data_for_loess_raw: Data output from the data prep step
        params : Parameters dictionary
        catalog : Dictionary of files locations
    Returns:
        final_loess_df: dataframe with loess output
    """
    start = pd.Timestamp.now()

    level_1 = params["ewb"]["data_management"]["levels"]["lvl1"]
    level_2 = params["ewb"]["data_management"]["levels"]["lvl2"]
    level_3 = params["ewb"]["data_management"]["levels"]["lvl3"]
    use_loess_from_R = params["ewb"]["modeling"]["loess"]["use_loess_from_R"]["enabled"]
    loess_indep_vars = params["ewb"]["modeling"]["loess"]["independent_features_linear"]
    dep = load_func(params["ewb"]["modeling"]["common"]["target_var"])(
        params["ewb"]["data_management"]["volume"]
    )
    loess_degf = params["ewb"]["modeling"]["loess"]["use_loess_from_R"]["loess_degree"]

    time_period = params["ewb"]["data_management"]["granularity"] + "_number"

    granularity = params["ewb"]["data_management"]["granularity"]
    min_weeks_months_loess = params["ewb"]["modeling"]["loess"]["min_weeks_months_loess"]
    seasonality_term = params["ewb"]["modeling"]["common"]["seasonality_term"]

    grouping_cols = list(filter(None, [level_1] + [level_2] + [level_3]))

    # filter data with enough observations
    data_for_loess = data_for_loess_raw[
        data_for_loess_raw[f"{granularity}_count"] > min_weeks_months_loess
    ].reset_index(drop=True)

    if use_loess_from_R:
        col_for_loess = list(
            filter(None, [level_1, level_2, level_3, time_period] + loess_indep_vars + [dep])
        )
        data_for_loess_r = data_for_loess[col_for_loess]
        # Defining the R script and loading the instance in Python
        r = robjects.r
        r["source"](str(params["ewb"]["data_management"]["loess_R_path"]))
        # Loading the function we have defined in R.
        loess_function_r = robjects.globalenv["loess_prediction"]
        # converting it into r object for passing into r function
        with localconverter(robjects.default_converter + pandas2ri.converter):
            df_r = robjects.conversion.py2rpy(data_for_loess_r)
        # Invoking the R function and getting the result
        base_loess_r = loess_function_r(
            df_r, level_1, level_2, level_3, time_period, loess_degf, dep, loess_indep_vars
        )
        # Converting it back to a pandas dataframe.
        final_loess_df = pandas2ri.rpy2py(base_loess_r)
        final_loess_df = final_loess_df.merge(data_for_loess_raw, how="inner")

    else:
        if len(level_3) > 0:
            predicted_dataframe = (
                data_for_loess.groupby([level_1, level_2, level_3])
                .apply(
                    loess_prediction,
                    params=params,
                )
                .reset_index()
            )
            final_loess_df = data_for_loess_raw.merge(
                predicted_dataframe[[level_1, level_2, level_3, time_period, "trend_term"]],
                on=[level_1, level_2, level_3, time_period],
                how="left",
            )

        else:
            predicted_dataframe = (
                data_for_loess.groupby([level_1, level_2])
                .apply(
                    loess_prediction,
                    params=params,
                )
                .reset_index()
            )

            final_loess_df = data_for_loess_raw.merge(
                predicted_dataframe[[level_1, level_2, time_period, "trend_term"]],
                on=[level_1, level_2, time_period],
                how="left",
            )
    # save outputs
    write_obj(
        final_loess_df,
        catalog["loess_prediction"]["filepath"],
        catalog["loess_prediction"]["filename"],
        catalog["loess_prediction"]["format"],
    )

    duration = pd.Timestamp.now() - start
    logger.info(f"Trend term calculated in {duration}s")

    return final_loess_df
