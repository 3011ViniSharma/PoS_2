# -*- coding: utf-8 -*-

import sys
from itertools import chain
from typing import Tuple

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.window import Window

from src.features.additional_input import *
from src.utils.logger import logger
from src.utils.utils import *
from src.utils.version_control import *
from src.features.automate_holiday_flags import *
import os
from src.utils.utils import load_csv_data
from src.features.promo_flag_creation import *


def data_preparation(
    input_df: DataFrame,
    parameters: dict,
    holidays_config: dict,
    covid_flag_1_df: dict,
    covid_flag_2_df: dict,
    catalog: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    This function performs data preparation
    Args:
        input_df : spark dataframe
        parameters : parameter dict
        holidays_config : holidays and covid parameters
        covid_flag_1_df : daraframe with covid flags for phase 1
        covid_flag_2_df : daraframe with covid flags for phase 2
        catalog : dict with catalog of data files

    Returns:
        raw_loess_all : preprocessed spark dataframe
        raw_data_good : preprocessed spark dataframe with only model eligible rows
        raw_data_drop : preprocessed spark dataframe with only model ineligible rows
        unique_count_df : dataframe with unique pair combinations
    """
    start = pd.Timestamp.now()

    ppg_def = parameters["ewb"]["data_management"]["ppg_def"]
    lvl1 = parameters["ewb"]["data_management"]["levels"]["lvl1"]
    lvl2 = parameters["ewb"]["data_management"]["levels"]["lvl2"]
    lvl3 = parameters["ewb"]["data_management"]["levels"]["lvl3"]
    colnames_weighted_avg = parameters["ewb"]["data_management"]["colnames_weighted_avg"]
    colnames_to_sum = parameters["ewb"]["data_management"]["colnames_to_sum"]
    additional_raw_variables = parameters["ewb"]["data_management"]["additional_raw_variables"]
    promo_flag = parameters["ewb"]["data_management"]["promo_flag"]
    value = parameters["ewb"]["data_management"]["value"]
    volume = parameters["ewb"]["data_management"]["volume"]
    column_with_date = parameters["ewb"]["data_management"]["column_with_date"]
    last_week_month_cutoff = parameters["ewb"]["data_management"]["last_weeks_months_cut_off"]
    distrib_var = parameters["ewb"]["data_management"]["distrib_var"]
    price = parameters["ewb"]["data_management"]["price_col"]
    params_to_log_transform = parameters["ewb"]["data_management"]["params_to_log_transform"]
    cpi_crit = parameters["ewb"]["data_management"]["cpi_crit"]
    eval_peaks_years = parameters["ewb"]["data_management"]["eval_peaks_years"]
    peak_percentage_lmt = parameters["ewb"]["data_management"]["peak_percentage_lmt"]
    granularity = parameters["ewb"]["data_management"]["granularity"]
    percentile_pairs = parameters["ewb"]["data_management"]["percentile_pairs"]
    week_count_out = parameters["ewb"]["data_management"]["week_count_out"]
    month_count_out = parameters["ewb"]["data_management"]["month_count_out"]
    cv_price_out = parameters["ewb"]["data_management"]["cv_price_out"]
    volume_filter_pct = parameters["ewb"]["data_management"]["volume_filter_pct"]
    lvl1_to_remove = parameters["ewb"]["data_management"]["lvl1_to_remove"]
    lvl2_to_remove = parameters["ewb"]["data_management"]["lvl2_to_remove"]
    weeks_or_months_to_remove = parameters["ewb"]["data_management"]["weeks_or_months_to_remove"]
    additional_filter = parameters["ewb"]["data_management"]["additional_filter"]
    add_filter4_val = parameters["ewb"]["data_management"]["add_filter4_val"]
    cv_price_crit = parameters["ewb"]["data_management"]["cv_price_crit"]
    items = parameters["ewb"]["data_management"]["items"]
    acv_promos = parameters["ewb"]["data_management"]["acv_promos"]
    acv_vars_list = parameters["ewb"]["data_management"]["acv_vars_list"]
    covid_flag_1 = parameters["holidays_config"]["covid_flags"]["covid_flag_1"]
    covid_flag_2 = parameters["holidays_config"]["covid_flags"]["covid_flag_2"]
    start_covid_period = parameters["holidays_config"]["covid_flags"]["start_covid_period"]
    covid_3_exclud_val = parameters["holidays_config"]["covid_flags"]["covid_3_exclud_val"]
    covid_3_period_start = parameters["holidays_config"]["covid_flags"]["covid_3_period_start"]
    covid_3_period_end = parameters["holidays_config"]["covid_flags"]["covid_3_period_end"]
    covid_4_exclud_val = parameters["holidays_config"]["covid_flags"]["covid_4_exclud_val"]
    covid_4_period_start = parameters["holidays_config"]["covid_flags"]["covid_4_period_start"]
    covid_4_period_end = parameters["holidays_config"]["covid_flags"]["covid_4_period_end"]
    bottom5_flag = parameters["ewb"]["data_management"]["bottom5_flag"]
    #imputation_lvl_list = parameters["ewb"]["qc_report"]["imputation_lvl_list"]
    #imputation_lvl_list_for_RF = parameters["ewb"]["qc_report"]["imputation_lvl_list_for_RF"]
    loess_indep_vars = parameters["ewb"]["modeling"]["loess"]["independent_features_linear"]
    # fixed_effect_structured = parameters["ewb"]["qc_report"]["fixed_effect_structured"]
    # fixed_effect_unstructured = parameters["ewb"]["qc_report"]["fixed_effect_unstructured"]
    # random_effect_structured = parameters["ewb"]["qc_report"]["random_effect_structured"]
    # random_effect_unstructured = parameters["ewb"]["qc_report"]["random_effect_unstructured"]
    opi_grouping = parameters["ewb"]["data_management"]["opi_grouping"]
    volume_promo_var = parameters["ewb"]["data_management"]["volume_promo_var"]
    oot_pred = parameters["ewb"]["data_management"]["oot_pred"]
    # pantry_loading_dynamic = parameters["ewb"]["data_management"]["pantry_loading_dynamic"]
    pantry_flags = parameters["ewb"]["data_management"]["pantry_flags"]
    max_pantry_flags = parameters["ewb"]["data_management"]["max_pantry_flags"]
    manufacturer_col = parameters["ewb"]["data_management"]["manufacturer_col"]
    manufacturer_name = parameters["ewb"]["data_management"]["manufacturer_name"]
    use_additional_input = parameters["ewb"]["data_management"]["use_additional_input"]
    add_df_list = parameters["ewb"]["data_management"]["additional_datasets_list"]
    outlier_retailers = parameters["ewb"]["data_management"]["outlier_retailers"]
    if oot_pred:
        pantry_loading = load_func(parameters["ewb"]["data_management"]["pantry_loading"])(
            max_pantry_flags
        )
    else:
        pantry_loading = load_func(parameters["ewb"]["data_management"]["pantry_loading"])(
            pantry_flags
        )

    grouping_columns = list(filter(None, [lvl1] + [lvl2] + [lvl3]))

    logger.info("Starting data preparation process")

    # remove all duplicates
    input_df = input_df.dropDuplicates()

    # add ppg columns to raw data set : where are we adding ppg columns from. Add only when ppg_def != NULL ( not clear)
    if len(ppg_def) > 0:
        input_df = create_concatenated_column(input_df, ppg_def, "ppg", "_")

    # create a dataframe with count unique value for lvl1,2,3
    unique_count_df = create_unique_pair(input_df, grouping_columns, additional_raw_variables)

    # ADD QC steps for unique pairs
    colnames_to_drop = QC_unique_pair(
        unique_count_df,
        additional_raw_variables,
        loess_indep_vars,
        grouping_columns,
    )

    # if any column has duplicate values stop, update the parameter and rerun the pipeline
    try:
        if len(colnames_to_drop) != 0:
            raise Exception("Raising exception")
    except Exception:
        logger.exception(
            f"Remove variables {colnames_to_drop} from level1, level2, level3 , report_main_var, report_var_1, report_var_2, independent_features_linear, any of the fixed_effects, any of the random_effects, or imputation_lvl_list parameters as these variable have more than one unique value per pair, that will lead to data inconsistency during further data grouping and calculations.Please update the parameter and run the pipeline again."
        )
        return None

    # Data Aggregation
    grouping_agg_columns = list(
        filter(
            None,
            set(grouping_columns + additional_raw_variables + [column_with_date] + [promo_flag]),
        )
    )
    # sum all the columns listed in colnames_to_sum
    #     input_df_agg_sum = create_sum_df(input_df, grouping_agg_columns, colnames_to_sum)
    #     # calculate weighted mean all the columns listed in colnames_weighted_avg parameter
    #     input_df_agg_wtd_avg = create_wtd_avg_df(
    #         input_df, grouping_agg_columns, colnames_weighted_avg, volume
    #     )
    #     # join these aggregated dataframes with input data
    #     aggregated_df = input_df_agg_sum.join(input_df_agg_wtd_avg, grouping_agg_columns, "inner")

    aggregated_df = create_agg_df(
        input_df, grouping_agg_columns, colnames_to_sum, colnames_weighted_avg, volume
    )
    input_df = create_concatenated_column(input_df, grouping_agg_columns, "unique_combo", "")
    try:
        assert aggregated_df.count() == input_df.select("unique_combo").distinct().count()
        logger.info("granularity of the data lines with the expectation")
    except:
        logger.exception("granularity of the data does NOT line with the expectation")

    # Date column creation
    aggregated_df = date_columns_creation(
        aggregated_df, column_with_date, granularity, last_week_month_cutoff
    )

    # add holidays
    aggregated_df = create_holiday_columns(aggregated_df, granularity, holidays_config)
    # add covid
    #     aggregated_df = create_covid_columns(aggregated_df, granularity, holidays_config)

    # Add price related variables to aggregated data
    # create a column called pair which is equal to level 1 + level 2 + level 3 concatenated string
    aggregated_df = create_concatenated_column(aggregated_df, grouping_columns, "pair", "_")

    # create a column called price which is the divison of value column (from parameter) and vol column from parameter too
    aggregated_df = calc_price(aggregated_df, value, volume)

     # create edlp_price column
    
    #define promo_flag based on acv tpr 
    aggregated_df = aggregated_df.withColumn("promo_flag2", F.when(F.col("acv_tpr") > 5.0,1).otherwise(0))

    w = (Window.partitionBy(list(filter(None, [lvl1] + [lvl2]))).orderBy(list(filter(None, [lvl1] + [lvl2] + [f"year_{granularity}"]))))
    
    #Calculate non-promo weeks and choose which pricing model to be selected: Historical/ Future
    aggregated_df = aggregated_df.withColumn("Complement_Promo_flag2", F.when(F.col("promo_flag2")==0,1).otherwise(0))
    
    #Count of non-promo weeks in the history
    aggregated_df = aggregated_df.withColumn("Non_Promo_Weeks", F.sum("Complement_Promo_flag2").over(w))
    
    #Depending on number of non-promo weeks, selection of historical or future looking price model
    aggregated_df = aggregated_df.withColumn("Model_Used", F.when(F.col('Non_Promo_Weeks')>7, F.lit("Historical")).otherwise(F.lit("Future")))

    w1 = (Window.partitionBy(list(filter(None, [lvl1] + [lvl2]))).orderBy(list(filter(None, [lvl1] + [lvl2] + [f"year_{granularity}"]))).rowsBetween(Window.unboundedPreceding, 0))
    w2 = (Window.partitionBy(list(filter(None, [lvl1] + [lvl2]))).orderBy(list(filter(None, [lvl1] + [lvl2] + [f"year_{granularity}"]))).rowsBetween(0, Window.unboundedFollowing))

    #Collect the sequence of historical and future non-promo prices
    aggregated_df = aggregated_df.withColumn("sequence1", F.collect_list(F.when(F.col("promo_flag2")==0,F.col("Price"))).over(w1))
    aggregated_df = aggregated_df.withColumn("sequence2", F.collect_list(F.when(F.col("promo_flag2")==0,F.col("Price"))).over(w2))

    #Have the size of lists, if in history more than 13 weeks availbale then use last 14 weeks (including current week)
    #                        if in future more than 7 weeks available then use first 8 weeks (including current week)
    aggregated_df = aggregated_df.withColumn("SIZE1", F.size(F.col('sequence1')))
    aggregated_df = aggregated_df.withColumn("SIZE2", F.size(F.col('sequence2')))
    aggregated_df = aggregated_df.withColumn("sequence11", F.when(F.col("SIZE1")<14,F.col('sequence1')).otherwise(F.slice(F.col('sequence1'),-14,14)))
    aggregated_df = aggregated_df.withColumn("sequence22", F.when(F.col("SIZE2")<8,F.col('sequence2')).otherwise(F.slice(F.col('sequence2'),1,8)))

    #Arrange the trimmed historican and future non-promo weeks price lists into descending order and choose 2nd highest if available else choose the top
    aggregated_df = aggregated_df.withColumn("sequence111", aggregated_df.sequence11.cast("array<float>"))
    aggregated_df = aggregated_df.withColumn("sequence111", F.sort_array(F.col("sequence111"), asc=False))#.select("*", F.col("sequence11")[1].alias("edlp_price_hist"))
    aggregated_df = aggregated_df.withColumn("edlp_price_hist", F.when(F.col("SIZE1")>0,F.col("sequence111")[1]).otherwise(F.col("sequence111")[0]))
    aggregated_df = aggregated_df.withColumn("sequence222", aggregated_df.sequence22.cast("array<float>"))
    aggregated_df = aggregated_df.withColumn("sequence222", F.sort_array(F.col("sequence222"), asc=False))
    aggregated_df = aggregated_df.withColumn("edlp_price_fut", F.when(F.col("SIZE2")>0,F.col("sequence222")[1]).otherwise(F.col("sequence222")[0]))
 
    #Based on flag defined earlier, assign edlp_price
    aggregated_df = aggregated_df.withColumn("edlp_final",F.when(F.col("Model_Used")=="Future", F.col("edlp_price_fut")).otherwise(F.col("edlp_price_hist")))

    aggregated_df = aggregated_df.withColumn("price", aggregated_df["price"].cast("float"))
    aggregated_df = aggregated_df.withColumn("edlp_final", aggregated_df["edlp_final"].cast("float"))

    aggregated_df = aggregated_df.withColumn("edlp_final", F.when(F.col("edlp_final").isNull(),F.col("price")).otherwise(F.col("edlp_final")))
    aggregated_df = aggregated_df.withColumn("edlp_final", F.when(F.col("edlp_final")<F.col("price"),F.col("price")).otherwise(F.col("edlp_final")))

    aggregated_df = aggregated_df.withColumn("edlp_price", F.col("edlp_final"))

    aggregated_df = aggregated_df.drop("sequence1","sequence2","sequence11","sequence22","sequence111","sequence222","SIZE1","SIZE2")
    
    #w = (Window.partitionBy(list(filter(None, [lvl1] + [lvl2]))).orderBy(list(filter(None, [lvl1] + [lvl2] + [f"year_{granularity}"]))).rowsBetween(-13, 0))
    # create an edlp_price column with the 2nd largest value in the window
#aggregated_df = aggregated_df.withColumn("sequence", F.sort_array(F.collect_list(F.col("price")).over(w), asc=False)).select("*", F.col("sequence")[1].alias("edlp_price"))

#aggregated_df = aggregated_df.withColumn("edlp_price",F.when((F.size(aggregated_df["sequence"]) == 1), F.col("price")).otherwise(F.col("edlp_price")))


    # create max price and discount price columns
    window = Window.partitionBy(["pair", "year"])
    aggregated_df = aggregated_df.withColumn("max_price", F.max("price").over(window))
    aggregated_df = aggregated_df.withColumn(
        "discount_percentage",
        F.when(
            F.col("max_price") != 0, ((F.col("max_price") - F.col("price")) / F.col("max_price"))
        ).otherwise(0),
    )
    
    
    # To ensure outlier retailers have zero distrib. variable
    aggregated_df = aggregated_df.withColumn(
        distrib_var,
        F.when(
            (~F.col(lvl2).isin(outlier_retailers))
            ,
            F.col(distrib_var),
            ).otherwise(0),
    )
    
    
    # filtering steps
    # create a general flag
    aggregated_df = aggregated_df.withColumn("general_flag", F.lit("GOOD"))

    # filter 1 - removing outliers and any user given pairs
    aggregated_df = outlier_filter(aggregated_df, value, volume, distrib_var)
    aggregated_df = bad_pair_filter(
        aggregated_df,
        lvl1,
        lvl2,
        lvl1_to_remove,
        lvl2_to_remove,
        weeks_or_months_to_remove,
        granularity,
    )

    # create average volume column
    aggregated_df = calculate_average_volume(aggregated_df, lvl1, lvl2, lvl3, volume)

    # filter 2
    aggregated_df = cumulative_flag_filter(
        aggregated_df,
        lvl1,
        lvl3,
        value,
        percentile_pairs,
        bottom5_flag,
        manufacturer_col,
        manufacturer_name,
    )
    # filter 3
    aggregated_df = week_count_filter(
        aggregated_df, lvl1, lvl2, lvl3, granularity, week_count_out, month_count_out
    )
    # filter 4
    aggregated_df = price_cv_filter(
        aggregated_df,
        lvl1,
        lvl2,
        lvl3,
        price,
        cv_price_out,
        additional_filter,
        cv_price_crit,
        add_filter4_val,
    )
    # filter 5
    aggregated_df = average_volume_filter(aggregated_df, volume, volume_filter_pct)
    # filter 6
    aggregated_df = week_count_filter(
        aggregated_df, lvl1, lvl2, lvl3, granularity, week_count_out, month_count_out
    )
    # # filter 7
    # aggregated_df = distribution_filter(aggregated_df, distrib_var)

    # sort data by lvl2,lvl1,lvl3,year_granularity
    sorting_cols = list(filter(None, [lvl2] + [lvl1] + [lvl3] + [f"year_{granularity}"]))
    aggregated_df = aggregated_df.sort(sorting_cols, ascending=True)

    # percentile calculation
    aggregated_df = percentile_calculation(aggregated_df, lvl1, lvl2, lvl3, volume)
    # create smoothing columns
    if bool(items):
        aggregated_df = create_smoothing_items(aggregated_df, items, lvl1, lvl2, lvl3, granularity)
    # create acv columns
    # if bool(items):
    aggregated_df = create_acv_colums(
        aggregated_df,
        lvl1,
        lvl2,
        lvl3,
        granularity,
        acv_promos,
        acv_vars_list,
        distrib_var,
        outlier_retailers
    )
    # convert columns to log form
    aggregated_df = convert_to_log(aggregated_df, params_to_log_transform, distrib_var)
    # get cpi
    aggregated_df = calculate_cpi(aggregated_df, lvl2, lvl3, value, volume, cpi_crit, granularity)
    # get opi
    aggregated_df = calculate_opi(
        aggregated_df,
        lvl1,
        lvl2,
        lvl3,
        value,
        volume,
        cpi_crit,
        granularity,
        opi_grouping,
    )
    # get xpi
    aggregated_df = calculate_xpi(aggregated_df, cpi_crit, granularity)
    # get peak flags
    if bool(eval_peaks_years):
        aggregated_df = get_peak_flag(
            aggregated_df, lvl1, lvl2, lvl3, eval_peaks_years, peak_percentage_lmt, volume
        )
    # create covid flags
    aggregated_df = create_covid_flag(
        aggregated_df,
        granularity,
        lvl2,
        covid_flag_1,
        covid_flag_1_df,
        covid_flag_2,
        covid_flag_2_df,
        start_covid_period,
        covid_3_exclud_val,
        covid_3_period_start,
        covid_3_period_end,
        covid_4_exclud_val,
        covid_4_period_start,
        covid_4_period_end,
    )
    
    # create promo acv columns
    if bool(volume_promo_var):
        aggregated_df, parameters = create_promo_acv(aggregated_df, parameters)
        
    # promo flags
    # aggregated_df, promo_threshold_summary = create_promo_flag(
    #     aggregated_df,
    #     outlier_retailers,
    #     parameters
    # )
    
    # promo_threshold_summary = convert_sparkdf_to_pandasdf(promo_threshold_summary)
    
    # write_obj(
    #     promo_threshold_summary,
    #     catalog["promo_threshold_summary"]["filepath"],
    #     catalog["promo_threshold_summary"]["filename"],
    #     catalog["promo_threshold_summary"]["format"],
    # )

    # # pantry loading
    # aggregated_df = pantry_load_dynamic(
    #     aggregated_df,
    #     lvl1,
    #     lvl2,
    #     lvl3,
    #     volume,
    #     sorting_cols,
    #     pantry_loading,
    # )
    # # aggregated_df = pantry_load(aggregated_df, lvl1, lvl2, lvl3, volume, sorting_cols)
    #     # ensure all lvl columns and addional variables columns are of type string
    #     to_str = grouping_columns + additional_raw_variables
    #     aggregated_df[string_columns] = aggregated_df.select([F.col(c).cast(StringType()).alias(c) for c in to_str])

    # write intermediate spark data
    aggregated_df = aggregated_df.withColumn(
        "sequence_opi", aggregated_df["sequence_opi"].cast("string")
    )
    #aggregated_df = aggregated_df.withColumn("sequence", aggregated_df["sequence"].cast("string"))
    aggregated_df.write.format("csv").mode("overwrite").options(header="true", sep=",").save(
        path="{0}{1}/{1}.{2}".format(
            catalog["spark_all_data"]["filepath"], catalog["spark_all_data"]["filename"], "csv"
        )
    )

    # additional data
    if use_additional_input == True:
        for df in add_df_list:
            globals()[df] = load_spark_data(catalog[df])
        for df in add_df_list:
            aggregated_df = add_new_data(globals()[df], aggregated_df, parameters, df)

    
    
    ## Addition of holiday flags
    hol_level = parameters["ewb"]["data_management"]["hol_level"]
    hol_peak_percentage_lmt = parameters["ewb"]["data_management"]["hol_peak_percentage_lmt"]
    hol_dip_percentage_lmt = parameters["ewb"]["data_management"]["hol_dip_percentage_lmt"]
    path = (
        catalog["holiday_list"]["filepath"]
        + catalog["holiday_list"]["filename"]
        + "."
        + catalog["holiday_list"]["format"]
    )
    if os.path.exists(path):
        holidays = load_csv_data(path)
    else:
        logger.info("HOLIDAYS LIST FILE not found. Please upload it to input config folder.")
        return None
    aggregated_df = automate_holiday_flags(aggregated_df, holidays, hol_level, eval_peaks_years, hol_peak_percentage_lmt, hol_dip_percentage_lmt, parameters)
    
    
    aggregated_df = convert_sparkdf_to_pandasdf(aggregated_df,catalog)
    
    # split dataset
    raw_loess_all = aggregated_df
    raw_data_good = aggregated_df[aggregated_df["general_flag"] == "GOOD"]
    raw_data_drop = aggregated_df[aggregated_df["general_flag"] != "GOOD"]
    try:
        assert raw_loess_all.shape[0] == raw_data_good.shape[0] + raw_data_drop.shape[0]
        logger.info("data was split correctly")
    except:
        logger.exception("data was NOT split correctly")

    duration = pd.Timestamp.now() - start
    logger.info(f"data preperation done in {duration}s")

    # convert spark dataframes to pandas dataframe
    #     raw_loess_all = convert_sparkdf_to_pandasdf(raw_loess_all)
    #     raw_data_good = convert_sparkdf_to_pandasdf(raw_data_good)
    #     raw_data_drop = convert_sparkdf_to_pandasdf(raw_data_drop)
    unique_count_df = convert_sparkdf_to_pandasdf(unique_count_df,catalog)

    # ensure all levels columns are of type str
    for dataframe in [raw_loess_all, raw_data_good, raw_data_drop]:
        dataframe[grouping_columns] = dataframe[grouping_columns].apply(
            lambda x: x.astype("str").str.strip()
        )

    # write outputs
    write_obj(
        raw_loess_all,
        catalog["intermediate_all_data"]["filepath"],
        catalog["intermediate_all_data"]["filename"],
        catalog["intermediate_all_data"]["format"],
    )

    write_obj(
        raw_data_good,
        catalog["intermediate_good_data"]["filepath"],
        catalog["intermediate_good_data"]["filename"],
        catalog["intermediate_good_data"]["format"],
    )

    write_obj(
        raw_data_drop,
        catalog["intermediate_bad_data"]["filepath"],
        catalog["intermediate_bad_data"]["filename"],
        catalog["intermediate_bad_data"]["format"],
    )

    write_obj(
        unique_count_df,
        catalog["intermediate_unique_pairs"]["filepath"],
        catalog["intermediate_unique_pairs"]["filename"],
        catalog["intermediate_unique_pairs"]["format"],
    )

    write_obj(
        parameters,
        catalog["run_parameters"]["filepath"],
        catalog["run_parameters"]["filename"],
        catalog["run_parameters"]["format"],
    )

    logger.info("Data preparation outputs and run parameters written to cloud")

    return raw_loess_all, raw_data_good, raw_data_drop, unique_count_df


# create a unique pair df
def create_unique_pair(
    input_df: DataFrame, grouping_cols: List[str], column_list: List[str]
) -> DataFrame:
    """
    This function create a dataframe counting the unique pairs by specified group for a given set of columns
    Args:
        input_df: spark dataframe
        grouping_cols : list of columns to aggregate by
        column_list : list of colums to create unique pairs for
    Returns:
        out_df : spark dataframe with unique pair counts for the specified column_list
    """
    start = pd.Timestamp.now()

    logger.info(f"creating unique pairs for {grouping_cols}")

    out_df = input_df.select(grouping_cols).distinct()
    for variable in list(set(column_list) - set(grouping_cols)):
        df_intermediate = input_df.groupBy(grouping_cols).agg(
            F.countDistinct(variable).alias(variable)
        )
        out_df = out_df.join(df_intermediate, grouping_cols, "inner")

    duration = pd.Timestamp.now() - start
    logger.info(f"unique pair dataframe created in {duration}s")

    return out_df


def QC_unique_pair(
    df: DataFrame,
    additional_raw_variables: List[str],
    loess_indep_vars: List[str],
    grouping_columns: List[str],
) -> List[str]:
    """
    This function alerts the user of columns which are used in modeling but do not satisfy the uniqueness criteria per pair
    Args:
        df : spark dataframe
        additional_raw_variables : list of categorical columns
        loess_indep_vars : list of columns for loess model
        grouping_columns : list of grouping columns
    Returns:
        colnames_to_drop: list of column names to drop because uniqueness criteria is not met
    """

    # getting all columns that have more than 1 value per pair (lvl1, 2, 3)
    pandas_df = df.toPandas()
    tokeep = list(pandas_df._get_numeric_data().columns)
    colnames_to_drop = [col for col in tokeep if (pandas_df[col].values > 1).any()]

    if len(colnames_to_drop) > 0:
        for i in colnames_to_drop:
            if i in additional_raw_variables:
                logger.error(
                    f"The {i} needs to be removed from the 'additional_raw_variables' parameter as {i} has more than one unique value per pair, that will lead to data inconsistency during further data grouping and calculations.If you still want to include {i} in the 'additional_raw_variables' parameter, change the raw data so that {i} has a unique value per pair, then restart the pipeline"
                )
            if i in loess_indep_vars:
                logger.error(
                    f"The {i} needs to be removed from the 'loess_indep_vars' parameter as {i} has more than one unique value per pair, that will lead to data inconsistency during further data grouping and calculations.If you still want to include {i} in the 'loess_indep_vars' parameter, change the raw data so that {i} has a unique value per pair, then restart the pipeline"
                )


    else:
        logger.info(
            "All variables in the 'additional_raw_variables' parameter have only one unique value per pair. It's great."
            "\n",
        )

    logger.info(
        "Final check: the number of pairs grouped by levels == the number of pairs grouped by additional_raw_variables"
    )
    grouped_levels_count = df.select(grouping_columns).distinct().count()
    grouped_additional_count = df.select(additional_raw_variables).distinct().count()
    print(grouped_levels_count == grouped_additional_count)

    return colnames_to_drop


# create agg df
def create_agg_df(
    input_df: DataFrame,
    grouping_cols: List[str],
    column_list_sum: List[str],
    column_list_wt_avg: List[str],
    weight_col: str,
) -> DataFrame:
    """
    This function creates an aggregated dataframe for a given set of columns aggregated by grouping_cols
    Args:
        input_df: spark dataframe
        grouping_cols : list of columns to aggregate by
        column_list_sum : list of colums to sum
        column_list_wt_avg : list of colums to weighted average
        weight_col : column that we need to weight by
    Returns:
        agg_df : spark dataframe with columns aggregated
    """
    from pyspark.sql.functions import sum as _sum_pyspark

    start = pd.Timestamp.now()
    all_agg_cols = column_list_sum + column_list_wt_avg
    intermediate_df = input_df
    # calculate new columns as sumproduct (column * weight column)  for the columns which needs to be agg as weighted average
    for column in column_list_wt_avg:
        input_df = (
            input_df.withColumn(column, F.col(column) * F.col(weight_col))
            # input_df.withColumn('sumproduct_'+column, F.col(column) * F.col(weight_col))
        )

    exprs = [_sum_pyspark(x).alias("{0}".format(x)) for x in all_agg_cols]
    agg_df = input_df.groupBy(grouping_cols).agg(*exprs)
    for column in column_list_wt_avg:
        agg_df = agg_df.withColumn(column, F.col(column) / F.col(weight_col))

    duration = pd.Timestamp.now() - start
    logger.info(f"aggregated dataframe created in {duration}s")
    return agg_df


# create sum agg df
def create_sum_df(
    input_df: DataFrame, grouping_cols: List[str], column_list: List[str]
) -> DataFrame:
    """
    This function creates an aggregated dataframe for a given set of columns aggregated by grouping_cols
    Args:
        input_df: spark dataframe
        grouping_cols : list of columns to aggregate by
        column_list : list of colums to sum
    Returns:
        out_df : spark dataframe with columns aggregated
    """
    start = pd.Timestamp.now()
    out_df = input_df.select(grouping_cols).distinct()
    for column in column_list:
        df_intermediate = input_df.groupBy(grouping_cols).agg(F.sum(column).alias(column))
        out_df = out_df.join(df_intermediate, grouping_cols, "inner")

    duration = pd.Timestamp.now() - start
    logger.info(f"sum aggregated dataframe created in {duration}s")

    return out_df


# create weighted avg agg df
def create_wtd_avg_df(
    input_df: DataFrame, grouping_cols: List[str], column_list: List[str], weight_col: str
) -> DataFrame:
    """
    This function create an aggregated dataframe for a given set of columns aggregated by grouping_cols
    Args:
        input_df: spark dataframe
        grouping_cols : list of columns to aggregate by
        column_list : list of colums to calculate weighted avg for
        weight_col : column that we need to weight by
    Returns:
        out_df : spark dataframe with columns aggregated
    """
    start = pd.Timestamp.now()

    out_df = input_df.select(grouping_cols).distinct()
    for column in column_list:
        df_intermediate = (
            input_df.withColumn("numerator", F.col(column) * F.col(weight_col))
            .groupBy(grouping_cols)
            .agg(
                F.sum(weight_col).alias("total_weight"),
                F.sum("numerator").alias("weighted_" + column),
            )
            .withColumn(column, F.col("weighted_" + column) / F.col("total_weight"))
        )
        columns_to_drop = [i for i in df_intermediate.columns if i.startswith("weighted_")] + [
            "numerator",
            "total_weight",
        ]
        df_intermediate = df_intermediate.drop(*columns_to_drop)
        out_df = out_df.join(df_intermediate, grouping_cols, "inner")

    duration = pd.Timestamp.now() - start
    logger.info(f"weighted aggregated dataframe created by weight = {weight_col} in {duration}s")

    return out_df


# create date columns


def date_columns_creation(
    input_df: DataFrame, date_col_name: str, granularity: str, last_week_month_cutoff: int
) -> DataFrame:
    """
    This function creates date related columns, i.e. full_date, monthname, year, etc.
    Args:
        input_df: spark dataframe
        date_col_name : name of the date column to be transformed
        granularity : data granularuty
        last_week_month_cutoff : n weeks/months to keep
    Returns:
        input_df : spark dataframe with date columns added
    """
    start = pd.Timestamp.now()

    input_df = input_df.withColumn("date", F.to_date(F.col(date_col_name)))
    input_df = input_df.withColumn("full_date", F.next_day(F.col("date"), "SUN"))
    input_df = input_df.withColumn("week", F.weekofyear(F.col("date")))
    input_df = input_df.withColumn(
        "week",
        F.when(F.length(input_df["week"]) == 1, F.concat(F.lit("0"), input_df["week"])).otherwise(
            input_df["week"]
        ),
    )
    input_df = input_df.withColumn("month", F.month(F.col(date_col_name)))
    input_df = input_df.withColumn("monthname", F.upper(F.date_format(F.col(date_col_name), "MMM")))
    input_df = input_df.withColumn(
        "month",
        F.when(F.length(input_df["month"]) == 1, F.concat(F.lit("0"), input_df["month"])).otherwise(
            input_df["month"]
        ),
    )
    input_df = input_df.withColumn("year", F.year(F.col(date_col_name)))
    input_df = input_df.withColumn(
        "year",
        F.when(
            (F.col("month") == "01") & ((F.col("week") == "53") | (F.col("week") == "52")),
            F.col("year") - 1,
        ).otherwise(F.col("year")),
    )
    input_df = input_df.withColumn(
        "year",
        F.when((F.col("month") == "12") & (F.col("week") == "01"), F.col("year") + 1).otherwise(
            F.col("year")
        ),
    )

    input_df = create_concatenated_column(input_df, ["year", "month"], "year_month", "")
    input_df = create_concatenated_column(input_df, ["year", "week"], "year_week", "")

    input_df = input_df.withColumn("year", F.year(F.col(date_col_name)))

    # create moving date columns
    w = Window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    input_df = input_df.withColumn("min_date", F.min("full_date").over(w).cast("timestamp"))

    input_df = (
        input_df.withColumn("continous_year", F.year("full_date") - F.year("min_date") + 1)
        .withColumn("month_number", F.months_between(F.col("full_date"), F.col("min_date")) + 1)
        .withColumn("continous_day", F.datediff("full_date", F.to_date("min_date")) + 1)
        .withColumn("week_number", ((F.col("continous_day") / 7) + 1).cast("int"))
    )
    input_df = input_df.withColumn("month_number", F.floor(F.col("month_number")))

    # keep only n last weeks or months worth of data
    input_df = filter_n_weeks_months(input_df, granularity, last_week_month_cutoff)

    # update moving date columns
    w = Window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    input_df = input_df.withColumn("min_date", F.min("full_date").over(w).cast("timestamp"))

    input_df = (
        input_df.withColumn("continous_year", F.year("full_date") - F.year("min_date") + 1)
        .withColumn("month_number", F.months_between(F.col("full_date"), F.col("min_date")) + 1)
        .withColumn("continous_day", F.datediff("full_date", F.to_date("min_date")) + 1)
        .withColumn("week_number", ((F.col("continous_day") / 7) + 1).cast("int"))
    )
    input_df = input_df.withColumn("month_number", F.floor(F.col("month_number")))

    duration = pd.Timestamp.now() - start
    logger.info(f"date columns created in {duration}s")

    return input_df


def filter_n_weeks_months(
    df: DataFrame, granularity: str, last_week_month_cutoff: int
) -> DataFrame:
    """
    This function keeps only the latest n weeks/months specified in last_week_month_cutoff
    Args:
        df : spark dataframe
        granularity : data granularuty
        last_week_month_cutoff : n weeks/months to keep
    Returns:
        df : spark dataframe
    """
    if bool(last_week_month_cutoff):
        maxval = df.select(F.max(f"{granularity}_number")).first()
        df = df.where(F.col(f"{granularity}_number") > (maxval[0] - last_week_month_cutoff))
    return df


# holiday flags
def create_holiday_columns(df: DataFrame, granularity: str, holidays_config: dict) -> DataFrame:
    """
    This function takes lists of dates in year_week/year_month format for holidays and
    creates dummy variable columns to indicate weeks with holidays running
    Args:
        df : spark dataframe
        granularity : data granularity
        holidays_config : dict with holidays, dates and filter column names and values
    Returns:
        df : pyspark dataframe with holiday columns
    """
    start = pd.Timestamp.now()

    for holiday in holidays_config["holidays"]:
        df = df.withColumn(holiday["name"], F.lit(0))
        if (len(holiday["filter_column"]) > 0) and (len(holiday["filter_values"]) > 0):
            df = df.withColumn(
                holiday["name"],
                F.when(
                    (F.col(holiday["filter_column"]).isin(holiday["filter_values"]))
                    & (F.col(f"year_{granularity}").isin(holiday["weeks_or_months"])),
                    1,
                ).otherwise(F.col(holiday["name"])),
            )
        else:
            df = df.withColumn(
                holiday["name"],
                F.when(
                    (F.col(f"year_{granularity}").isin(holiday["weeks_or_months"])), 1
                ).otherwise(F.col(holiday["name"])),
            )
    duration = pd.Timestamp.now() - start
    logger.info(f"holiday columns created in {duration}s")

    try:
        for holiday in holidays_config["holidays"]:
            assert set(
                df.select(holiday["name"]).distinct().rdd.map(lambda r: r[0]).collect()
            ).issubset({0, 1})

        logger.info(f"holiday column is binary")
    except AssertionError as e:
        logger.exception("holiday column is not binary:", e)

    return df


# covid flags
def create_covid_columns(df: DataFrame, granularity: str, holidays_config: dict) -> DataFrame:
    """
    This function takes start and end date in year_week/year_month format for covid phases and
    creates dummy variable columns to indicate weeks with covid phase running. Not used in default
    pipeline - create_covid_flag is used instead.
    Args:
        df : pyspark dataframe with raw modeling data
        granularity : data granularity
        holidays_config : dict with covid phases, dates and filter column name and values
    Returns:
        df : pyspark dataframe with covid columns
    """
    start = pd.Timestamp.now()

    for phase in holidays_config["covid_phases"]:
        df = df.withColumn(phase["name"], F.lit(0))
        start = phase["weeks_or_months"]["start"]
        end = phase["weeks_or_months"]["end"]
        dates = np.arange(start, end + 1).tolist()
        if (len(phase["filter_column"]) > 0) and (len(phase["filter_values"]) > 0):
            df = df.withColumn(
                phase["name"],
                F.when(
                    (F.col(phase["filter_column"]).isin(phase["filter_values"]))
                    & (F.col(f"year_{granularity}").isin(dates)),
                    1,
                ).otherwise(F.col(phase["name"])),
            )
        else:
            start = phase["weeks_or_months"]["start"]
            end = phase["weeks_or_months"]["end"]
            dates = np.arange(start, end + 1).tolist()
            df = df.withColumn(
                phase["name"],
                F.when((F.col(f"year_{granularity}").isin(dates)), 1).otherwise(
                    F.col(phase["name"])
                ),
            )
    duration = pd.Timestamp.now() - start
    logger.info(f"covid columns created in {duration}s")

    try:
        for phase in holidays_config["covid_phases"]:
            assert set(
                df.select(phase["name"]).distinct().rdd.map(lambda r: r[0]).collect()
            ).issubset({0, 1})
        logger.info(f"covid column is binary")
    except AssertionError as e:
        logger.exception("covid column is not binary:", e)
        exceptions.append(e)

    return df


def calc_price(input_df: DataFrame, value: str, volume: str) -> DataFrame:
    """
    This function calculates price
    Args:
        input_df : spark dataframe
        value : column for value
        volume : column for volume
    Returns:
        input_df : spark dataframe with price column added
    """
    start = pd.Timestamp.now()

    input_df = input_df.withColumn(
        "price", F.when(F.col(volume) == 0, 0).otherwise(F.lit(F.col(value) / F.col(volume)))
    )

    duration = pd.Timestamp.now() - start
    logger.info(f"price column created in {duration}s")

    return input_df


# Percentile calculations
def percentile_calculation(
    df: DataFrame, lvl1: str, lvl2: str, lvl3: str, volume: str
) -> DataFrame:
    """
    This function creates 5 volume percentile columns in the dataframe
    Args:
        df : spark dataframe
        lvl1 : level of aggregation
        lvl2 : level of aggregation
        lvl3 : level of aggregation
        volume : column containing volume
    Returns:
        df : spark dataframe with percentile columns
    """
    start = pd.Timestamp.now()

    grouping_cols = list(filter(None, [lvl1] + [lvl2] + [lvl3] + ["general_flag"]))

    grp_window = Window.partitionBy(grouping_cols)
    quantiles = F.expr(f"percentile_approx({volume}, array(0.02, 0.05, 0.95, 0.98, 0.99))")
    for i, name in enumerate(["p2", "p5", "p95", "p98", "p99"]):
        df = df.withColumn(
            name,
            F.when((F.col("general_flag") == "GOOD"), quantiles.over(grp_window)[i]).otherwise(
                None
            ),
        )

    duration = pd.Timestamp.now() - start
    logger.info(f"percentile calculation done by {lvl1}, {lvl2} for {volume} in {duration}s")

    return df


# Create smoothing items columns


def create_smoothing_items(
    df: DataFrame, items: str, lvl1: str, lvl2: str, lvl3: str, granularity: str
) -> DataFrame:
    """
    This function creates smoothing item columns
    Args:
        df : spark dataframe
        items: distribution points column
        lvl1 : level of aggregation
        lvl2 : level of aggregation
        lvl3 : level of aggregation
        granularity : data granularity
    Returns:
        df : spark dataframe with smoothing items columns added
    """

    start = pd.Timestamp.now()
    sorting_cols = list(filter(None, [lvl2] + [lvl1] + [lvl3] + [f"year_{granularity}"]))
    grouping_cols = list(filter(None, [lvl2] + [lvl1] + ["general_flag"]))

    df = df.withColumn(
        "avg_items",
        F.when((F.col("general_flag") == "GOOD"), F.col(items) / 100).otherwise(None),
    )
    w = Window.partitionBy(grouping_cols).orderBy(sorting_cols).rowsBetween(-3, 0)
    df = df.withColumn(
        "smooth_avg_items",
        F.when((F.col("general_flag") == "GOOD"), F.mean("avg_items").over(w)).otherwise(None),
    )
    df = df.withColumn(
        "log_avg_items",
        F.when(
            (F.col("avg_items") > 0),
            F.log(F.col("avg_items")),
        ).otherwise(F.when((F.col("avg_items") <= 0), 0).otherwise(None)),
    )
    df = df.withColumn(
        "log_smooth_avg_items",
        F.when(
            (F.col("smooth_avg_items") > 0),
            F.log(F.col("smooth_avg_items")),
        ).otherwise(F.when((F.col("smooth_avg_items") <= 0), 0).otherwise(None)),
    )

    duration = pd.Timestamp.now() - start
    logger.info(f"smoothing columns created in  {duration}s")

    return df


# Create acv columns


def create_acv_colums(
    df: DataFrame,
    lvl1: str,
    lvl2: str,
    lvl3: str,
    granularity: str,
    acv_promos: List[str],
    acv_vars_list: List[str],
    distrib_var: List[str],
    outlier_retailers: List[str]
) -> DataFrame:
    """
    This function creates acv columns
    Args:
        df : spark dataframe
        lvl1 : level of aggregation
        lvl2 : level of aggregation
        lvl3 : level of aggregation
        granularity : data granularity
        acv_promos : list of columns with promo acv values
        acv_vars_list : list of column names for acv variables
        distrib_var : columns containing distribution
    Returns:
        df : spark dataframe with acv columns added
    """

    start = pd.Timestamp.now()
    sorting_cols = list(filter(None, [lvl2] + [lvl1] + [lvl3] + [f"year_{granularity}"]))
    grouping_cols = list(filter(None, [lvl2] + [lvl1] + ["general_flag"]))

    # create acv_any_promo : is the sum of all acv promo columns specified
    if len(acv_promos) > 0:
        df = df.withColumn("acv_any_promo", sum(df[col] for col in acv_promos))

    # creating special acv,log_acv variables
    if len(acv_vars_list) > 0:
        for i, var in enumerate(acv_promos):
            df = df.withColumn(
                f"{var}_by_100",
                F.when(
                    (~F.col(lvl2).isin(outlier_retailers))        #to ensure outlier retailers do not have acv values
                    &
                    (F.col("general_flag") == "GOOD") & (F.col(acv_promos[i]) > 0),
                    F.col(acv_promos[i]) / 100,
                ).otherwise(
                    F.when(
                        (F.col("general_flag") == "GOOD") & (F.col(acv_promos[i]) <= 0), 0
                    ).otherwise(None)
                ),
            )
            df = df.withColumn(
                f"{var}_log",
                F.when(
                    (~F.col(lvl2).isin(outlier_retailers))       #to ensure outlier retailers do not have acv values     
                    &
                    (F.col(acv_promos[i]) > 0),
                    F.log(F.col(acv_promos[i])),
                ).otherwise(F.when(F.col(acv_promos[i]) <= 0, 0).otherwise(None)),
            )

    # create smooth acv column
    w = Window.partitionBy(grouping_cols).orderBy(sorting_cols).rowsBetween(-3, 0)
    df = df.withColumn(
        "smooth_acv",
        F.when((F.col("general_flag") == "GOOD"), F.mean(F.col(distrib_var)).over(w)).otherwise(
            None
        ),
    )
    df = df.withColumn("acv_diff", F.col(distrib_var) - F.col("smooth_acv"))

    duration = pd.Timestamp.now() - start
    logger.info(f"acv columns created in  {duration}s")

    return df


# Create Log transformed features


def convert_to_log(
    df: DataFrame, params_to_log_transform: List[str], distrib_var: str
) -> DataFrame:
    """
    Helper function to convert and filter out values after they are converted to log values
    Args:
        df : pyspark dataframe with raw modeling data
        params_to_log_transform : parameters on which to apply log operation
        distrib_var : columns containing distribution
    Returns:
        df : spark dataframe with log prefix columns with converted values
    """
    start = pd.Timestamp.now()

    for old_column in params_to_log_transform:
        new_column = "log_" + old_column
        df = df.withColumn(
            new_column,
            F.when(
                (F.col(old_column) > 0),
                F.log(F.col(old_column)),
            ).otherwise(F.when((F.col(old_column) <= 0), 0).otherwise(None)),
        )

        if old_column == distrib_var:

            df = df.withColumn(
                new_column,
                F.when(
                    (
                        (
                            (F.col(new_column).isNull())
                            | (F.isnan(new_column))
                            | (F.col(new_column) <= 0)
                        )
                    ),
                    F.lit(0),
                ).otherwise(F.col(new_column)),
            )

        else:

            df = df.withColumn(
                new_column,
                F.when(
                    (F.col(new_column) == 0),
                    F.lit(0),
                ).otherwise(F.col(new_column)),
            )

    duration = pd.Timestamp.now() - start
    logger.info(f"{params_to_log_transform} converted to log in {duration}s")

    return df


# CPI calculation


def calculate_cpi(
    df: DataFrame, lvl2: str, lvl3: str, value: str, volume: str, cpi_crit: float, granularity: str
) -> DataFrame:
    """
    This function calculates CPI for a given pair and week
    Args:
        df : pyspark dataframe with raw modeling data
        lvl2 : level of aggregation
        lvl3 : level of aggregation
        value : value column
        volume : volume column
        cpi_crit : threshold for CPI
        granularity : data granularity
    Returns:
        df: spark dataframe with CPI and LOG CPI columns
    """
    start = pd.Timestamp.now()

    grouping_cols = list(filter(None, [lvl2] + [lvl3] + [f"year_{granularity}"] + ["general_flag"]))
    grp_window = Window.partitionBy(grouping_cols)

    df = df.withColumn("total_value", F.sum(value).over(grp_window))

    df = df.withColumn("total_volume", F.sum(volume).over(grp_window))

    df = df.withColumn(
        "cpi",
        (df.total_value - df[value]) / (df.total_volume - df[volume]),
    )
    df = df.withColumn(
        "log_cpi",
        F.when((df.cpi >= cpi_crit), F.log(df.cpi)).otherwise(
            F.when((df.cpi == 0), 0).otherwise(0.01)
        ),
    )
    df = df.withColumn(
        "log_cpi",
        F.when(df.cpi.isNull(), 0).otherwise(F.col("log_cpi")),
    )

    if df.agg(F.sum("cpi")).collect()[0][0] == 0:  # output the msg and stop pipeline
        logger.warning(
            "All log_cpi' values are 0, because in each group only 1 observation is found, resp. 'cpi' cannot be calculated.",
            "\nPlease try to increase latest_weeks_cut_off value or change the values for lvl2 or/and for lvl3, because with the \
                present values you can not calculate models with cpi (S2 and U2",
        )
    else:
        duration = pd.Timestamp.now() - start
        logger.info(f"cpi calculation done at {granularity} level in {duration}s")
        return df


# OPI calculation


def calculate_opi(
    df: DataFrame,
    lvl1: str,
    lvl2: str,
    lvl3: str,
    value: str,
    volume: str,
    cpi_crit: float,
    granularity: str,
    opi_grouping: List[str],
) -> DataFrame:
    """
    This function calculates OPI for a given pair and week.
    Args:
        df: spark dataframe
        lvl1 : level of aggregation
        lvl2 : level of aggregation
        lvl3 : level of aggregation
        value : value column
        volume : volume column
        cpi_crit : threshold for CPI
        granularity : data granularity
        opi_grouping: columns to group by
    Returns:
        df: pyspark dataframe with OPI and LOG OPI columns
    """
    start = pd.Timestamp.now()

    grouping_cols = list(filter(None, opi_grouping + ["general_flag"]))
    grp_window = Window.partitionBy(grouping_cols)

    df = df.withColumn("total_value_manuf", F.sum(value).over(grp_window))
    df = df.withColumn("total_volume_manuf", F.sum(volume).over(grp_window))

    df = df.withColumn(
        "opi", (df.total_value_manuf - df[value]) / (df.total_volume_manuf - df[volume])
    ).fillna(0, subset=["opi"])

    df = df.withColumn(
        "log_opi",
        F.when(df.opi >= cpi_crit, F.log(df.opi)).otherwise(
            F.when(df.opi < cpi_crit, F.log(F.lit(0.0001))).otherwise(None)
        ),
    )

    # create edlp_opi column
    w = Window.orderBy(
        list(filter(None, [lvl1] + [lvl2] + [lvl3] + [f"year_{granularity}"]))
    ).rowsBetween(-13, 0)
    # create an edlp_opi column with the 2nd largest value in the window
    df = df.withColumn(
        "sequence_opi", F.sort_array(F.collect_list(F.col("opi")).over(w), asc=False)
    ).select("*", F.col("sequence_opi")[1].alias("edlp_opi"))
    df = df.withColumn(
        "edlp_opi",
        F.when((F.size(df["sequence_opi"]) == 1), F.col("opi")).otherwise(F.col("edlp_opi")),
    )

    if df.agg(F.sum("opi")).collect()[0][0] == 0:  # output the msg and stop pipeline
        logger.warning(
            "All log_opi' values are 0, because in each group only 1 observation is found, resp. 'opi' cannot be calculated.",
            "\nPlease try to increase latest_weeks_cut_off value or change the values for lvl2 or/and for lvl3, because with the \
                present values you can not calculate models with opi (S2 and U2",
        )
    else:
        duration = pd.Timestamp.now() - start
        logger.info(f"opi calculation done at {granularity} level in {duration}s")
        return df


# XPI calculation


def calculate_xpi(df: DataFrame, cpi_crit: float, granularity: str) -> DataFrame:
    """
    This function calculates XPI for a given pair and week.
    Args:
        df: spark dataframe
        cpi_crit : threshold for CPI
        granularity : data granularity
    Returns:
        df: spark dataframe with XPI and LOG XPI columns
    """
    start = pd.Timestamp.now()

    df = df.withColumn(
        "total_value_xpi",
        F.col("total_value") - F.col("total_value_manuf"),
    )
    df = df.withColumn(
        "total_volume_xpi",
        F.col("total_volume") - F.col("total_volume_manuf"),
    )

    df = df.withColumn("xpi", (df.total_value_xpi) / (df.total_volume_xpi))

    df = df.withColumn(
        "log_xpi",
        F.when(df.xpi >= cpi_crit, F.log(df.xpi)).otherwise(
            F.when(df.xpi == 0, 0).otherwise(F.log(F.lit(0.01)))
        ),
    )

    df = df.withColumn(
        "log_xpi",
        F.when(df.xpi.isNull(), 0).otherwise(F.col("log_xpi")),
    )

    if df.agg(F.sum("xpi")).collect()[0][0] == 0:  # output the msg and stop pipeline
        logger.warning(
            "All log_xpi' values are 0, because in each group only 1 observation is found, resp. 'xpi' cannot be calculated.",
            "\nPlease try to increase latest_weeks_cut_off value or change the values for lvl2 or/and for lvl3, because with the \
                present values you can not calculate models with xpi (S2 and U2",
        )
    else:
        duration = pd.Timestamp.now() - start
        logger.info(f"xpi calculation done at {granularity} level in {duration}s")
        return df


# Create Peak flags


def get_year_volume(
    df: DataFrame, lvl1: str, lvl2: str, lvl3: str, peak_year: int, volume: str
) -> DataFrame:
    """
    Helper function to get mean volume for one year
    Args:
        df: spark dataframe
        lvl1 : level of aggregation
        lvl2 : level of aggregation
        lvl3 : level of aggregation
        peak_year : year to evaluate
        volume : volume column
    Returns:
        df: spark dataframe with year volume calculated
    """
    grouping_cols = list(filter(None, [lvl1] + [lvl2] + [lvl3] + ["year"]))
    return df.withColumn(
        f"vol_{str(peak_year)[-2:]}",
        F.when(
            F.col("year") == peak_year, F.mean(volume).over(Window.partitionBy(grouping_cols))
        ).otherwise(0),
    ).fillna(0)


def get_peak_percent(df: DataFrame, peak_year: int, volume: str) -> DataFrame:
    """
    Helper function to calculate peaks in percentage for one year
    Args:
        df: spark dataframe
        peak_year : year to evaluate
        volume : volume column
    Returns:
        df: spark dataframe with year peak calculated
    """
    df = df.withColumn(
        f"peak{str(peak_year)[-2:]}",
        F.when(
            (F.col("year") == peak_year),
            (
                ((F.col(volume) - F.col(f"vol_{str(peak_year)[-2:]}")) * 100)
                / F.col(f"vol_{str(peak_year)[-2:]}")
            ),
        ).otherwise(0),
    ).fillna(0)
    return df.withColumn(
        f"peak{str(peak_year)[-2:]}", F.round(F.col(f"peak{str(peak_year)[-2:]}"), 7)
    )


def get_peak_flag(
    df: DataFrame,
    lvl1: str,
    lvl2: str,
    lvl3: str,
    peak_years: List[int],
    peak_percentage_lmt: int,
    volume: str,
) -> DataFrame:
    """
    This function creates peak flags to control for abrupt movements in volume if peak_years parameter is defined
    Args:
        df: spark dataframe
        lvl1 : level of aggregation
        lvl2 : level of aggregation
        lvl3 : level of aggregation
        peak_years : list of years to evaluate
        peak_percentage_lmt: threshold value for peak
        volume : volume column
    Returns:
        df: spark dataframe with peak flags columns for specified years
    """
    start = pd.Timestamp.now()
    df = df.withColumn("peak_flag", F.lit(0))

    for peak_year in peak_years:
        df = get_year_volume(df, lvl1, lvl2, lvl3, peak_year, volume)
        df = get_peak_percent(df, peak_year, volume)
        df = df.withColumn(
            f"peak_flag{str(peak_year)[-2:]}",
            F.when(F.col(f"peak{str(peak_year)[-2:]}") > peak_percentage_lmt, 1).otherwise(0),
        ).fillna(0)
        # assign peak flag = 1 if any year peak flag is 1
        df = df.withColumn(
            "peak_flag",
            F.when(F.col(f"peak_flag{str(peak_year)[-2:]}") == 1, 1).otherwise(F.col("peak_flag")),
        )
    duration = pd.Timestamp.now() - start
    logger.info(f"peak flags created for {lvl1}, {lvl2}, {lvl3} in {duration}s")
    return df


def bad_pair_filter(
    input_df: DataFrame,
    lvl1: str,
    lvl2: str,
    lvl1_to_remove: List[str],
    lvl2_to_remove: List[str],
    weeks_or_months_to_remove: List[int],
    granularity: str,
) -> DataFrame:
    """
    This function flags pair which you anticipatorily know that are not suitable for modeling, you can remove it from raw data using this filter.
    not used in the pipeline.
    Args:
        input_df: spark dataframe
        lvl1 : level of aggregation
        lvl2 : level of aggregation
        lvl1_to_remove : list of lvl1 values to remove
        lvl2_to_remove : list of lvl2 values to remove
        weeks_or_months_to_remove : list of weeks or months in format 'yyyyww' or 'yyyymm' to remove
        granularity: string denoting granularity of the data - either 'week' or 'month'
    Returns:
        input_df: pyspark dataframe with filtering flag updated
    """
    start = pd.Timestamp.now()

    input_df = input_df.withColumn(
        "general_flag",
        F.when(
            (
                F.col(lvl1).isin(lvl1_to_remove)
                | F.col(lvl2).isin(lvl2_to_remove)
                | F.col(f"year_{granularity}").isin(weeks_or_months_to_remove)
            )
            & (F.col("general_flag") == "GOOD"),
            "DROP: Data Cleaning",
        ).otherwise(F.col("general_flag")),
    )

    duration = pd.Timestamp.now() - start
    logger.info(f"general flag updated for bad pairs of {lvl1}, {lvl2} in {duration}s")

    return input_df


def outlier_filter(input_df: DataFrame, value: str, volume: str, distrib_var: str) -> DataFrame:
    """
    This function flags rows that have 0 or negative values for value or volume and infinite for distrib_var
    Args:
        input_df: spark dataframe
        value: value column
        volume : volume column
        distrib_var : distribution column
    Returns:
        input_df: pyspark dataframe with filtering flag updated
    """
    start = pd.Timestamp.now()

    input_df = input_df.withColumn(
        "general_flag",
        F.when(
            ((F.col(value) <= 0) | (F.col(volume) <= 0) | (F.col(distrib_var).isin([np.inf])))
            & (F.col("general_flag") == "GOOD"),
            "DROP: Data Cleaning",
        ).otherwise(F.col("general_flag")),
    )

    duration = pd.Timestamp.now() - start
    logger.info(f"general flag updated with outlier filter in {duration}s")

    return input_df


def cumulative_flag_filter(
    input_df: DataFrame,
    lvl1: str,
    lvl3: str,
    value: str,
    percentile_pairs: float,
    bottom5_flag: str,
    manufacturer_col: str,
    manufacturer_name: List[str],
) -> DataFrame:
    """
    This function flags lvl1 and lvl3 combinations which don't contribute to the top percentile_pair value of sales
    Args:
        input_df: spark dataframe
        lvl1 : level of aggregation
        lvl3 : level of aggregation
        value: value column
        percentile_pairs: percentile threshold
        bottom5_flag: column containing bottom5_flag if present in input data
    Returns:
        input_df: spark dataframe with filtering flag updated
    """
    start = pd.Timestamp.now()

    grouping_cols = list(filter(None, [lvl1] + [lvl3]))
    select_cols = grouping_cols + [value]

    if len(bottom5_flag) > 0:
        F.when(
            (F.col(bottom5_flag) == 0) & (F.col("general_flag") == "GOOD"),
            "DROP: Outlier - Negative",
        ).otherwise(F.col("general_flag"))

    else:

        # create pair summary df with value, pct_value and cumulative pct
        pair_summary = input_df.select([col for col in select_cols]).filter(
            F.col("general_flag") == "GOOD"
        )
        # total value
        total_sum = pair_summary.agg({f"{value}": "sum"}).first()[0]

        pair_summary = (
            pair_summary.groupBy(grouping_cols)
            .agg(F.sum(value).alias("total_value"))
            .withColumn("pct_value", F.col("total_value") / total_sum)
        )
        pair_summary = pair_summary.withColumn(
            "cum_percent",
            F.sum("pct_value").over(Window.partitionBy().orderBy(F.col("pct_value").desc())),
        )

        # add cumulative flag
        pair_summary = pair_summary.withColumn(
            "cumulative_flag", F.when((F.col("cum_percent") <= percentile_pairs), 1).otherwise(0)
        )
        pair_summary = pair_summary.withColumn(
            "general_flag_pair",
            F.when((F.col("cumulative_flag") == 1), "GOOD").otherwise(
                f"DROP: {percentile_pairs*100}% cut off"
            ),
        )
        pair_summary_drop = pair_summary.select(
            grouping_cols + ["general_flag_pair", "cumulative_flag"]
        ).filter(F.col("cumulative_flag") == 0)

        # join with input_df with updated general flag
        input_df = input_df.join(pair_summary, grouping_cols, "left")
        input_df = (
            input_df.withColumn(
                "general_flag_final",
                F.when(
                    (
                        (input_df["general_flag"] != input_df["general_flag_pair"])
                        & (~F.col(manufacturer_col).isin(manufacturer_name))
                    ),
                    input_df["general_flag_pair"],
                ).otherwise(input_df["general_flag"]),
            )
            .drop("general_flag", "general_flag_pair")
            .withColumnRenamed("general_flag_final", "general_flag")
        )

    duration = pd.Timestamp.now() - start
    logger.info(f"general flag updated with cumulative flags in {duration}s")

    return input_df


def week_count_filter(
    df: DataFrame,
    lvl1: str,
    lvl2: str,
    lvl3: str,
    granularity: str,
    week_count_out: int,
    month_count_out: int,
) -> DataFrame:
    """
    This function flags combinations of levels 1-3, where week/month count < minimum number of weeks/months data required for robust coefficient estimation
    to avoid a situation where we have only 1 unique observation per pair, and mathematically we cannot compute the coefficient of variation.
    Args:
        df: spark dataframe
        lvl1 : level of aggregation
        lvl2 : level of aggregation
        lvl3 : level of aggregation
        granularity : data granularity
        week_count_out : minimal number of weeks
        month_count_out : minimal number of months
    Returns:
        df: spark dataframe with filtering flag updated
    """
    start = pd.Timestamp.now()

    # count unique weeks
    grouping_cols = list(filter(None, [lvl1] + [lvl2] + [lvl3] + ["general_flag"]))

    window = Window.partitionBy(grouping_cols)
    df = df.withColumn(
        f"{granularity}_count",
        F.when(
            F.col("general_flag") == "GOOD",
            F.size(F.collect_set(f"year_{granularity}").over(window)),
        ).otherwise(F.size(F.collect_set(f"year_{granularity}").over(window))),
    )
    # set flag
    if granularity == "week":
        df = df.withColumn(
            "general_flag",
            F.when(
                (F.col(f"{granularity}_count") < week_count_out)
                & (F.col("general_flag") == "GOOD"),
                "DROP: Data Cleaning",
            ).otherwise(F.col("general_flag")),
        )
    if granularity == "month":
        df = df.withColumn(
            "general_flag",
            F.when(
                (F.col(f"{granularity}_count") < month_count_out)
                & (F.col("general_flag") == "GOOD"),
                "DROP: Data Cleaning",
            ).otherwise(F.col("general_flag")),
        )

    duration = pd.Timestamp.now() - start
    logger.info(f"general flag updated with week count filter in {duration}s")

    return df


def price_cv_filter(
    df: DataFrame,
    lvl1: str,
    lvl2: str,
    lvl3: str,
    price_column: str,
    cv_price_out: int,
    additional_filter: bool,
    cv_price_crit: float,
    add_filter4_val: List[str],
) -> DataFrame:
    """
    This function flags combinations of levels 1-3, where coefficient of variance of price <= price variation coefficient, specified in the config file
    Args:
        df : pyspark dataframe with raw modeling data
        lvl1 : level of aggregation
        lvl2 : level of aggregation
        lvl3 : level of aggregation
        price_column : price column
        cv_price_out : threshold for cv
        additional_filter : boolean value, whether to apply additional filter to keep values dropped by cv_price_out
        cv_price_crit : threshold for cv for level 2 values from add_filter4_val if additional_filter == True
        add_filter4_val : list of level 2 values to keep if cv is above cv_price_crit
    Returns:
        df: spark dataframe with filtering flag updated
    """
    start = pd.Timestamp.now()

    # calculate cv
    grouping_cols = list(filter(None, [lvl1] + [lvl2] + [lvl3] + ["general_flag"]))
    window = Window.partitionBy(grouping_cols)
    df = df.withColumn(
        "std",
        F.when(
            (F.col("general_flag") == "GOOD"), F.stddev_samp(price_column).over(window)
        ).otherwise(None),
    )
    df = df.withColumn(
        "mean",
        F.when((F.col("general_flag") == "GOOD"), F.mean(price_column).over(window)).otherwise(
            None
        ),
    )
    df = df.withColumn(
        "cv_price",
        F.when((F.col("general_flag") == "GOOD"), (100.0 * df["std"]) / df["mean"]).otherwise(None),
    ).drop(*["total", "std", "mean"])

    # set flag
    df = df.withColumn(
        "general_flag",
        F.when(
            ((F.col("cv_price") < cv_price_out) | (F.col("cv_price").isNull()))
            & (F.col("general_flag") == "GOOD"),
            "DROP: Data Cleaning",
        ).otherwise(F.col("general_flag")),
    )

    if additional_filter:
        df = df.withColumn(
            "general_flag",
            F.when(
                (F.col("cv_price").isNull() == False)
                & (F.col("general_flag") == "DROP: Data Cleaning")
                & (F.col("cv_price") >= cv_price_crit)
                & (F.col(lvl2).isin(add_filter4_val)),
                "GOOD",
            ).otherwise(F.col("general_flag")),
        )

    duration = pd.Timestamp.now() - start
    logger.info(f"general flag updated with price variation coefficient information in {duration}s")

    return df


def calculate_average_volume(
    df: DataFrame, lvl1: str, lvl2: str, lvl3: str, volume: str
) -> DataFrame:
    """
    This function creates average volume column
    Args:
        df:  spark dataframe with raw modeling data
        lvl1 : level of aggregation
        lvl2 : level of aggregation
        lvl3 : level of aggregation
        volume: volume column
    Returns:
        df: spark dataframe with average volume column
    """
    start = pd.Timestamp.now()
    grouping_cols = list(filter(None, [lvl1] + [lvl2] + [lvl3] + ["general_flag"]))
    window = Window.partitionBy(grouping_cols)
    df = df.withColumn(
        "avg_volume",
        F.when((F.col("general_flag") == "GOOD"), F.mean(volume).over(window)).otherwise(None),
    )

    duration = pd.Timestamp.now() - start
    logger.info(f"average volume colum for good pairs created in {duration}s")

    return df


def average_volume_filter(df: DataFrame, volume: str, volume_filter_pct: float) -> DataFrame:
    """
    This function flags rows where volume is less than 10% of the average volume
    Args:
        df : spark dataframe
        volume : volume column
        volume_filter_pct: threshold for volume
    Returns:
        df : spark dataframe with filtering flag updated
    """
    start = pd.Timestamp.now()

    df = df.withColumn(
        "general_flag",
        F.when(
            (F.col(volume) < F.col("avg_volume") * volume_filter_pct)
            & (F.col("general_flag") == "GOOD"),
            "DROP: Data Cleaning",
        ).otherwise(F.col("general_flag")),
    )

    duration = pd.Timestamp.now() - start
    logger.info(f"general flag updated with average volume information in {duration}s")

    return df


def distribution_filter(df: DataFrame, distribution: str) -> DataFrame:
    """
    This function flags rows where weighted distribution = 0. not used in pipeline
    Args:
        df : spark dataframe
        distribution : column with distribution
    Returns:
        df: spark dataframe with filtering flag updated
    """
    start = pd.Timestamp.now()

    df = df.withColumn(
        "general_flag",
        F.when(
            (F.col(distribution) == 0) & (F.col("general_flag") == "GOOD"),
            "DROP: Data Cleaning",
        ).otherwise(F.col("general_flag")),
    )

    duration = pd.Timestamp.now() - start
    logger.info(f"general flag updated with distribution filter in {duration}s")

    return df


def create_covid_flag(
    df: DataFrame,
    granularity: str,
    lvl2: str,
    covid_flag_1: bool,
    covid_flag_1_df: DataFrame,
    covid_flag_2: bool,
    covid_flag_2_df: DataFrame,
    start_covid_period: int,
    covid_3_exclud_val: List[str],
    covid_3_period_start: int,
    covid_3_period_end: int,
    covid_4_exclud_val: List[str],
    covid_4_period_start: int,
    covid_4_period_end: int,
) -> DataFrame:
    """
    This function creates covid flags
    Args:
        df: spark dataframe
        granularity: granularity of the dataset ('week' or 'month')
        lvl2: level 2 in params
        covid_flag_1: boolean whether to create covid_flag_1
        covid_flag_1_df: df with specifications of covid filters for flag_1
        covid_flag_2: boolean whether to create covid_flag_2
        covid_flag_2_df: df with specifications of covid filters for flag_2
        start_covid_period: the year_week or year_month (depending on granularity of the data) when covid period started
        covid_3_exclud_val: level filters to exclude from covid_flag_3 calculation
        covid_3_period_start: the year_week or year_month (depending on granularity of the data) when covid_flag_3 period started
        covid_3_period_end: the year_week or year_month (depending on granularity of the data) when covid_flag_3 period ended
        covid_4_exclud_val: level filters to exclude from covid_flag_4 calculation
        covid_4_period_start: the year_week or year_month (depending on granularity of the data) when covid_flag_4 period started
        covid_4_period_end: the year_week or year_month (depending on granularity of the data) when covid_flag_4 period ended
    Returns:
        df: spark dataframe with covid flag columns
    """
    start = pd.Timestamp.now()

    df = df.withColumn(f"year_{granularity}", F.col(f"year_{granularity}").cast("int"))
    # create covid_period
    df = df.withColumn(
        "covid_period", F.when((F.col(f"year_{granularity}") > start_covid_period), 1).otherwise(0)
    )

    # create covid flag 1
    if covid_flag_1:
        # convert lvl2 column to match case input data
        covid_flag_1_df = covid_flag_1_df.withColumn(lvl2, F.upper(F.col(lvl2)))
        # join covid table with input data
        df = df.join(covid_flag_1_df, [lvl2, f"year_{granularity}"], "left")
        df = df.withColumn(
            "covid_flag_1",
            F.when((F.col("covid_flag_1").isNull() == True), 0).otherwise(F.col("covid_flag_1")),
        )

    else:
        logger.warning(
            "File 'input_covid_flag_1' from data_prep_catalog is unavailable"
            + "\nPlease upload required csv and restart the pipeline. Otherwise covid_flag_1 will not be created.",
        )

    # create covid flag 2
    if covid_flag_2:
        # convert lvl2 column to match case in input data
        covid_flag_2_df = covid_flag_2_df.withColumn(lvl2, F.upper(F.col(lvl2)))
        # join covid table with input data
        df = df.join(covid_flag_2_df, [lvl2, f"year_{granularity}"], "left")
        df = df.withColumn(
            "covid_flag_2",
            F.when((F.col("covid_flag_2").isNull() == True), 0).otherwise(F.col("covid_flag_2")),
        )
    else:
        logger.warning(
            "File 'input_covid_flag_2' from data_prep_catalog is unavailable"
            + "\nPlease upload required csv and restart the pipeline. Otherwise covid_flag_2 will not be created.",
        )
    # create covid flag 3
    df = df.withColumn(
        "covid_flag_3",
        F.when(
            (~F.col(lvl2).isin(covid_3_exclud_val))
            & (F.col(f"year_{granularity}") >= covid_3_period_start)
            & (F.col(f"year_{granularity}") <= covid_3_period_end),
            1,
        ).otherwise(0),
    )

    # create covid flag 4
    df = df.withColumn(
        "covid_flag_4",
        F.when(
            (~F.col(lvl2).isin(covid_4_exclud_val))
            & (F.col(f"year_{granularity}") >= covid_4_period_start)
            & (F.col(f"year_{granularity}") <= covid_4_period_end),
            1,
        ).otherwise(0),
    )

    duration = pd.Timestamp.now() - start
    logger.info(f"covid flags created in {duration}s")

    return df


def create_promo_acv(df: DataFrame, params: dict) -> Tuple[DataFrame, dict]:
    """
    This function adds the promo acv variables
    Args:
        df : spark dataframe
        params : parameters dictionary
    Returns:
        df : spark dataframe with promo acv columns added to the dataframe
        params : updated params dictionary
    """
    start = pd.Timestamp.now()

    volume_promo_var = params["ewb"]["data_management"]["volume_promo_var"]
    volume_ratio = params["ewb"]["data_management"]["volume_ratio"]
    promo_flag = params["ewb"]["data_management"]["promo_flag"]
    granularity = params["ewb"]["data_management"]["granularity"]
    lvl2 = params["ewb"]["data_management"]["levels"]["lvl2"]
    volume = params["ewb"]["data_management"]["volume"]

    # create acv promo columns
    df = df.withColumn(
        "share_on_promo",
        F.when(
            (F.col("general_flag") == "GOOD"),
            F.col(volume_promo_var) / F.col(volume),
        ).otherwise(None),
    )
    df = df.withColumn(
        "sop_bucket",
        F.when((df.share_on_promo < 0.2) & (F.col("general_flag") == "GOOD"), "<20").otherwise(
            F.when(
                (df.share_on_promo <= 0.5)
                & (df.share_on_promo > 0.2)
                & (F.col("general_flag") == "GOOD"),
                "20-50",
            ).otherwise(
                F.when(
                    (df.share_on_promo > 0.5) & (F.col("general_flag") == "GOOD"), ">50"
                ).otherwise(None)
            )
        ),
    )

    grouping_cols = list(filter(None, [lvl2] + [f"year_{granularity}"] + ["general_flag"]))
    w = Window.partitionBy(grouping_cols)
    df = df.withColumn(
        "total_promo_volume",
        F.when((F.col("general_flag") == "GOOD"), F.sum(volume_promo_var).over(w)).otherwise(None),
    )
    df = df.withColumn(
        "promo_acv",
        F.when(
            (F.col("general_flag") == "GOOD") & (F.col(volume_promo_var) == 0),
            0.001,
        ).otherwise(F.col(volume_promo_var) / F.col("total_promo_volume")),
    )
    df = df.withColumn(
        "log_promo",
        F.when(
            (F.col("general_flag") == "GOOD"),
            F.log("promo_acv"),
        ).otherwise(None),
    )

    # adding "promo" column to the main dataset
    # if bool(promo_flag) == False:
    #    df = df.withColumn(
    #        "promo",
    #        F.when(
    #            (F.col(general_flag_column) == "GOOD"),
    #            0,
    #        ).otherwise(None),
    #    )
    #    df = df.withColumn(
    #        "promo",
    #        F.when(
    #            (F.col(general_flag_column) == "GOOD") & (F.col("share_on_promo") > volume_ratio),
    #            1,
    #        ).otherwise(F.col("promo")),
    #    )
    #    params["ewb"]["data_management"]["promo_flag"] = "promo"

    duration = pd.Timestamp.now() - start
    logger.info(f"promo acv columns created in {duration}s")

    return df, params


def pantry_load(df, lvl1, lvl2, lvl3, volume, sorting_cols):
    """
    This function adds pantry loading columns
    Args:
        df: spark dataframe
        lvl1 : level of aggregation
        lvl2 : level of aggregation
        lvl3 : level of aggregation
        volume : volume column
        sorting_cols : list of columns on which to sort data
    Returns:
        df: spark dataframe with pantry loading columns
    """
    start = pd.Timestamp.now()

    grouping_cols = list(filter(None, [lvl1] + [lvl2] + [lvl3]))

    df = df.withColumn(
        "promo_flag",
        F.when(
            ((F.col("price") / F.col("edlp_price") <= 0.93) & (F.col("acv_any_promo") >= 5))
            | ((F.col("acv_any_promo") >= 25))
            | ((F.col(lvl2) == "Walmart Corp-RMA - Walmart") & (F.col("acv_any_promo") >= 15))
            | ((F.col(lvl2) == "Total US - Conv") & (F.col("price") / F.col("edlp_price") <= 0.93)),
            1,
        ).otherwise(0),
    )

    # create a volume dataframe and join back with input dataframe
    vol_df = (
        df.filter((F.col("general_flag") == "GOOD") & (F.col("promo_flag") == 0))
        .groupBy(grouping_cols)
        .agg(F.mean(volume))
        .withColumnRenamed(f"avg({volume})", "avg_vol")
    )
    df = df.join(vol_df, grouping_cols, "left")

    # sort data by lvl1, lvl2, lvl3, year_granularity
    df = df.sort(sorting_cols, ascending=True)

    for i in range(3):
        df = df.withColumn(
            f"pl{i+1}",
            F.when(
                (F.col("promo_flag") == 0)
                & (
                    F.lag(df["promo_flag"], offset=i + 1).over(
                        Window.partitionBy(grouping_cols).orderBy(sorting_cols)
                    )
                    == 1
                )
                & (F.col(volume) < F.col("avg_vol")),
                1,
            ).otherwise(0),
        )

    duration = pd.Timestamp.now() - start
    logger.info(f"pantry loading done in {duration}s")

    return df


def pantry_load_dynamic(df, lvl1, lvl2, lvl3, volume, sorting_cols, pantry_loading):
    """
    Add pantry loading columns
    Args:
        df: pyspark dataframe with raw modeling data
        params : parameters dictionary

    Returns:
        return: pyspark dataframe with pantry loading columns
    """
    start = pd.Timestamp.now()

    grouping_cols = list(filter(None, [lvl1] + [lvl2] + [lvl3]))


    # # create a volume dataframe and join back with input dataframe
    # avg_vol_window = Window.partitionBy(grouping_cols)
    # vol_df = (
    #     df.filter((F.col('general_flag') == "GOOD") & (F.col("promo_flag") == 0))
    #     .groupBy(grouping_cols)
    #     .agg(F.mean(volume))
    #     .withColumnRenamed(f"avg({volume})", "avg_vol")
    # )
    # df = df.join(vol_df, grouping_cols, "left")

    # sort data by lvl1, lvl2, lvl3, year_granularity
    df = df.sort(sorting_cols, ascending=True)

    for i in range(len(pantry_loading)):
        df = df.withColumn(
            f"pl{i+1}",
            F.when(
                F.lag(df["promo_flag"], offset=i + 1).over(
                    Window.partitionBy(grouping_cols).orderBy(sorting_cols)
                )
                == 1,
                1,
            ).otherwise(0),
        ).fillna(0, subset=[f"pl{i+1}"])

    duration = pd.Timestamp.now() - start
    logger.info(f"pantry loading done in {duration}s")

    return df
