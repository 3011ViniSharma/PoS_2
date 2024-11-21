# -*- coding: utf-8 -*-
import sys

from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame

from src.data.data_load import *
from src.utils.config import *
from src.utils.logger import *
from src.utils.utils import *
from src.utils.version_control import *


def date_columns_creation(input_df: DataFrame, params: dict, which_df: str) -> DataFrame:
    """
    This function creates date related columns, i.e. full_date, monthname, year, etc.
    Args:
        input_df: spark dataframe
        params : name of the date column
        which_df: string identifying which additional data is processed
    Returns:
        input_df : spark dataframe with date columns added
    """
    date_col_name = params["additional_input_params"][which_df]["add_date_col_name"]

    input_df = input_df.withColumn(date_col_name, F.to_date(F.col(date_col_name)))
    input_df = input_df.withColumn("week", F.weekofyear(F.col(date_col_name)))
    input_df = input_df.withColumn(
        "week",
        F.when(F.length(input_df["week"]) == 1, F.concat(F.lit("0"), input_df["week"])).otherwise(
            input_df["week"]
        ),
    )
    input_df = input_df.withColumn("month", F.month(F.col(date_col_name)))
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
    return input_df


def aggregate_built_in(
    input_df: DataFrame,
    transformation: str,
    grouping_cols: List[str],
    columns: List[str],
    aliases: List[str],
    granularity: str,
) -> DataFrame:
    """
    This function aggregates dataframe by grouping_cols and performs a given transformation on a specific set of columns
    Args:
        input_df: spark dataframe
        transformation: transformation to apply
        grouping_cols : list of columns to aggregate by
        columns : list of colums to transform
        aliases: list of aliases
    Returns:
        out_df : spark dataframe with columns aggregated
    """
    sorting_cols = grouping_cols + [f"year_{granularity}"]
    out_df = input_df.select(grouping_cols).distinct()
    for column, alias in zip(columns, aliases):
        spark_expr = f"{transformation}({column})"
        input_df = input_df.orderBy(sorting_cols)
        df_intermediate = input_df.groupBy(grouping_cols).agg(F.expr(spark_expr).alias(alias))
        out_df = out_df.join(df_intermediate, grouping_cols, "inner")

    return out_df


def aggregate_wtd_average(
    input_df: DataFrame,
    transformation: str,
    grouping_cols: List[str],
    columns: List[str],
    aliases: List[str],
    weight_col: str,
) -> DataFrame:
    """
    This function aggregates dataframe by grouping_cols and performs a wiighted average transformation on a specific set of columns
    Args:
        input_df: spark dataframe
        transformation: transformation to apply
        grouping_cols : list of columns to aggregate by
        columns : list of colums to transform
        aliases: list of aliases
        weight_col : column that we need to weight by
    Returns:
        out_df : spark dataframe with columns aggregated
    """
    out_df = input_df.select(grouping_cols).distinct()
    for column, alias in zip(columns, aliases):
        df_intermediate = (
            input_df.withColumn("numerator", F.col(column) * F.col(weight_col))
            .groupBy(grouping_cols)
            .agg(
                F.sum(weight_col).alias("total_weight"),
                F.sum("numerator").alias("weighted_" + column),
            )
            .withColumn(alias, F.col("weighted_" + column) / F.col("total_weight"))
        )
        columns_to_drop = [i for i in df_intermediate.columns if i.startswith("weighted_")] + [
            "numerator",
            "total_weight",
        ]
        df_intermediate = df_intermediate.drop(*columns_to_drop)
        out_df = out_df.join(df_intermediate, grouping_cols, "inner")
    return out_df


def aggregate_data(standardised_additional: DataFrame, params: dict, which_df: str) -> DataFrame:
    """
    This function aggregates additional input dataset to a given level and applies specified transformations
    Args:
        standardised_additional: standardised additional data
        params: parameters dictionary
        which_df: string identifying which additional data is processed
    Returns:
        out_df: aggregated dataframe
    """
    add_granularity = params["additional_input_params"][which_df]["add_granularity"]
    add_time_granularity = params["additional_input_params"][which_df]["add_time_granularity"]
    transformations = params["additional_input_params"][which_df]["aggregation_params"]

    granularity = params["ewb"]["data_management"]["granularity"]
    colnames_to_dummy = params["additional_input_params"][which_df]["colnames_to_dummy"]
    key_columns = add_granularity + ["year_" + granularity]

    aliases = [
        d["aliases"] for d in params["additional_input_params"][which_df]["aggregation_params"]
    ]
    flat_aliases = [item for sublist in aliases for item in sublist]
    parsed_key_for_dummy = [col for col in colnames_to_dummy if col not in flat_aliases]
    aggregated_df = standardised_additional.select(key_columns + parsed_key_for_dummy).distinct()
    for transformation in transformations:
        func = transformation["transformation"]
        columns = transformation["columns"]
        aliases = transformation["aliases"]
        if func == "wtd_average":
            weight_by = transformation["weight_by"]
            out_df = aggregate_wtd_average(
                standardised_additional,
                func,
                key_columns + parsed_key_for_dummy,
                columns,
                aliases,
                weight_by,
            )
            aggregated_df = aggregated_df.join(out_df, key_columns + parsed_key_for_dummy, "inner")
        else:
            out_df = aggregate_built_in(
                standardised_additional,
                func,
                key_columns + parsed_key_for_dummy,
                columns,
                aliases,
                granularity,
            )
            aggregated_df = aggregated_df.join(out_df, key_columns + parsed_key_for_dummy, "inner")
    check_unique_values(standardised_additional, parsed_key_for_dummy, key_columns)
    return aggregated_df.fillna(0)


def dummy_encode(additional_df: DataFrame, params: dict, which_df: str) -> DataFrame:
    """
    This function creates dummy vars for categorical columns
    Args:
        additional_df: standardised additional data
        params: parameters dictionary
        which_df: string identifying which additional data is processed
    Returns:
        additional_df: dataframe with dummies
    """
    colnames_to_dummy = params["additional_input_params"][which_df]["colnames_to_dummy"]
    for colname in colnames_to_dummy:
        if dict(additional_df.dtypes)[colname] == "string":
            categories = additional_df.select(colname).distinct().rdd.flatMap(lambda x: x).collect()
            exprs = [
                F.when(F.col(colname) == category, 1).otherwise(0).alias(colname + "_" + category)
                for category in categories
            ]
            additional_df = additional_df.select("*", *exprs)
            logger.info(f"dummy encoded {colname}")
            dummy_colnames = [colname + "_" + cat for cat in categories]
        else:
            logger.info(
                f"{colname} is of type {dict(additional_df.dtypes)[colname]}. Can't create dummy. SKIPPING."
            )
    return additional_df, dummy_colnames


def check_unique_values(additional_df, colnames_to_dummy: List, key_columns: List):
    """
    Helper function to perform create dummy vars for categorical columns
    Args:
        additional_df: standardised additional data
        colnames_to_dummy: list of categorical columns to perform dummy transformation
        key_columns: list of key columns
    """
    if (
        additional_df.groupBy(key_columns).sum().count()
        != additional_df.groupBy(key_columns + colnames_to_dummy).sum().count()
    ):
        logger.info(
            "WARNING: colnames_to_dummy contain columns that have more than one unique value per group. This will cause incorrect merge. \n Make sure each column has only one unique value per group manually or by adding it to aggregation or remove columns from colnames_to_dummy. \n Stopping the pipeline"
        )
        sys.exit()
    else:
        logger.info(
            "All columns in colnames_to_dummy contain only one unique value per group. That is good"
        )


def perform_merge(
    input_data: DataFrame,
    standardised_additional: DataFrame,
    colnames_to_keep: List,
    key_columns: List,
    colnames_to_dummy: List,
) -> DataFrame:
    """
    Helper function to perform merge
    Args:
        standardised_df: input data after data prep
        additional_df: standardised additional data
        colnames_to_keep: list of columns to keep after merge
        key_columns: list of key columns
        colnames_to_dummy: list of key columns to dummy transform
    Returns:
        merged_df: merged dataframe
    """
    if colnames_to_keep == "all":
        merged_df = input_data.join(standardised_additional, key_columns, "left")
    elif len(colnames_to_keep) > 0:
        merged_df = input_data.join(
            standardised_additional[colnames_to_keep + key_columns + colnames_to_dummy],
            key_columns,
            "left",
        )
    else:
        logger.info(
            "Looks like your colnames_to_keep parameter and aggregation_params are empty. Please specify either one. Stopping the pipeline"
        )
        sys.exit()
    return merged_df


def merge_data(
    input_data: DataFrame, standardised_additional: DataFrame, params: dict, which_df: str
) -> DataFrame:
    """
    This function merges additional data to standardised data. If additional data is more granular, it aggregates
    additional input dataset to a given level and applies specified transformations
    Args:
        standardised_df: input data after data prep
        additional_df: standardised additional data
        params: parameters dictionary
        which_df: string identifying which additional data is processed
    Returns:
        merged_df: aggregated dataframe
    """
    lvl1 = params["ewb"]["data_management"]["levels"]["lvl1"]
    lvl2 = params["ewb"]["data_management"]["levels"]["lvl2"]
    lvl3 = params["ewb"]["data_management"]["levels"]["lvl3"]
    granularity = params["ewb"]["data_management"]["granularity"]
    value = params["ewb"]["data_management"]["value"]

    add_granularity = params["additional_input_params"][which_df]["add_granularity"]
    add_time_granularity = params["additional_input_params"][which_df]["add_time_granularity"]
    add_date_col_name = params["additional_input_params"][which_df]["add_date_col_name"]

    aggregation_params = params["additional_input_params"][which_df]["aggregation_params"]
    colnames_to_keep = params["additional_input_params"][which_df]["colnames_to_keep"]
    colnames_to_dummy = params["additional_input_params"][which_df]["colnames_to_dummy"]
    granularity_dict = {"day": 1, "week": 2, "month": 3}

    if add_time_granularity == granularity:
        key_columns = [
            col for col in list(filter(None, [lvl1, lvl2, lvl3])) if col in add_granularity
        ] + ["year_" + granularity]
        if len(params["additional_input_params"][which_df]["colnames_to_dummy"]) > 0:
            standardised_additional, dummy_colnames = dummy_encode(
                standardised_additional, params, which_df
            )
        merged_df = perform_merge(
            input_data,
            standardised_additional,
            colnames_to_keep + dummy_colnames,
            key_columns,
            colnames_to_dummy,
        )

    elif granularity_dict[add_time_granularity] < granularity_dict[granularity]:
        aggregated_df = aggregate_data(standardised_additional, params, which_df)
        if len(params["additional_input_params"][which_df]["colnames_to_dummy"]) > 0:
            aggregated_df, dummy_colnames = dummy_encode(aggregated_df, params, which_df)
        key_columns = [
            col for col in list(filter(None, [lvl1, lvl2, lvl3])) if col in add_granularity
        ] + ["year_" + granularity]
        merged_df = input_data.join(aggregated_df, key_columns, "left")

    elif granularity_dict[add_time_granularity] > granularity_dict[granularity]:
        key_columns = [
            col for col in list(filter(None, [lvl1, lvl2, lvl3])) if col in add_granularity
        ] + ["year_" + add_time_granularity]
        if len(params["additional_input_params"][which_df]["colnames_to_dummy"]) > 0:
            standardised_additional, dummy_colnames = dummy_encode(
                standardised_additional, params, which_df
            )
        merged_df = perform_merge(
            input_data,
            standardised_additional,
            colnames_to_keep + dummy_colnames,
            key_columns,
            colnames_to_dummy,
        )

    # asserting the sum of value is as expected to make sure the merge is correct
    try:
        assert (input_data.select(F.ceil(F.sum(value))).collect()[0][0]) == (
            merged_df.select(F.ceil(F.sum(value))).collect()[0][0]
        )
        logger.info("Sum of value is as expected after the merge")
    except:
        logger.info(
            "WARNING: Sum of value is NOT as expected after the merge. Please make sure the granularity parameters are defined correctly"
        )

    # how many rows have new values
    if len(params["additional_input_params"][which_df]["aggregation_params"]) > 0:
        new_col_name = params["additional_input_params"][which_df]["aggregation_params"][0][
            "aliases"
        ][0]
    elif colnames_to_keep != "all":
        new_col_name = colnames_to_keep[0]
    elif colnames_to_keep == "all":
        new_col_name = standardised_additional.columns[0]
    else:
        logger.info(
            "Looks like your colnames_to_keep parameter and aggregation_params are empty. Please specify either one"
        )
    pct_rows = round(
        merged_df.where(F.col(new_col_name).isNotNull()).count() / merged_df.count() * 100
    )
    if pct_rows == 0:
        logger_info(
            'Please check that your merge keys are definied correctly and that the values in the key columns are the same for main and additional data"'
        )
    logger.info(f"{pct_rows}% rows got new values after the merge")
    return merged_df.fillna(0)


def transform_with_custom_function(merged_df: DataFrame, params: dict, which_df: str) -> DataFrame:
    """
    This function applies the transformation specified in data schema to the dataframe
    Args:
        merged_df: spark dataframe
        params: parameters dictionary
        which_df: string identifying which additional data is processed
    Returns:
        formatted_data: spark dataframe
    """
    transformations = params["additional_input_params"][which_df]["custom_transformations"]

    for transformation in transformations:
        try:
            formatted_data = merged_df.withColumn(
                transformation["column_name"], F.expr(transformation["formula"])
            )
            logger.info(
                f"created {transformation['column_name']} with {F.expr(transformation['formula'])} formula"
            )
        except:
            logger.info(
                f"WARNING: {transformation} is a NOT valid expression. \n SKIPPING THIS ONE"
            )
    return formatted_data


def add_new_data(
    additional_data: DataFrame, input_data: DataFrame, params: dict, which_df: str
) -> DataFrame:
    """
    Master function to aggregate and merge additional input and apply custom transformations
    Args:
        additional_data: data prep step output
        df: raw additional data
        df_params: parameters dict for dataframe
        df_format: input format for dataframe
        which_df: string identifying which additional data is processed
    Returns:
        df: merged dataframe
    """
    logger.info("Starting new input processing")
    logger.info(f"Processing {which_df}")
    start = pd.Timestamp.now()
    standardised_additional = standardise_batch(
        additional_data, params["additional_schema"][which_df]
    )
    standardised_additional = date_columns_creation(standardised_additional, params, which_df)
    merged_df = merge_data(input_data, standardised_additional, params, which_df)
    if len(params["additional_input_params"][which_df]["custom_transformations"]) > 0:
        formatted_data = transform_with_custom_function(merged_df, params, which_df)
        duration = pd.Timestamp.now() - start
        logger.info(f"{which_df} processing done in {duration}s")
        return formatted_data
    else:
        duration = pd.Timestamp.now() - start
        logger.info(f"{which_df} processing done in {duration}s")
        return merged_df
