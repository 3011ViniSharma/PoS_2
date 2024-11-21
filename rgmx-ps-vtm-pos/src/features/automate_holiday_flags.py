from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import count_distinct
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.window import Window
import sys
from itertools import chain
from typing import Tuple
from typing import List
from src.utils.logger import logger
from pyspark.sql.functions import concat, col, lit

def automate_holiday_flags(
    input_df: DataFrame,
    regional_holiday_list: DataFrame,
    level: dict,
    eval_years: List[int],
    peak_percentage_lmt: int,
    dip_percentage_lmt: int,
    parameters: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function performs data preparation
    Args:
        input_df : data prep file
        regional_holiday_list : weekly holiday list for a region
        level : holidays aggregation level (retailer or category)
        eval_years : years in the dataframe
        peak_percentage_lmt : threshold to create peak flag
        dip_percentage_lmt : threshold to create dip flag
        parameters : parameter dict
    Returns:
        raw_loess_all : updated data prep with new holiday columns appended
    """
    volume = parameters["ewb"]["data_management"]["volume"]
    granularity = parameters["ewb"]["data_management"]["granularity"]
    # Create a spark session
    spark = SparkSession.builder.getOrCreate()
    regional_holiday_list = spark.createDataFrame(regional_holiday_list)
    # Creating aggregated data at category/retailer level
    df_intermediate = input_df.groupBy([f'year_{granularity}', 'year', f'{granularity}',
                                        level]).agg(F.sum(volume).alias(volume))
    
    # # Joining identified weeks to pre-existing holidays list
    
    df_intermediate = df_intermediate.join(regional_holiday_list, f"year_{granularity}", "left")
    
    # Creating peak flags on the aggregated data
    start = pd.Timestamp.now()
    
    if bool(eval_years):
        aggregated_df = get_peak_flag(
            df_intermediate, eval_years, peak_percentage_lmt, volume, level
        )
        
    # Flag cases with holiday consistency in peaks flags
    aggregated_df_2 = aggregated_df.groupBy(["holiday_description", level]).agg(F.sum('peak_flag').alias('holiday_sum'))
    aggregated_df_3 = aggregated_df_2.withColumn(
        "holiday_flag",
        F.when((F.col("holiday_sum") > 2 ), "YOY-Peak").otherwise(
            None
        ),
    )
    
    aggregated_df_3 = aggregated_df_3.where(F.col("holiday_flag") == "YOY-Peak")
    aggregated_df_4 = aggregated_df.join(aggregated_df_3, ["holiday_description", level], "left")
    aggregated_df_4 = aggregated_df_4.where(F.col("holiday_flag") == "YOY-Peak")
    aggregated_df_4 = aggregated_df_4[[f'year_{granularity}', f"{granularity}", 'holiday_flag', level, "holiday_description"]]
    
    # Defining holiday flag
    aggregated_df_4 = aggregated_df_4.withColumn("holiday_flag",concat(lit("Peak_Holiday_"), col("holiday_description")))
    aggregated_df_4 = aggregated_df_4.where(F.col("holiday_description").isNotNull())
    aggregated_df_4 = aggregated_df_4.sort(['holiday_flag', f'year_{granularity}', level], ascending=True)
    
    # Creating a dictionary out of identified holidays
    
    df1 = aggregated_df_4.groupBy("holiday_flag").agg(F.collect_list(f"year_{granularity}").alias(f"year_{granularity}")).orderBy("holiday_flag")
    dict_1 = {row['holiday_flag']:row[f'year_{granularity}'] for row in df1.collect()}
    
    df2 = aggregated_df_4.groupBy("holiday_flag").agg(F.collect_list(level).alias(level)).orderBy("holiday_flag")
    dict_2 = {row['holiday_flag']:row[level] for row in df2.collect()}
    
    # Joining to data prep file
    
    for holiday in dict_1:
        if len(dict_1[holiday]) > 2:
            if holiday in dict_2:
                input_df = input_df.withColumn(
                    holiday,
                    F.when(
                        (F.col(f"year_{granularity}").isin(dict_1[holiday])) & (F.col(level).isin(dict_2[holiday])),
                        1).otherwise(0),
                )
            
    duration = pd.Timestamp.now() - start
    logger.info(f"peak holiday flags done in {duration}s")        
    
    ### Holidays with YOY dips
    
    # Creating dip flags on the aggregated data
    start = pd.Timestamp.now()
    
    if bool(eval_years):
        aggregated_df = get_dip_flag(
            df_intermediate, eval_years, dip_percentage_lmt, volume, level
        )
        
    # Flag cases with holiday consistency in peaks flags    
    aggregated_df_2 = aggregated_df.groupBy(["holiday_description", level]).agg(F.sum('dip_flag').alias('holiday_sum'))
    aggregated_df_3 = aggregated_df_2.withColumn(
        "holiday_flag",
        F.when((F.col("holiday_sum") > 2), "YOY-dip").otherwise(   
            None
        ),
    )
    
    aggregated_df_3 = aggregated_df_3.where(F.col("holiday_flag") == "YOY-dip")
    aggregated_df_4 = aggregated_df.join(aggregated_df_3, ["holiday_description", level], "left")
    aggregated_df_4 = aggregated_df_4.where(F.col("holiday_flag") == "YOY-dip")
    
    aggregated_df_4 = aggregated_df_4[[f'year_{granularity}', f"{granularity}", 'holiday_flag', level, "holiday_description"]]
    
    # Defining holiday flag
    aggregated_df_4 = aggregated_df_4.withColumn("holiday_flag",concat(lit("Dip_Holiday_"), col("holiday_description")))
    aggregated_df_4 = aggregated_df_4.where(F.col("holiday_description").isNotNull())
    aggregated_df_4 = aggregated_df_4.sort(['holiday_flag', f'year_{granularity}', level], ascending=True)
    
    # Creating a dictionary out of identified holidays
    
    df1 = aggregated_df_4.groupBy("holiday_flag").agg(F.collect_list(f"year_{granularity}").alias(f"year_{granularity}")).orderBy("holiday_flag")
    dict_1 = {row['holiday_flag']:row[f'year_{granularity}'] for row in df1.collect()}
    
    df2 = aggregated_df_4.groupBy("holiday_flag").agg(F.collect_list(level).alias(level)).orderBy("holiday_flag")
    dict_2 = {row['holiday_flag']:row[level] for row in df2.collect()}
    
    # Joining to data prep file
    
    for holiday in dict_1:
        if len(dict_1[holiday]) > 2:
            if holiday in dict_2:
                input_df = input_df.withColumn(
                    holiday,
                    F.when(
                        (F.col(f"year_{granularity}").isin(dict_1[holiday])) & (F.col(level).isin(dict_2[holiday])),
                        1).otherwise(0),
                )
            
    duration = pd.Timestamp.now() - start
    logger.info(f"dip holiday flags done in {duration}s")
    return input_df
    
    
# To Capture YOY peaks

def get_year_volume(
    df: DataFrame, peak_year: int, volume: str,  level: str
) -> DataFrame:
    """
    Helper function to get mean volume for one year
    Args:
        df: spark dataframe
        peak_year : year to evaluate
        volume : volume column
    Returns:
        df: spark dataframe with year volume calculated
    """
    grouping_cols = list(filter(None, ["year"] + [level]))
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
    peak_years: List[int],
    peak_percentage_lmt: int,
    volume: str,
    level: str
) -> DataFrame:
    """
    This function creates peak flags to control for abrupt movements in volume if peak_years parameter is defined
    Args:
        df: spark dataframe
        peak_years : list of years to evaluate
        peak_percentage_lmt: threshold value for peak
        volume : volume column
    Returns:
        df: spark dataframe with peak flags columns for specified years
    """
    start = pd.Timestamp.now()
    df = df.withColumn("peak_flag", F.lit(0))

    for peak_year in peak_years:
        df = get_year_volume(df, peak_year, volume, level)
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
    #logger.info(f"peak flags created in {duration}s")
    return df



# To Capture YOY dips

def get_year_volume(
    df: DataFrame, dip_year: int, volume: str,  level: str
) -> DataFrame:
    """
    Helper function to get mean volume for one year
    Args:
        df: spark dataframe
        dip_year : year to evaluate
        volume : volume column
    Returns:
        df: spark dataframe with year volume calculated
    """
    grouping_cols = list(filter(None, ["year"] + [level]))
    return df.withColumn(
        f"vol_{str(dip_year)[-2:]}",
        F.when(
            F.col("year") == dip_year, F.mean(volume).over(Window.partitionBy(grouping_cols))
        ).otherwise(0),
    ).fillna(0)


def get_dip_percent(df: DataFrame, dip_year: int, volume: str) -> DataFrame:
    """
    Helper function to calculate dips in percentage for one year
    Args:
        df: spark dataframe
        dip_year : year to evaluate
        volume : volume column
    Returns:
        df: spark dataframe with year dip calculated
    """
    df = df.withColumn(
        f"dip{str(dip_year)[-2:]}",
        F.when(
            (F.col("year") == dip_year),
            (
                ((F.col(volume) - F.col(f"vol_{str(dip_year)[-2:]}")) * 100)
                / F.col(f"vol_{str(dip_year)[-2:]}")
            ),
        ).otherwise(0),
    ).fillna(0)
    return df.withColumn(
        f"dip{str(dip_year)[-2:]}", F.round(F.col(f"dip{str(dip_year)[-2:]}"), 7)
    )

def get_dip_flag(
    df: DataFrame,
    dip_years: List[int],
    dip_percentage_lmt: int,
    volume: str,
    level: str
) -> DataFrame:
    """
    This function creates dip flags to control for abrupt movements in volume if dip_years parameter is defined
    Args:
        df: spark dataframe
        dip_years : list of years to evaluate
        dip_percentage_lmt: threshold value for dip
        volume : volume column
    Returns:
        df: spark dataframe with dip flags columns for specified years
    """
    start = pd.Timestamp.now()
    df = df.withColumn("dip_flag", F.lit(0))

    for dip_year in dip_years:
        df = get_year_volume(df, dip_year, volume, level)
        df = get_dip_percent(df, dip_year, volume)
        df = df.withColumn(
            f"dip_flag{str(dip_year)[-2:]}",
            F.when(F.col(f"dip{str(dip_year)[-2:]}") < dip_percentage_lmt, 1).otherwise(0),
        ).fillna(0)
        # assign dip flag = 1 if any year dip flag is 1
        df = df.withColumn(
            "dip_flag",
            F.when(F.col(f"dip_flag{str(dip_year)[-2:]}") == 1, 1).otherwise(F.col("dip_flag")),
        )
    duration = pd.Timestamp.now() - start
    #logger.info(f"dip flags created in {duration}s")
    return df    