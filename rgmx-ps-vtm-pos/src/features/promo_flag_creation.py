###Promo Threshold and Promo Flag Creation

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import count_distinct
from pyspark.sql.dataframe import DataFrame
import pandas as pd
import numpy as np
from pyspark.sql.functions import lit 
from src.utils.logger import logger
from typing import List
from pyspark.sql import *
from src.utils.utils import *


def create_promo_flag(
    input_df: DataFrame,
    outlier_retailers: List[str],
    parameters: dict
) -> DataFrame:
    """
    Adds promo flag column
    Args:
        df: pyspark dataframe
        outlier_retailers: retailers for which acv is not available (usually seen in EPOS retailers)
        params: parameters dictionary

    Returns:
        return: pyspark dataframe with promo flag column & promo threshold generated for each retailer 
    """
    start = pd.Timestamp.now()
    
    volume = parameters["ewb"]["data_management"]["volume"]
    granularity = parameters["ewb"]["data_management"]["granularity"]
    lvl2 = parameters["ewb"]["data_management"]["levels"]["lvl2"]
    
    covid_start = 0
    covid_end = 0
    
    if granularity == "week":
        covid_start = 202007
        covid_end = 202041
    elif granularity == "month":
        covid_start = 202002
        covid_end = 202010
    
    spark = SparkSession.builder.getOrCreate()
    
    #Promo Share at retailer level based upon promo sales volume
    
    df_intermediate = input_df.groupBy([lvl2]).agg(F.sum(volume).alias(volume),
                                                   F.sum("promo_sales_volume").alias("promo_sales_volume"))
    
    df_intermediate = df_intermediate.withColumn(
        "promo_perc",
        F.col("promo_sales_volume")/F.col(volume)
    )
    
    #Storing promo shares in a dictionary
    
    dict_1 = {row[lvl2]:row['promo_perc'] for row in df_intermediate.collect()}
    
    
    
    #Creating dataframe with calculated promo shares by calling promo_flag_creation function
    
    data = pd.DataFrame(columns=[lvl2,'discount_par','acv_par','promo_share_calc', 'promo_share_diff'])
    
    level_2 = []
    discount = []
    acv = []
    promo_share = []
    promo_share_diff = []
    
    new_df = convert_sparkdf_to_pandasdf(input_df)
    
    for level in dict_1:
        if level not in outlier_retailers:
            for discount_par in np.arange(0.8, 1.0, 0.05):
                for acv_par in range(0, 35, 5):
                    level_2.append(level)
                    discount.append(discount_par)
                    acv.append(acv_par)
                    promo_share.append(promo_flag_creation(new_df, discount_par, acv_par, level, parameters))
                    promo_share_diff.append(abs(dict_1[level] - promo_flag_creation(new_df, discount_par, acv_par, level, parameters)))
                    
            
    
    data[lvl2] = level_2
    data["discount_par"] = discount
    data["acv_par"] = acv
    data["promo_share_calc"] = promo_share
    data["promo_share_diff"] = promo_share_diff
    
    
    #Finding ideal thresholds where calculated promo shares and actual promo shares are similar

    data_1 = data.loc[data.groupby(lvl2)['promo_share_diff'].idxmin(), [lvl2,'discount_par', 'acv_par', 'promo_share_calc',
                                                                        'promo_share_diff']]
    
    
    duration = pd.Timestamp.now() - start
    logger.info(f"promo threshold created in {duration}s")
    
    
    # Creating dictionary out of identified thresholds
    
    data_1 = spark.createDataFrame(data_1)
    dict_discount = {row[lvl2]:row['discount_par'] for row in data_1.collect()}
    dict_acv = {row[lvl2]:row['acv_par'] for row in data_1.collect()}
    

    #Adding promo flag to data
    
    input_df = input_df.withColumn('promo_flag', lit(0))
    
    #Promo flag for retailers with promo acv data, during covid period
    
    for level in dict_1:
        if level not in outlier_retailers:
            input_df = input_df.withColumn(
                "promo_flag",
                F.when(
                    ((F.col(lvl2) == level)
                     &
                     ((F.col(f'year_{granularity}') > covid_start) & (F.col(f'year_{granularity}') < covid_end))
                     &
                     (((F.col("price") / F.col("edlp_price") < dict_discount[level])) | (F.col("acv_any_promo") > dict_acv[level]))
                    ),
                    1,
                ).otherwise(input_df["promo_flag"]),
            )
    
    #Promo flag for retailers with promo acv data, not during covid period
    
    for level in dict_1:
        if level not in outlier_retailers:
            input_df = input_df.withColumn(
                "promo_flag",
                F.when(
                    ((F.col(lvl2) == level)
                     &
                     ((F.col(f'year_{granularity}') <= covid_start) | (F.col(f'year_{granularity}') >= covid_end))
                     &
                     (((F.col("price") / F.col("edlp_price") < dict_discount[level]) & (F.col("acv_any_promo") > 5)) |
                      (F.col("acv_any_promo") > dict_acv[level]))
                    ),
                    1,
                ).otherwise(input_df["promo_flag"]),
            )
   

    #Promo flag for retailers with no promo acv data
    
    for level in outlier_retailers:
        input_df = input_df.withColumn(
            "promo_flag",
            F.when(
                ((F.col(lvl2) == level)
                 &
                 (((F.col("price") / F.col("edlp_price") < 0.95)) & (F.col("peak_flag") == 1))
                ),
                1,
            ).otherwise(input_df["promo_flag"]),
        )
    
    input_df = input_df.withColumn(
        "promo_flag_outlier_retailers",
        F.when(
            ((F.col(lvl2).isin(outlier_retailers))
            ),
            input_df["promo_flag"],
        ).otherwise(0),
    )
    
    duration = pd.Timestamp.now() - start
    logger.info(f"promo flag done in {duration}s")
    
    return input_df, data_1


#Defining a function to create promo flags at retailer level based upon discount & acv threshold
    
def promo_flag_creation(df, discount_threshold, acv_threshold, level, parameters):
    """
    This function adds promo share
    Args:
    df: pandas dataframe
    discount_threshold : edlp based discount threshold
    acv_threshold : promo acv threshold
    level : retailer
    """
    lvl2 = parameters["ewb"]["data_management"]["levels"]["lvl2"]
    volume = parameters["ewb"]["data_management"]["volume"]
    granularity = parameters["ewb"]["data_management"]["granularity"]
    
    covid_start = 0
    covid_end = 0
    
    if granularity == "week":
        covid_start = 202007
        covid_end = 202041
    elif granularity == "month":
        covid_start = 202002
        covid_end = 202010
    
    df = df[(df[lvl2] == level)]
    df = df.assign(promo_flag_calc = 0)
    
    df["promo_flag_calc"] = np.where(
        ((df[f'year_{granularity}'] > covid_start) & (df[f'year_{granularity}'] < covid_end))
        &
        ((df['price'] / df['edlp_price'] < discount_threshold) | (df['acv_any_promo'] > acv_threshold))
        ,
        1,
        df["promo_flag_calc"]
    )
    
    df["promo_flag_calc"] = np.where(
        ((df[f'year_{granularity}'] <= covid_start) | (df[f'year_{granularity}'] >= covid_end))
        &
        (((df['price'] / df['edlp_price'] < discount_threshold) & (df['acv_any_promo'] > 5)) |
         (df['acv_any_promo'] > acv_threshold))
        ,
        1,
        df["promo_flag_calc"]
    )
    
    df_1 = df[(df["promo_flag_calc"] == 1)]
    
    calc_promo_share = df_1[volume].sum()/df[volume].sum()
    
    
    return calc_promo_share

