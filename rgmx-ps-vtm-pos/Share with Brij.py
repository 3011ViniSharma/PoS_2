# Databricks notebook source


# COMMAND ----------

#check file name in folders
dbutils.fs.ls('dbfs:/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/00_input/')

# COMMAND ----------

#check file name in folders
dbutils.fs.ls('dbfs:/FileStore/tables/POS_WB_Training/00_input')

# COMMAND ----------

# MAGIC %md
# MAGIC #Load Libraries

# COMMAND ----------

#!pip install --upgrade pip
!pip install bios
!pip install julia
!pip install missingpy
!pip install pygam
!pip install rpy2==3.5.4
!pip install pyspark==3.2.1
!pip install pandas==1.2.4
!pip install treelib

# COMMAND ----------

# MAGIC %md
# MAGIC # General Pipeline For POS Data Prep, Create Structure & Volume Transfer Matrix

# COMMAND ----------

import os
import sys
from pathlib import Path
from pyspark.sql.types import *
from pyspark.sql.functions import *

PROJECT_DIR = Path(Path.cwd()).parents[1] #change 0 to 1 if you have this notebook in a folder
name_of_ewb = 'EWB_POS_WB_v1.1_manual_pull' # econometrics directory
name_of_ps = 'POS_PSVTM_manual_pull' # purchase structure directory rgmx_pos_ps_vtm rgmx_ewb-econometrics_clone
sys.path.append(str(PROJECT_DIR / name_of_ewb )) # append econometrics directory to system directory
sys.path.append(str(PROJECT_DIR / name_of_ps )) # append purchase structure directory to system directory

# COMMAND ----------

# Load created modules 
from src.data.data_load import *
from src.models.loess import *
from src.features.data_prep import *
from src.utils.logger import logger
from src.utils.utils import *
from src.utils.version_control import *
from src.utils.config import *
from src_psvtm.models.create_structure import *
from src_psvtm.models.vtm import *
from src_psvtm.utils import *
from src_psvtm.models.rgm_outputs import *

# COMMAND ----------

from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
rpy2_logger.setLevel(logging.ERROR) 

# COMMAND ----------

## load params from ewb (econometrics work bench)
logger.info("Reading econometrics data catalogs and parameters...")
base_ewb = str(PROJECT_DIR /  name_of_ewb) + "/" 
params_ewb = load_params(base_ewb)

catalog_ewb = load_catalog(
    base_ewb,
    params_ewb["ewb"]["data_management"]["dir_path_spark"],
    params_ewb["ewb"]["data_management"]["dir_path"],
)

# COMMAND ----------

#load parameters from purchase structure directory
logger.info("Reading purchase structure parameters...")
base_ps = (str(PROJECT_DIR / name_of_ps)) + '/' 

params_ps = load_params(base_ps)
params_ps_vtm = params_ps['ps_vtm']

# COMMAND ----------

catalog_ps = load_catalog(
    base_ps,
    params_ewb["ewb"]["data_management"]["dir_path_spark"],
    params_ewb["ewb"]["data_management"]["dir_path"],
)

# COMMAND ----------

#set the ewb parameters and catalog in ps structure for run loess
set_params_catalog_ewb(params_ewb,catalog_ewb) # add additional commentary here

# COMMAND ----------

# MAGIC %md
# MAGIC ### Date Preparation From Econometrics Module

# COMMAND ----------

## Do not run only needed
# read inputs
input_data = load_spark_data(catalog_ewb["input_data_prep"])

#filter data for category - if needed
#input_data = input_data.filter(input_data.CATEGORY_ID == 'ENERGY DRINKS')

columns = StructType([])
covid_flag_1_df = spark.createDataFrame(data = [],
                           schema = columns)
#load_spark_data(catalog["input_covid_flag_1"])
covid_flag_2_df = covid_flag_1_df = spark.createDataFrame(data = [],
                           schema = columns)
#load_spark_data(catalog["input_covid_flag_2"])
logger.info("Running data preprocessing pipeline...")
standardised_df = standardise_batch(input_data, params_ewb["schema"])
# standardised_df = standardised_df.withColumn("acv_any_promo", lit(1)) # can create acv_tpr dummy here
standardised_df = standardised_df.withColumn("acv_tpr", lit(1))

# COMMAND ----------

standardised_df.display(5)

# COMMAND ----------

raw_loess_all, raw_data_good, raw_data_drop, unique_count_df = data_preparation(
    standardised_df,
    params_ewb,
    params_ewb["holidays_config"],
    covid_flag_1_df,
    covid_flag_2_df,
    catalog_ewb,
)

# COMMAND ----------

parameters["ewb"]["data_management"]

# COMMAND ----------

#### ONLY USE WHEN NEEDED ####
### for specific top 95_flag during client specific situation filter out for top 95 flag
# raw_loess_all =raw_loess_all[raw_loess_all['top_95_flag'] == 1].reset_index(drop=True) 

# COMMAND ----------

# save loess as input data
write_obj(raw_loess_all, 
          catalog_ps['input_data']['filepath'],
          catalog_ps['input_data']['filename'], 
          catalog_ps['input_data']['format'] )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Structure Steps

# COMMAND ----------

# Used saved raw loess below
#raw_loess_all = read_obj(catalog_ps['input_data']['filepath'],
#          catalog_ps['input_data']['filename'], 
#          catalog_ps['input_data']['format'] )

# COMMAND ----------

base_loess = raw_loess_all
base_loess.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Data Prep For Create Structure

# COMMAND ----------

# ## CREATE CUSTOM ATTTRIBUTE BASED ON PARAM FILE
data_for_PS=custom_group_columns_df(base_loess, params_ps_vtm)

# COMMAND ----------

write_obj(data_for_PS, 
          catalog_ps['data_for_ps']['filepath'],
          catalog_ps['data_for_ps']['filename'], 
          catalog_ps['data_for_ps']['format'] )

# COMMAND ----------

# Subset for general flag that meet the about the criteria set in data prep as input
good_data_for_PS = data_for_PS[data_for_PS ['general_flag'] == 'GOOD'].reset_index(drop=True) 

# COMMAND ----------

# create fx by subsetting only 'GOOD' data
fx = good_data_for_PS ['general_flag'] == 'GOOD'
fx = fx.reset_index(drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Purchase Structure for Mapped ("GOOD") Data

# COMMAND ----------

#load parameters from purchase structure directory
logger.info("Reading purchase structure parameters...")
base_ps = (str(PROJECT_DIR / name_of_ps)) + '/' 

params_ps = load_params(base_ps)
params_ps_vtm = params_ps['ps_vtm']

# COMMAND ----------

attrlist = np.array(params_ps_vtm["default_vars"]["attrlist"])
attrlist

# COMMAND ----------

# #Optional
# # # add custom attributes
# # good_data_for_PS['product_pack_size_or_total_size'] = good_data_for_PS['product_pack_size_or_total_size'].astype(int)
# # good_data_for_PS['leq_g_600'] = ['<=600mL' if s<=600 else '>600' for s in good_data_for_PS['product_pack_size_or_total_size']]
# # print(good_data_for_PS['leq_g_600'].value_counts())
# good_data_for_PS['lseq_400'] = ['<=400mL' if s<401 else '>400mL' for s in good_data_for_PS['product_unit_size']]
# print(good_data_for_PS['lseq_400'].value_counts())
# # good_data_for_PS['packtype'] = ['LATA' if p=='LATA' else 'BOTELLA' for p in good_data_for_PS['product_attr_4']]
# # print(good_data_for_PS['packtype'].value_counts())

# # # Get the attribute list
# attrlist = np.array(params_ps_vtm["default_vars"]["attrlist"])
# attrlist = np.append(attrlist, np.array(['lseq_400']))
# print(attrlist)

# COMMAND ----------

# print(good_data_for_PS['dr_beckmann_flag'].value_counts())

# COMMAND ----------

# print(good_data_for_PS['top_brand_flag'].value_counts())

# COMMAND ----------

# MAGIC %md 
# MAGIC #Top Level Break

# COMMAND ----------

#function for structure
res = create_structure(attr_df= good_data_for_PS, attrlist=attrlist, filt_idx=fx,nlevels= params_ps_vtm["default_vars"]["nlevels"], params=params_ps_vtm )

# COMMAND ----------

## get purchase structure
PS = res[0]
interim_res = res[1]
PS.drop_duplicates()

# COMMAND ----------

interim_res

# COMMAND ----------

# run attribute summary to fetch indexes at a specific level
# upate fx to filter down the tree levels
op=get_attr_summary(input_df=good_data_for_PS, attrlist=attrlist, input_filt=fx, size_validated=0.045, useshare=True, params=params_ps_vtm)

# COMMAND ----------

#get raw indexes
op[1]

# COMMAND ----------

#get normalized indexes
op[0]

# COMMAND ----------

good_data_for_PS.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/good_data_for_PS.csv")

# COMMAND ----------

# PS.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/PS_Top.csv")
# interim_res.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/interim_res_Top.csv")
PS.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/PS_Top_v1_Full_Attrs.csv")
interim_res.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/interim_res_Top_v1_Full_Attrs.csv")
# op.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/ps_full_data/op_Top.csv")


# COMMAND ----------

display(PS)

# COMMAND ----------

display(interim_res)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Function to force a change tree structure based on user needs

# COMMAND ----------

## get purchase structure
PS = res[0]
interim_res = res[1]
PS.drop_duplicates()

# COMMAND ----------

# Creating Additional Colums in PS Copy Data
PS_ADDED_COLUMS=PS.copy()
PS_ADDED_COLUMS.insert(2, "X3", "-")
PS_ADDED_COLUMS.insert(3, "X4", "-")
PS_ADDED_COLUMS.insert(4, "X5", "-")
PS_ADDED_COLUMS.insert(5, "X6", "-")
PS_ADDED_COLUMS.insert(6, "X7", "-")
PS_ADDED_COLUMS.insert(7, "X8", "-")
PS_ADDED_COLUMS.insert(8, "X9", "-")
PS_ADDED_COLUMS.insert(9, "X10", "-")
PS_ADDED_COLUMS.insert(12, "W3", "-")
PS_ADDED_COLUMS.insert(13, "W4", "-")
PS_ADDED_COLUMS.insert(14, "W5", "-")
PS_ADDED_COLUMS.insert(15, "W6", "-")
PS_ADDED_COLUMS.insert(16, "W7", "-")
PS_ADDED_COLUMS.insert(17, "W8", "-")
PS_ADDED_COLUMS.insert(18, "W9", "-")
PS_ADDED_COLUMS.insert(19, "W10", "-")
PS_ADDED_COLUMS.drop_duplicates()

# COMMAND ----------

PS_ADDED_COLUMS.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/PS_Top_v1_Full_Attrs_with_10_Added_columns.csv")

# COMMAND ----------

# PS_copy = PS.copy()
# interim_res_f = interim_res.copy()

# # PS_f, l_select, node_rows = get_inputs_force_PS(PS_copy, good_data_for_PS)

# PS_f, l_select, node_rows = get_inputs_force_PS(PS_copy, attr_df)

# COMMAND ----------

PS_f_Top_Lvl = PS_ADDED_COLUMS.copy()
interim_res_f_Top_Lvl = interim_res.copy()
PS_f_Top_Lvl, l_select, node_rows = get_inputs_force_PS(PS_f_Top_Lvl, good_data_for_PS,attrlist)


# COMMAND ----------

# PS_f.drop_duplicates()
PS_f_Top_Lvl.drop_duplicates()

# COMMAND ----------

display(PS_f_Top_Lvl)

# COMMAND ----------

l_select

# COMMAND ----------

node_rows

# COMMAND ----------

# MAGIC %md
# MAGIC ###Forcing in condoms

# COMMAND ----------

PS_f_condom_specific = PS_FORCED_Condom_Lubes.copy()
interim_res_condom_specific = INTERIM_RES_FORCED_Condom_Lubes.copy()
PS_f_condom, l_select, node_rows = get_inputs_force_PS(PS_f_condom_specific, good_data_for_PS,attrlist)

# COMMAND ----------

PS_f_condom.drop_duplicates()

# COMMAND ----------

l_select

# COMMAND ----------

node_rows

# COMMAND ----------

# MAGIC %md
# MAGIC ###Forcing in condoms -> durex

# COMMAND ----------

PS_f_condom_durex_specific = PS_FORCED_CONDOMS_ps_between.copy()
interim_res_condom_durex_specific = INTERIM_RES_CONDOMS_ps_between.copy()
PS_f_condom_durex, l_select, node_rows = get_inputs_force_PS(PS_f_condom_durex_specific, good_data_for_PS,attrlist)

# COMMAND ----------

PS_f_condom_durex.drop_duplicates()

# COMMAND ----------

l_select

# COMMAND ----------

node_rows

# COMMAND ----------

# MAGIC %md
# MAGIC ###Forcing in condoms -> durex -> Large 

# COMMAND ----------

PS_f_condom_durex_large_specific = PS_FORCED_CONDOMS_DUREX_ps_between_read_13July.copy()
interim_res_condom_durex_large_specific = INTERIM_RES_CONDOMS_DUREX_ps_between_13July.copy()
PS_f_condom_durex_large, l_select, node_rows = get_inputs_force_PS(PS_f_condom_durex_large_specific, good_data_for_PS_read_13July,attrlist)

# COMMAND ----------

PS_f_condom_durex_large.drop_duplicates()

# COMMAND ----------

l_select

# COMMAND ----------

node_rows

# COMMAND ----------

# MAGIC %md
# MAGIC ###Forcing in condoms -> durex -> Large ->feel initmate

# COMMAND ----------

PS_f_condom_durex_large_feel_inti_specific = PS_FORCED_CONDOMS_DUREX_LARGE_ps_between.copy()
interim_res_condom_durex_large_feel_inti_specific = INTERIM_RES_CONDOMS_DUREX_LARGE_ps_between.copy()
PS_f_condom_durex_large_feel_inti, l_select, node_rows = get_inputs_force_PS(PS_f_condom_durex_large_feel_inti_specific, good_data_for_PS_read_13July,attrlist)

# COMMAND ----------

PS_f_condom_durex_large_feel_inti.drop_duplicates()

# COMMAND ----------

l_select

# COMMAND ----------

node_rows

# COMMAND ----------

# MAGIC %md
# MAGIC ###Forcing in condoms -> durex -> Regu ->

# COMMAND ----------

PS_f_condom_durex_regu_specific = PS_FORCED_CONDOMS_DUREX_LARGE_FEEL_INTIMATE_ps_between.copy()
interim_res_condom_durex_regu_specific = INTERIM_RES_CONDOMS_DUREX_LARGE_FEEL_INTIMATE_ps_between.copy()
PS_f_condom_durex_regu, l_select, node_rows = get_inputs_force_PS(PS_f_condom_durex_regu_specific, good_data_for_PS_read_13July,attrlist)

# COMMAND ----------

PS_f_condom_durex_regu.drop_duplicates()

# COMMAND ----------

l_select

# COMMAND ----------

node_rows

# COMMAND ----------

# MAGIC %md
# MAGIC ###Forcing in Lubes & Gel

# COMMAND ----------

PS_f_lubes_specific = PS_FORCED_CONDOMS_DUREX_ps_between_read_14July.copy()
interim_res_lubes_specific = INTERIM_RES_CONDOMS_DUREX_ps_between_14July.copy()
PS_f_lubes, l_select, node_rows = get_inputs_force_PS(PS_f_lubes_specific, good_data_for_PS_read_14July,attrlist)

# COMMAND ----------

PS_f_lubes.drop_duplicates()

# COMMAND ----------

l_select

# COMMAND ----------

node_rows

# COMMAND ----------

# MAGIC %md
# MAGIC ###Forcing in Lubes & Gel -> Durex -> Not_Basic

# COMMAND ----------

PS_f_lubes_durex_not_basic_specific = PS_FORCED_LUBES_ps_between.copy()
interim_res_lubes_durex_not_basic_specific = INTERIM_RES_LUBES_ps_between.copy()
PS_f_lubes_durex_not_basic, l_select, node_rows = get_inputs_force_PS(PS_f_lubes_durex_not_basic_specific, good_data_for_PS_read_14July,attrlist)

# COMMAND ----------

PS_f_lubes_durex_not_basic.drop_duplicates()

# COMMAND ----------

l_select

# COMMAND ----------

node_rows

# COMMAND ----------

# MAGIC %md
# MAGIC ###Forcing in Lubes & Gel -> Not_Durex -> Gel

# COMMAND ----------

PS_f_lubes_not_durex_gel_specific = PS_FORCED_LUBES_NOT_DUREX_GEL_ps_between.copy()
interim_res_lubes_not_durex_gel_specific = INTERIM_RES_LUBES_NOT_DUREX_GEL_ps_between.copy()
PS_f_lubes_not_durex_gel, l_select, node_rows = get_inputs_force_PS(PS_f_lubes_not_durex_gel_specific, good_data_for_PS_read_14July,attrlist)

# COMMAND ----------

PS_f_lubes_not_durex_gel.drop_duplicates()

# COMMAND ----------

l_select

# COMMAND ----------

node_rows

# COMMAND ----------

# MAGIC %md
# MAGIC ###Forcing in Lubes & Gel -> Not_Durex -> Not Gel

# COMMAND ----------

PS_f_lubes_not_durex_not_gel_specific = PS_FORCED_LUBES_NOT_DUREX_GEL_PL_AO_ps_between.copy()
interim_res_lubes_not_durex_not_gel_specific = INTERIM_RES_LUBES_NOT_DUREX_GEL_PL_AO_ps_between.copy()
PS_f_lubes_not_durex_not_gel, l_select, node_rows = get_inputs_force_PS(PS_f_lubes_not_durex_not_gel_specific, good_data_for_PS_read_14July,attrlist)

# COMMAND ----------

PS_f_lubes_not_durex_not_gel.drop_duplicates()

# COMMAND ----------

l_select

# COMMAND ----------

node_rows

# COMMAND ----------

# MAGIC %md
# MAGIC ###FORCING THE BREAK

# COMMAND ----------

# # create strucutre after forcing level 4 using attribures in attrlist_btw
# Top Lvl Force for Condom and Lubes
# force_res =create_structure_between(attr_df = good_data_for_PS,
#                                     purchase_struct_df = PS_f_Top_Lvl,
#                                     interim_result = interim_res_f_Top_Lvl,
#                                     attrlist = np.array(['product_sub_category', 'lubes_natural_vs_ao', 'durex_flag', 'top_brand_flag', 'skyn_flag', 'pl_flag', 'mates_flag', 'balance_activ_flag', 'ky_flag', 'gel_flag', 'jelly_flag', 'cream_flag', 'ampoul_flag', 'feeling_intimacy_flag', 'basic_confidence_flag', 'fun_adventure_flag', 'whtlines_lubes_flag', 'basconfd_lubes_flag', 'funadvnt_lubes_flag', 'condom_size', 'latex_vs_non_latex', 'skyn_durex_ao', 'condom_size_large_xl_combined', 'subbrand_nm', 'flavors', 'pack_sizes', 'need_state', 'form', 'pack_size_lseq10_condo']),
#                                     filt_idx = node_rows,
#                                     lvl = l_select,
#                                     nlevels = 3, #params_ps_vtm['default_vars']['nlevels'],
#                                     params = params_ps_vtm) 

# # Top Lvl Force for Condom and Lubes
# force_res_condom_specific =create_structure_between(attr_df = good_data_for_PS,
#                                     purchase_struct_df = PS_f_condom,
#                                     interim_result = interim_res_condom_specific,
#                                     attrlist = np.array(['product_sub_category', 'lubes_natural_vs_ao', 'durex_flag', 'top_brand_flag', 'skyn_flag', 'pl_flag', 'mates_flag', 'balance_activ_flag', 'ky_flag', 'gel_flag', 'jelly_flag', 'cream_flag', 'ampoul_flag', 'feeling_intimacy_flag', 'basic_confidence_flag', 'fun_adventure_flag', 'whtlines_lubes_flag', 'basconfd_lubes_flag', 'funadvnt_lubes_flag', 'condom_size', 'latex_vs_non_latex', 'skyn_durex_ao', 'condom_size_large_xl_combined', 'subbrand_nm', 'flavors', 'pack_sizes', 'need_state', 'form', 'pack_size_lseq10_condo']),
#                                     filt_idx = node_rows,
#                                     lvl = l_select,
#                                     nlevels = 4, #params_ps_vtm['default_vars']['nlevels'],
#                                     params = params_ps_vtm) 

# # Top Lvl Force for Condom and Lubes -> Durex
# force_res_condom_durex_specific =create_structure_between(attr_df = good_data_for_PS,
#                                     purchase_struct_df = PS_f_condom_durex,
#                                     interim_result = interim_res_condom_durex_specific,
#                                     attrlist = np.array(['product_sub_category', 'lubes_natural_vs_ao', 'durex_flag', 'top_brand_flag', 'skyn_flag', 'pl_flag', 'mates_flag', 'balance_activ_flag', 'ky_flag', 'gel_flag', 'jelly_flag', 'cream_flag', 'ampoul_flag', 'feeling_intimacy_flag', 'basic_confidence_flag', 'fun_adventure_flag', 'whtlines_lubes_flag', 'basconfd_lubes_flag', 'funadvnt_lubes_flag', 'condom_size', 'latex_vs_non_latex', 'skyn_durex_ao', 'condom_size_large_xl_combined', 'subbrand_nm', 'flavors', 'pack_sizes', 'need_state', 'form', 'pack_size_lseq10_condo']),
#                                     filt_idx = node_rows,
#                                     lvl = l_select,
#                                     nlevels = 5, #params_ps_vtm['default_vars']['nlevels'],
#                                     params = params_ps_vtm)

# # Top Lvl Force for Condom and Lubes -> Durex -> Large
# force_res_condom_durex_large_specific =create_structure_between(attr_df = good_data_for_PS_read_13July,
#                                     purchase_struct_df = PS_f_condom_durex_large,
#                                     interim_result = interim_res_condom_durex_large_specific,
#                                     attrlist = np.array(['product_sub_category', 'lubes_natural_vs_ao', 'durex_flag', 'top_brand_flag', 'skyn_flag', 'pl_flag', 'mates_flag', 'balance_activ_flag', 'ky_flag', 'gel_flag', 'jelly_flag', 'cream_flag', 'ampoul_flag', 'feeling_intimacy_flag', 'basic_confidence_flag', 'fun_adventure_flag', 'whtlines_lubes_flag', 'basconfd_lubes_flag', 'funadvnt_lubes_flag', 'condom_size', 'latex_vs_non_latex', 'skyn_durex_ao', 'condom_size_large_xl_combined', 'subbrand_nm', 'flavors', 'pack_sizes', 'need_state', 'form', 'pack_size_lseq10_condo']),
#                                     filt_idx = node_rows,
#                                     lvl = l_select,
#                                     nlevels = 6, #params_ps_vtm['default_vars']['nlevels'],
#                                     params = params_ps_vtm)

# # Top Lvl Force for Condom and Lubes -> Durex -> LArge -> feel inti
# force_res_condom_durex_large_feel_inti_specific =create_structure_between(attr_df = good_data_for_PS_read_13July,
#                                     purchase_struct_df = PS_f_condom_durex_large_feel_inti,
#                                     interim_result = interim_res_condom_durex_large_feel_inti_specific,
#                                     attrlist = np.array(['product_sub_category', 'lubes_natural_vs_ao', 'durex_flag', 'top_brand_flag', 'skyn_flag', 'pl_flag', 'mates_flag', 'balance_activ_flag', 'ky_flag', 'gel_flag', 'jelly_flag', 'cream_flag', 'ampoul_flag', 'feeling_intimacy_flag', 'basic_confidence_flag', 'fun_adventure_flag', 'whtlines_lubes_flag', 'basconfd_lubes_flag', 'funadvnt_lubes_flag', 'condom_size', 'latex_vs_non_latex', 'skyn_durex_ao', 'condom_size_large_xl_combined', 'subbrand_nm', 'flavors', 'pack_sizes', 'need_state', 'form', 'pack_size_lseq10_condo']),
#                                     filt_idx = node_rows,
#                                     lvl = l_select,
#                                     nlevels = 7, #params_ps_vtm['default_vars']['nlevels'],
#                                     params = params_ps_vtm)

# # Top Lvl Force for Condom and Lubes -> Durex -> Regu ->
# force_res_condom_durex_regu_specific =create_structure_between(attr_df = good_data_for_PS_read_13July,
#                                     purchase_struct_df = PS_f_condom_durex_regu,
#                                     interim_result = interim_res_condom_durex_regu_specific,
#                                     attrlist = np.array(['product_sub_category', 'lubes_natural_vs_ao', 'durex_flag', 'top_brand_flag', 'skyn_flag', 'pl_flag', 'mates_flag', 'balance_activ_flag', 'ky_flag', 'gel_flag', 'jelly_flag', 'cream_flag', 'ampoul_flag', 'feeling_intimacy_flag', 'basic_confidence_flag', 'fun_adventure_flag', 'whtlines_lubes_flag', 'basconfd_lubes_flag', 'funadvnt_lubes_flag', 'condom_size', 'latex_vs_non_latex', 'skyn_durex_ao', 'condom_size_large_xl_combined', 'subbrand_nm', 'flavors', 'pack_sizes', 'need_state', 'form', 'pack_size_lseq10_condo']),
#                                     filt_idx = node_rows,
#                                     lvl = l_select,
#                                     nlevels = 8, #params_ps_vtm['default_vars']['nlevels'],
#                                     params = params_ps_vtm)

# # Top Lvl Force for Lubes & Gel
# force_res_lubes_specific =create_structure_between(attr_df = good_data_for_PS_read_14July,
#                                     purchase_struct_df = PS_f_lubes,
#                                     interim_result = interim_res_lubes_specific,
#                                     attrlist = np.array(['product_sub_category', 'lubes_natural_vs_ao', 'durex_flag', 'top_brand_flag', 'skyn_flag', 'pl_flag', 'mates_flag', 'balance_activ_flag', 'ky_flag', 'gel_flag', 'jelly_flag', 'cream_flag', 'ampoul_flag', 'feeling_intimacy_flag', 'basic_confidence_flag', 'fun_adventure_flag', 'whtlines_lubes_flag', 'basconfd_lubes_flag', 'funadvnt_lubes_flag', 'condom_size', 'latex_vs_non_latex', 'skyn_durex_ao', 'condom_size_large_xl_combined', 'subbrand_nm', 'flavors', 'pack_sizes', 'need_state', 'form', 'pack_size_lseq10_condo']),
#                                     filt_idx = node_rows,
#                                     lvl = l_select,
#                                     nlevels = 5, #params_ps_vtm['default_vars']['nlevels'],
#                                     params = params_ps_vtm)

# # Top Lvl Force for Lubes & Gel -> Durex ->Not Basic Conf
# force_res_lubes_durex_not_basic_specific =create_structure_between(attr_df = good_data_for_PS_read_14July,
#                                     purchase_struct_df = PS_f_lubes_durex_not_basic,
#                                     interim_result = interim_res_lubes_durex_not_basic_specific,
#                                     attrlist = np.array(['product_sub_category', 'lubes_natural_vs_ao', 'durex_flag', 'top_brand_flag', 'skyn_flag', 'pl_flag', 'mates_flag', 'balance_activ_flag', 'ky_flag', 'gel_flag', 'jelly_flag', 'cream_flag', 'ampoul_flag', 'feeling_intimacy_flag', 'basic_confidence_flag', 'fun_adventure_flag', 'whtlines_lubes_flag', 'basconfd_lubes_flag', 'funadvnt_lubes_flag', 'condom_size', 'latex_vs_non_latex', 'skyn_durex_ao', 'condom_size_large_xl_combined', 'subbrand_nm', 'flavors', 'pack_sizes', 'need_state', 'form', 'pack_size_lseq10_condo']),
#                                     filt_idx = node_rows,
#                                     lvl = l_select,
#                                     nlevels = 6, #params_ps_vtm['default_vars']['nlevels'],
#                                     params = params_ps_vtm)

# # Top Lvl Force for Lubes & Gel -> Durex ->Not Basic Conf
# force_res_lubes_durex_not_basic_specific =create_structure_between(attr_df = good_data_for_PS_read_14July,
#                                     purchase_struct_df = PS_f_lubes_durex_not_basic,
#                                     interim_result = interim_res_lubes_durex_not_basic_specific,
#                                     attrlist = np.array(['product_sub_category', 'lubes_natural_vs_ao', 'durex_flag', 'top_brand_flag', 'skyn_flag', 'pl_flag', 'mates_flag', 'balance_activ_flag', 'ky_flag', 'gel_flag', 'jelly_flag', 'cream_flag', 'ampoul_flag', 'feeling_intimacy_flag', 'basic_confidence_flag', 'fun_adventure_flag', 'whtlines_lubes_flag', 'basconfd_lubes_flag', 'funadvnt_lubes_flag', 'condom_size', 'latex_vs_non_latex', 'skyn_durex_ao', 'condom_size_large_xl_combined', 'subbrand_nm', 'flavors', 'pack_sizes', 'need_state', 'form', 'pack_size_lseq10_condo']),
#                                     filt_idx = node_rows,
#                                     lvl = l_select,
#                                     nlevels = 6, #params_ps_vtm['default_vars']['nlevels'],
#                                     params = params_ps_vtm)

# # Top Lvl Force for Lubes & Gel -> Durex ->Not Basic Conf
# force_res_lubes_not_durex_gel_specific =create_structure_between(attr_df = good_data_for_PS_read_14July,
#                                     purchase_struct_df = PS_f_lubes_not_durex_gel,
#                                     interim_result = interim_res_lubes_not_durex_gel_specific,
#                                     attrlist = np.array(['product_sub_category', 'lubes_natural_vs_ao', 'durex_flag', 'top_brand_flag', 'skyn_flag', 'pl_flag', 'mates_flag', 'balance_activ_flag', 'ky_flag', 'gel_flag', 'jelly_flag', 'cream_flag', 'ampoul_flag', 'feeling_intimacy_flag', 'basic_confidence_flag', 'fun_adventure_flag', 'whtlines_lubes_flag', 'basconfd_lubes_flag', 'funadvnt_lubes_flag', 'condom_size', 'latex_vs_non_latex', 'skyn_durex_ao', 'condom_size_large_xl_combined', 'subbrand_nm', 'flavors', 'pack_sizes', 'need_state', 'form', 'pack_size_lseq10_condo']),
#                                     filt_idx = node_rows,
#                                     lvl = l_select,
#                                     nlevels = 7, #params_ps_vtm['default_vars']['nlevels'],
#                                     params = params_ps_vtm)

# Top Lvl Force for Lubes & Gel -> not Durex -> not gel ->
force_res_lubes_not_durex_not_gel_specific =create_structure_between(attr_df = good_data_for_PS_read_14July,
                                    purchase_struct_df = PS_f_lubes_not_durex_not_gel,
                                    interim_result = interim_res_lubes_not_durex_not_gel_specific,
                                    attrlist = np.array(['product_sub_category', 'lubes_natural_vs_ao', 'durex_flag', 'top_brand_flag', 'skyn_flag', 'pl_flag', 'mates_flag', 'balance_activ_flag', 'ky_flag', 'gel_flag', 'jelly_flag', 'cream_flag', 'ampoul_flag', 'feeling_intimacy_flag', 'basic_confidence_flag', 'fun_adventure_flag', 'whtlines_lubes_flag', 'basconfd_lubes_flag', 'funadvnt_lubes_flag', 'condom_size', 'latex_vs_non_latex', 'skyn_durex_ao', 'condom_size_large_xl_combined', 'subbrand_nm', 'flavors', 'pack_sizes', 'need_state', 'form', 'pack_size_lseq10_condo']),
                                    filt_idx = node_rows,
                                    lvl = l_select,
                                    nlevels = 8, #params_ps_vtm['default_vars']['nlevels'],
                                    params = params_ps_vtm)

# COMMAND ----------

# PS_FORCED_Condom_Lubes=force_res[0]
# PS_FORCED_Condom_Lubes.drop_duplicates()
# #display(PS_FORCED_Condom_Lubes)
# PS_FORCED_CONDOMS_ps_between=force_res_condom_specific[0]
# PS_FORCED_CONDOMS_ps_between.drop_duplicates()
# PS_FORCED_CONDOMS_DUREX_ps_between=force_res_condom_durex_specific[0]
# PS_FORCED_CONDOMS_DUREX_ps_between.drop_duplicates()
# PS_FORCED_CONDOMS_DUREX_LARGE_ps_between=force_res_condom_durex_large_specific[0]
# PS_FORCED_CONDOMS_DUREX_LARGE_ps_between.drop_duplicates()
# PS_FORCED_CONDOMS_DUREX_LARGE_FEEL_INTIMATE_ps_between=force_res_condom_durex_large_feel_inti_specific[0]
# PS_FORCED_CONDOMS_DUREX_LARGE_FEEL_INTIMATE_ps_between.drop_duplicates()
# PS_FORCED_CONDOMS_DUREX_REGULAR_ps_between=force_res_condom_durex_regu_specific[0]
# PS_FORCED_CONDOMS_DUREX_REGULAR_ps_between.drop_duplicates()
# PS_FORCED_LUBES_ps_between=force_res_lubes_specific[0]
# PS_FORCED_LUBES_ps_between.drop_duplicates()
# PS_FORCED_LUBES_DUREX_NOT_BASIC_ps_between=force_res_lubes_durex_not_basic_specific[0]
# PS_FORCED_LUBES_DUREX_NOT_BASIC_ps_between.drop_duplicates()
# PS_FORCED_LUBES_NOT_DUREX_GEL_ps_between=force_res_lubes_durex_not_basic_specific[0]
# PS_FORCED_LUBES_NOT_DUREX_GEL_ps_between.drop_duplicates()
# PS_FORCED_LUBES_NOT_DUREX_GEL_PL_AO_ps_between=force_res_lubes_not_durex_gel_specific[0]
# PS_FORCED_LUBES_NOT_DUREX_GEL_PL_AO_ps_between.drop_duplicates()
PS_FORCED_LUBES_NOT_DUREX_NOT_GEL_ps_between=force_res_lubes_not_durex_not_gel_specific[0]
PS_FORCED_LUBES_NOT_DUREX_NOT_GEL_ps_between.drop_duplicates()

# COMMAND ----------

# PS_FORCED_Condo_Lube.drop_duplicates()

# COMMAND ----------

# INTERIM_RES_FORCED_Condom_Lubes=force_res[1]
# INTERIM_RES_FORCED_Condom_Lubes
# # display(INTERIM_RES_FORCED_Condom_Lubes)
# INTERIM_RES_CONDOMS_ps_between=force_res_condom_specific[1]
# INTERIM_RES_CONDOMS_ps_between
# INTERIM_RES_CONDOMS_DUREX_ps_between=force_res_condom_durex_specific[1]
# INTERIM_RES_CONDOMS_DUREX_ps_between
# INTERIM_RES_CONDOMS_DUREX_LARGE_ps_between=force_res_condom_durex_large_specific[1]
# INTERIM_RES_CONDOMS_DUREX_LARGE_ps_between
# INTERIM_RES_CONDOMS_DUREX_LARGE_FEEL_INTIMATE_ps_between=force_res_condom_durex_large_feel_inti_specific[1]
# INTERIM_RES_CONDOMS_DUREX_LARGE_FEEL_INTIMATE_ps_between
# INTERIM_RES_CONDOMS_DUREX_REGULAR_ps_between=force_res_condom_durex_regu_specific[1]
# INTERIM_RES_CONDOMS_DUREX_REGULAR_ps_between
# INTERIM_RES_LUBES_ps_between=force_res_lubes_specific[1]
# INTERIM_RES_LUBES_ps_between
# INTERIM_RES_LUBES_DUREX_NOT_BASIC_ps_between=force_res_lubes_durex_not_basic_specific[1]
# INTERIM_RES_LUBES_DUREX_NOT_BASIC_ps_between
# INTERIM_RES_LUBES_NOT_DUREX_GEL_ps_between=force_res_lubes_durex_not_basic_specific[1]
# INTERIM_RES_LUBES_NOT_DUREX_GEL_ps_between
# INTERIM_RES_LUBES_NOT_DUREX_GEL_PL_AO_ps_between=force_res_lubes_not_durex_gel_specific[1]
# INTERIM_RES_LUBES_NOT_DUREX_GEL_PL_AO_ps_between
INTERIM_RES_LUBES_NOT_DUREX_NOT_GEL_ps_between=force_res_lubes_not_durex_not_gel_specific[1]
INTERIM_RES_LUBES_NOT_DUREX_NOT_GEL_ps_between

# COMMAND ----------

# #Saving Forced Tree by Levels
# PS_FORCED_Condom_Lubes.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/PS_FORCED_Condom_Lubes.csv")
# INTERIM_RES_FORCED_Condom_Lubes.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/INTERIM_RES_FORCED_Condom_Lubes.csv")
# #Saving Forced Tree for Top -> Condoms
# PS_FORCED_CONDOMS_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/PS_FORCED_CONDOMS_ps_between.csv")
# INTERIM_RES_CONDOMS_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/INTERIM_RES_CONDOMS_ps_between.csv")
# #Saving Forced Tree for Top -> Condoms -> Durex
# PS_FORCED_CONDOMS_DUREX_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/PS_FORCED_CONDOMS_DUREX_ps_between.csv")
# INTERIM_RES_CONDOMS_DUREX_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/INTERIM_RES_CONDOMS_DUREX_ps_between.csv")
# #Saving Forced Tree for Top -> Condoms -> Durex ->Large
# PS_FORCED_CONDOMS_DUREX_LARGE_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/PS_FORCED_CONDOMS_DUREX_LARGE_ps_between_v2.csv")
# INTERIM_RES_CONDOMS_DUREX_LARGE_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/INTERIM_RES_CONDOMS_DUREX_LARGE_ps_between_v2.csv")
# #Saving Forced Tree for Top -> Condoms -> Durex ->Large -> feeling intimate
# PS_FORCED_CONDOMS_DUREX_LARGE_FEEL_INTIMATE_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/PS_FORCED_CONDOMS_DUREX_LARGE_FEEL_INTIMATE_ps_between.csv")
# INTERIM_RES_CONDOMS_DUREX_LARGE_FEEL_INTIMATE_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/INTERIM_RES_CONDOMS_DUREX_LARGE_FEEL_INTIMATE_ps_between.csv")
# #Saving Forced Tree for Top -> Condoms -> Durex ->Regular ->
# PS_FORCED_CONDOMS_DUREX_REGULAR_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/PS_FORCED_CONDOMS_DUREX_REGULAR_ps_between_v2.csv")
# INTERIM_RES_CONDOMS_DUREX_REGULAR_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/INTERIM_RES_CONDOMS_DUREX_REGULAR_ps_between_v2.csv")
# #Saving Forced Tree for Top -> Lubes ->
# PS_FORCED_LUBES_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/PS_FORCED_LUBES_ps_between.csv")
# INTERIM_RES_LUBES_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/INTERIM_RES_LUBES_ps_between.csv")
# #Saving Forced Tree for Top -> Lubes ->
# PS_FORCED_LUBES_DUREX_NOT_BASIC_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/PS_FORCED_LUBES_DUREX_NOT_BASIC_ps_between.csv")
# INTERIM_RES_LUBES_DUREX_NOT_BASIC_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/INTERIM_RES_LUBES_DUREX_NOT_BASIC_ps_between.csv")
# #Saving Forced Tree for Top -> Lubes -> Not Durex -> Gel
# PS_FORCED_LUBES_NOT_DUREX_GEL_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/PS_FORCED_LUBES_NOT_DUREX_GEL_ps_between.csv")
# INTERIM_RES_LUBES_NOT_DUREX_GEL_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/INTERIM_RES_LUBES_NOT_DUREX_GEL_ps_between.csv")
# #Saving Forced Tree for Top -> Lubes -> Not Durex -> Gel ->
# PS_FORCED_LUBES_NOT_DUREX_GEL_PL_AO_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/PS_FORCED_LUBES_NOT_DUREX_GEL_PL_AO_ps_between.csv")
# INTERIM_RES_LUBES_NOT_DUREX_GEL_PL_AO_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/INTERIM_RES_LUBES_NOT_DUREX_GEL_PL_AO_ps_between.csv")
#Saving Forced Tree for Top -> Lubes -> Not Durex -> NOT Gel ->
PS_FORCED_LUBES_NOT_DUREX_NOT_GEL_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/PS_FORCED_LUBES_NOT_DUREX_NOT_GEL_ps_between.csv")
INTERIM_RES_LUBES_NOT_DUREX_NOT_GEL_ps_between.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/INTERIM_RES_LUBES_NOT_DUREX_NOT_GEL_ps_between.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ####13Th July, Morning 11:30. Reading the Data from condom-->durex Part

# COMMAND ----------


good_data_for_PS_read_13July=pd.read_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/good_data_for_PS.csv")
good_data_for_PS_read_13July.drop("Unnamed: 0", axis=1, inplace = True)

PS_FORCED_CONDOMS_DUREX_ps_between_read_13July=pd.read_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/PS_FORCED_CONDOMS_DUREX_ps_between.csv")
PS_FORCED_CONDOMS_DUREX_ps_between_read_13July.drop("Unnamed: 0", axis=1, inplace = True)

INTERIM_RES_CONDOMS_DUREX_ps_between_13July=pd.read_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/INTERIM_RES_CONDOMS_DUREX_ps_between.csv")
INTERIM_RES_CONDOMS_DUREX_ps_between_13July.drop("Unnamed: 0", axis=1, inplace = True)

# COMMAND ----------

display(good_data_for_PS_read_13July)

# COMMAND ----------


display(PS_FORCED_CONDOMS_DUREX_ps_between_read_13July)

# COMMAND ----------

PS_FORCED_CONDOMS_DUREX_ps_between_read_13July.drop_duplicates()

# COMMAND ----------

display(INTERIM_RES_CONDOMS_DUREX_ps_between_13July)

# COMMAND ----------

# MAGIC %md
# MAGIC ####14Th July, Morning 11:30. Reading the Data from COMPLETED CONDOMS BREAK

# COMMAND ----------

good_data_for_PS_read_14July=pd.read_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/good_data_for_PS.csv")
good_data_for_PS_read_14July.drop("Unnamed: 0", axis=1, inplace = True)

PS_FORCED_CONDOMS_DUREX_ps_between_read_14July=pd.read_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/PS_FORCED_CONDOMS_DUREX_REGULAR_ps_between_v2.csv")
PS_FORCED_CONDOMS_DUREX_ps_between_read_14July.drop("Unnamed: 0", axis=1, inplace = True)

INTERIM_RES_CONDOMS_DUREX_ps_between_14July=pd.read_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/INTERIM_RES_CONDOMS_DUREX_REGULAR_ps_between_v2.csv")
INTERIM_RES_CONDOMS_DUREX_ps_between_14July.drop("Unnamed: 0", axis=1, inplace = True)

# COMMAND ----------

good_data_for_PS_read_14July=pd.read_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/good_data_for_PS.csv")
good_data_for_PS_read_14July.drop("Unnamed: 0", axis=1, inplace = True)

PS_FORCED_LUBES_NOT_DUREX_GEL_ps_between=pd.read_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/PS_FORCED_LUBES_DUREX_NOT_BASIC_ps_between.csv")
PS_FORCED_LUBES_NOT_DUREX_GEL_ps_between.drop("Unnamed: 0", axis=1, inplace = True)

INTERIM_RES_LUBES_NOT_DUREX_GEL_ps_between=pd.read_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/INTERIM_RES_LUBES_DUREX_NOT_BASIC_ps_between.csv")
INTERIM_RES_LUBES_NOT_DUREX_GEL_ps_between.drop("Unnamed: 0", axis=1, inplace = True)


# COMMAND ----------

PS_FORCED_LUBES_NOT_DUREX_GEL_ps_between.drop_duplicates()

# COMMAND ----------

display(INTERIM_RES_CONDOMS_DUREX_ps_between_14July)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Other Parts

# COMMAND ----------

# # PS_FORCED_Condo_Lube.drop_duplicates()
# PS_FORCED_CONDOM_DUREX_SKYN_AO.drop_duplicates()

# COMMAND ----------

good_data_for_PS.columns

# COMMAND ----------

good_data_for_PS["NOT_SKYN_DUREX"].unique()

# COMMAND ----------

# adding few colums in PS_FORCED_Condo_Lube
PS_FORCED_Condo_Lube_v1=PS_FORCED_Condo_Lube.copy()
PS_FORCED_Condo_Lube_v1.insert(3, "X4", "-")
PS_FORCED_Condo_Lube_v1.insert(4, "X5", "-")
PS_FORCED_Condo_Lube_v1.insert(5, "X6", "-")
PS_FORCED_Condo_Lube_v1.insert(6, "X7", "-")
PS_FORCED_Condo_Lube_v1.insert(7, "X8", "-")
PS_FORCED_Condo_Lube_v1.insert(8, "X9", "-")
PS_FORCED_Condo_Lube_v1.insert(9, "X10", "-")
PS_FORCED_Condo_Lube_v1.insert(13, "W4", "-")
PS_FORCED_Condo_Lube_v1.insert(14, "W5", "-")
PS_FORCED_Condo_Lube_v1.insert(15, "W6", "-")
PS_FORCED_Condo_Lube_v1.insert(16, "W7", "-")
PS_FORCED_Condo_Lube_v1.insert(17, "W8", "-")
PS_FORCED_Condo_Lube_v1.insert(18, "W9", "-")
PS_FORCED_Condo_Lube_v1.insert(19, "W10", "-")
PS_FORCED_Condo_Lube_v1

# COMMAND ----------

PS_FORCED_Condo_Lube_v2=PS_FORCED_Condo_Lube.copy()
PS_FORCED_Condo_Lube_v2.insert(3, "X4", "-")
PS_FORCED_Condo_Lube_v2.insert(7, "W4", "-")
PS_FORCED_Condo_Lube_v2

# COMMAND ----------

PS_FORCED_Condo_Lube_v1.drop_duplicates()

# COMMAND ----------

PS_FORCED_Condo_Lube_v1.to_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/output/PS_FORCED_Condo_Lube_v1_Lvl1.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Purchase structure before forcing 

# COMMAND ----------

PS.drop_duplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC ######Purchase structure after forcing 

# COMMAND ----------

# PS_f['X5'] = "-"
# PS_f['W5'] = "-"
# PS_f = PS_f[[I for I in PS_f.columns if I.startswith("X")] + [I for I in PS_f.columns if I.startswith("W")]]
PS_manu_f2.drop_duplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC ######Purchase structure after creating new splits below the forced level 

# COMMAND ----------

parent_child_data = create_tree_df(PS_manu_2 , params_ps_vtm)
plot_tree_function(parent_child_data)

# COMMAND ----------

write_obj(PS_manu_f2, 
          catalog_ps['ps_data']['filepath'],
          catalog_ps['ps_data']['filename'], 
          catalog_ps['ps_data']['format'] )
write_obj(interim_res_f2, 
          catalog_ps['intermediate_data']['filepath'],
          catalog_ps['intermediate_data']['filename'], 
          catalog_ps['intermediate_data']['format'] )

# COMMAND ----------

# run if using saved data without running pipeline
# PS = read_obj( 
#           catalog_ps['ps_data']['filepath'],
#           catalog_ps['ps_data']['filename'], 
#           catalog_ps['ps_data']['format'] )
# interim_res = read_obj( 
#           catalog_ps['intermediate_data']['filepath'],
#           catalog_ps['intermediate_data']['filename'], 
#           catalog_ps['intermediate_data']['format'] )

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Create Purchase Structure For Unmapped Data

# COMMAND ----------

unmapped_data_for_PS =data_for_PS[data_for_PS['general_flag'].str.contains("DROP")].reset_index(drop=True)
PS_unmapped = create_structure_unmapped_data(unmapped_data_for_ps=unmapped_data_for_PS, purchase_struct=PS_manu_f2,nlevels=params_ps_vtm["default_vars"]["nlevels"])

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Create Full Purchase Structure From Mapped and Unmapped data

# COMMAND ----------

full_data_for_PS = pd.concat([good_data_for_PS, unmapped_data_for_PS],ignore_index=True)
full_PS =pd.concat([PS_manu_f2, PS_unmapped], ignore_index=True)

# COMMAND ----------

full_PS['W3'] = full_PS['W3'].fillna('-')
full_PS.drop_duplicates()
full_PS['W4'] = full_PS['W4'].fillna('-')
full_PS.drop_duplicates()
full_PS['W5'] = full_PS['W5'].fillna('-')
full_PS.drop_duplicates()

# COMMAND ----------

write_obj(full_PS, 
          catalog_ps['ps_full_data']['filepath'],
          catalog_ps['ps_full_data']['filename'], 
          catalog_ps['ps_full_data']['format'] )

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Plot Tree Results

# COMMAND ----------

tree_df =create_tree_df(PS_manu_2,params_ps_vtm)

# COMMAND ----------

plot_tree_function(tree_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Volume Transfer Matrix Steps

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step by Step VTM

# COMMAND ----------

params_ps_vtm["vtm"]

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC #### Create Sku Matrix

# COMMAND ----------

vtm_input_data=pd.read_csv("/dbfs/FileStore/tables/RB_UK_Wave_5/SWB_POS_BASED/ps_input/vtm_input_data/VTM_INPUT_SWB.csv")

# COMMAND ----------

vtm_input_data.head(5)


# COMMAND ----------

# create calibrated Sku Matrix - Can skip this step and move on to run full data pipeline
skuM = create_vtm(vtm_data = vtm_input_data, params =  params_ps_vtm)
skuM

# COMMAND ----------

# MAGIC %md
# MAGIC #### Tone Down R Batch

# COMMAND ----------

# included in run vtm full pipeline
X_sku_MT = tone_down_r_batch(skuM = skuM, vtm_data = vtm_input_data, params = params_ps_vtm )
X_sku_MT

# COMMAND ----------

# MAGIC %md 
# MAGIC ####Selective Interaction - Optional

# COMMAND ----------

# create subcategory matrix to define/limit interactions
# import pandas as pd
retailer_to_0 = pd.DataFrame([['Coles', 1, 0, 0],['Metcash WW', 0,1,0],['Woolworths', 0,0,1]], columns=['retailer_id','Coles', 'Metcash WW', 'Woolworths'])
retailer_to_0


# COMMAND ----------

X_sku_MT_selective = get_selective_interaction(vtm_input_data, retailer_to_0, X_sku_MT, params_ps_vtm)
X_sku_MT_selective

# COMMAND ----------

# write_obj(vtm_input_data, 
#        catalog_ps['sku_MT_after_selective_int']['filepath'],
#     catalog_ps['sku_MT_after_selective_int']['filename'], 
#           catalog_ps['sku_MT_after_selective_int']['format'] )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Calculate Walk Rate Using VTM Input Data

# COMMAND ----------

wr = calc_walkrate(X_sku_MT, vtm_input_data, params = params_ps_vtm )
wr.plot.hist()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Build Volume Transfer Matrix Using Walk Rate 

# COMMAND ----------

# create volume transfer matrix and display top 10 rows
VTM = vtm(X_sku_MT, vtm_input_data, wr, params = params_ps_vtm )
VTM.head(10)

# COMMAND ----------

display(VTM)


# COMMAND ----------

display(skuM)


# COMMAND ----------

display(X_sku_MT)

# COMMAND ----------

display(wr)

# COMMAND ----------

write_obj(VTM, 
          catalog_ps['vtm_data']['filepath'],
          catalog_ps['vtm_data']['filename'], 
          catalog_ps['vtm_data']['format'] )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Calculate Flow Combinations In Batch

# COMMAND ----------

flow_combos_dict = batch_calc_flow_combos(X_sku_MT = X_sku_MT, vtm_data = vtm_input_data, wr = wr,params = params_ps_vtm )
# to view flow combination use string with concatenated levels
flow_combos_dict['X2']

# COMMAND ----------

flow_combos_dict['X2']
display(flow_combos_dict['X2'])

# COMMAND ----------

flow_combos_dict['X2_X3']
display(flow_combos_dict['X2_X3'])

# COMMAND ----------

flow_combos_dict['X2_X3_X4']
display(flow_combos_dict['X2_X3_X4'])

# COMMAND ----------

flow_combos_dict['X2_X3_X4_X5']
display(flow_combos_dict['X2_X3_X4_X5'])

# COMMAND ----------

flow_combos_dict['X2_X3_X4_X5_X6']
display(flow_combos_dict['X2_X3_X4_X5_X6'])

# COMMAND ----------

flow_combos_dict['X2_X3_X4_X5_X6_X7']
display(flow_combos_dict['X2_X3_X4_X5_X6_X7'])

# COMMAND ----------

flow_combos_dict['form']
display(flow_combos_dict['form'])

# COMMAND ----------

flow_combos_dict['need_state']
display(flow_combos_dict['need_state'])

# COMMAND ----------

flow_combos_dict['latex flag']
display(flow_combos_dict['latex flag'])

# COMMAND ----------

flow_combos_dict['brand_nm']
display(flow_combos_dict['brand_nm'])

# COMMAND ----------

flow_combos_dict['retailer_id']
display(flow_combos_dict['retailer_id'])

# COMMAND ----------

# save all flow combinations in one excel file as different sheets
save_to_excel(flow_combos_dict,catalog_ps,"vtm_flow_combinations")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run Full VTM Pipeline

# COMMAND ----------

VTM = run_full_vtm(vtm_data = vtm_input_data, params = params_ps_vtm )
VTM

# COMMAND ----------

write_obj(VTM, 
          catalog_ps['vtm_data']['filepath'],
          catalog_ps['vtm_data']['filename'], 
          catalog_ps['vtm_data']['format'] )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create RGMX Output files

# COMMAND ----------

#create endnode column using correct number of levels in structure - update X column values
vtm_input_data['END_NODE'] = ""
vtm_input_data['END_NODE'] = vtm_input_data.apply(lambda row: row['X1':'X5'].str.cat(sep=" - "), axis=1)

#create UID column
vtm_input_data['UID'] = vtm_input_data.apply(lambda row: row[['ppg_nm', 'retailer_id']].str.cat(sep="_"), axis=1)

# COMMAND ----------

rgmx_parameters = params_ps_vtm["rgm_outputs"]
rgmx_parameters["RGM_SIMULATION_CONFIG"].update(params_ps_vtm["vtm"]["walkrate"])

rgmx_outputs = rgmx_outputs_generator(
    X_sku_MT,
    vtm_input_data,
    rgmx_parameters,
    catalog_ps
)

# COMMAND ----------

psvtm_df = rgmx_outputs['RGM_PS_VTM']
simulation_df = rgmx_outputs['RGM_SIMULATION_CONFIG']
ps_struct_df = rgmx_outputs['RGM_PS_STRUCTURE']
sourcefeeds_df = rgmx_outputs['SOURCE_FEEDS_MANIFEST']

# COMMAND ----------

simulation_df.shape

# COMMAND ----------

write_obj(psvtm_df, catalog_ps['RGM_PS_VTM']['filepath'], catalog_ps['RGM_PS_VTM']['filename'], catalog_ps['RGM_PS_VTM']['format'] )

write_obj(simulation_df, catalog_ps['RGM_SIMULATION_CONFIG']['filepath'], catalog_ps['RGM_SIMULATION_CONFIG']['filename'], catalog_ps['RGM_SIMULATION_CONFIG']['format'] )

write_obj(ps_struct_df, catalog_ps['RGM_PS_STRUCTURE']['filepath'], catalog_ps['RGM_PS_STRUCTURE']['filename'], catalog_ps['RGM_PS_STRUCTURE']['format'] )

write_obj(sourcefeeds_df, catalog_ps['SOURCE_FEEDS_MANIFEST']['filepath'], catalog_ps['SOURCE_FEEDS_MANIFEST']['filename'], catalog_ps['SOURCE_FEEDS_MANIFEST']['format'] )

# COMMAND ----------

ps_struct_df.head(15)

# COMMAND ----------

display(psvtm_df)

# COMMAND ----------

display(simulation_df)

# COMMAND ----------

display(ps_struct_df)

# COMMAND ----------

display(sourcefeeds_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Prepare VTM Data

# COMMAND ----------

# full_PS=read_obj(
#           catalog_ps['ps_full_data']['filepath'],
#           catalog_ps['ps_full_data']['filename'], 
#           catalog_ps['ps_full_data']['format'] )

# COMMAND ----------

# prepares input data by grouping columns based on levels of aggregation
# check vtm_input_data.shape[0] for total unique UIDs in data and make sure the number is correct
vtm_input_data = prepare_vtm_data(x = full_data_for_PS, PS = full_PS, params = params_ps_vtm)
vtm_input_data.head(2)

# COMMAND ----------

# manual checks
dbutils.fs.ls('dbfs:/FileStore/tables/POS_WB_Training/output/ps_full_data')

# COMMAND ----------

full_data_for_PS.to_csv("/dbfs/FileStore/tables/POS_WB_Training/output/ps_full_data/manual_full_data_for_PS_extract.csv")

# COMMAND ----------

# drop duplicate columns if any
vtm_input_data_1=vtm_input_data
vtm_input_data_1 = vtm_input_data_1.iloc[:-8]
vtm_input_data_1.head(2)

# COMMAND ----------

# save input data if needed to FileStore
write_obj(vtm_input_data, 
       catalog_ps['vtm_input_data']['filepath'],
    catalog_ps['vtm_input_data']['filename'], 
          catalog_ps['vtm_input_data']['format'] )

# COMMAND ----------

#check file name in folders
dbutils.fs.ls('dbfs:/FileStore/tables/POS_WB_Training/ps_input/vtm_input_data/')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### END OF PIPELINE

# COMMAND ----------


