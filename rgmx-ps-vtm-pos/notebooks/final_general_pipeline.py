# Databricks notebook source
#!pip install --upgrade pip
!pip install bios
!pip install julia
!pip install missingpy
!pip install pygam
!pip install rpy2==3.4.5
!pip install pyspark==3.2.1
!pip install pandas==1.4.0
!pip install treelib
!pip install xlsxwriter

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
name_of_wb = 'rgmx-ps-vtm-commit' # workbench directory
# sys.path.append(str(PROJECT_DIR / name_of_wb )) # append workbench directory to system directory

# COMMAND ----------

# #Re-Assigning Repo path, in case modules aren't loaded due to cluster environment
# sys.path=["/Workspace/Repos/yashaswa.verma@reckitt.com/Yash_Merge_Econ_Pos_PSVTM"]+sys.path
# print(sys.path)

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
base_ewb = str(PROJECT_DIR / name_of_wb) + "/" 
params_ewb = load_params(base_ewb)

catalog_ewb = load_catalog(
    base_ewb,
    params_ewb["ewb"]["data_management"]["dir_path_spark"],
    params_ewb["ewb"]["data_management"]["dir_path"],
)

# COMMAND ----------

#load parameters from purchase structure directory
logger.info("Reading purchase structure parameters...")
base_ps = (str(PROJECT_DIR / name_of_wb)) + '/'

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
#optional
standardised_df = standardised_df.withColumn("acv_any_promo", lit(1)) # can create acv_any_promo dummy here
#optional
standardised_df = standardised_df.withColumn("acv_tpr", lit(1)) # can create acv_tpr dummy here

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
# raw_loess_all = read_obj(catalog_ps['input_data']['filepath'],
#          catalog_ps['input_data']['filename'], 
#          catalog_ps['input_data']['format'] )

# COMMAND ----------

base_loess = raw_loess_all
base_loess.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Data Prep For Create Structure

# COMMAND ----------

## CREATE CUSTOM ATTTRIBUTE BASED ON PARAM FILE
data_for_PS=custom_group_columns_df(base_loess, params_ps_vtm)

# COMMAND ----------

### ONLY USE WHEN NEEDED ###
# we can decide to drop certain pair summaries as needed based on the drop_thresh in parameters file
#pair_summary_drop = get_pair_summary_drop(input_data_flags_df, params = params_ps_vtm )
#data_for_PS = drop_pairs(input_data_flags_df, pair_summary_drop, params_ps_vtm )

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

attrlist = np.array(params_ps_vtm["default_vars"]["attrlist"])
attrlist

# COMMAND ----------

# OPTIONAL
# add custom attributes
# example
# good_data_for_PS['product_pack_size_or_total_size'] = good_data_for_PS['product_pack_size_or_total_size'].astype(int)

# Get new attribute list
# attrlist = np.array(params_ps_vtm["default_vars"]["attrlist"])
# attrlist = np.append(attrlist, np.array(['product_pack_size_or_total_size']))
# print(attrlist)

# COMMAND ----------

#Generate Automated Purchase Structure
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
op=get_attr_summary(input_df=good_data_for_PS, attrlist=attrlist, input_filt=fx, size_validated=0.075, useshare=True, params=params_ps_vtm)

# COMMAND ----------

#get raw indexes
op[1]

# COMMAND ----------

#get normalized indexes
op[0]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Function to force a change tree structure based on user needs

# COMMAND ----------

# OPTIONAL

# # Creating Additional Colums in PS Copy Data
# PS_ADDED_COLUMS=PS.copy()
# PS_ADDED_COLUMS=PS_FORCED_top_lvl.copy()
# PS_ADDED_COLUMS.insert(2, "X3", "-")
# PS_ADDED_COLUMS.insert(5, "W3", "-")
# PS_ADDED_COLUMS.insert(3, "X4", "-")
# PS_ADDED_COLUMS.insert(7, "W4", "-")
# PS_ADDED_COLUMS.insert(4, "X5", "-")
# PS_ADDED_COLUMS.insert(5, "X6", "-")
# PS_ADDED_COLUMS.insert(6, "X7", "-")
# PS_ADDED_COLUMS.insert(7, "X8", "-")
# PS_ADDED_COLUMS.insert(8, "X9", "-")
# PS_ADDED_COLUMS.insert(9, "X10", "-")
# PS_ADDED_COLUMS.insert(12, "W3", "-")
# PS_ADDED_COLUMS.insert(13, "W4", "-")
# PS_ADDED_COLUMS.insert(14, "W5", "-")
# PS_ADDED_COLUMS.insert(9, "W5", "-")
# PS_ADDED_COLUMS.insert(16, "W7", "-")
# PS_ADDED_COLUMS.insert(17, "W8", "-")
# PS_ADDED_COLUMS.insert(18, "W9", "-")
# PS_ADDED_COLUMS.insert(19, "W10", "-")
# PS_ADDED_COLUMS.drop_duplicates()

# COMMAND ----------

good_data_for_PS[attrlist].head()

# COMMAND ----------

# create a copy of old results
PS_copy = PS.copy()
interim_res_f = interim_res.copy()

PS_f, l_select, node_rows = get_inputs_force_PS(PS_copy, good_data_for_PS, attrlist)

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
PS_f.drop_duplicates()

# COMMAND ----------

# create strucutre after forcing level using attribures in attrlist_btw
force_res =create_structure_between(attr_df = good_data_for_PS,
                                    purchase_struct_df = PS_f,
                                    interim_result = interim_res_f,
                                    attrlist = attrlist,
                                    filt_idx = node_rows,
                                    lvl = l_select,
                                    nlevels = params_ps_vtm['default_vars']['nlevels'],
                                    params = params_ps_vtm) 

# COMMAND ----------

# MAGIC %md
# MAGIC ######Purchase structure after creating new splits below the forced level 

# COMMAND ----------

PS_1 = force_res[0]
PS_1.drop_duplicates()

# COMMAND ----------

interim_result_df = force_res[1]
interim_result_df   #[interim_result_df.L==4]

# COMMAND ----------

parent_child_data = create_tree_df(PS_1 , params_ps_vtm)
plot_tree_function(parent_child_data)

# COMMAND ----------

write_obj(PS, 
          catalog_ps['ps_data']['filepath'],
          catalog_ps['ps_data']['filename'], 
          catalog_ps['ps_data']['format'] )
write_obj(interim_res, 
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
PS_unmapped = create_structure_unmapped_data(unmapped_data_for_ps=unmapped_data_for_PS, purchase_struct=PS,nlevels=params_ps_vtm["default_vars"]["nlevels"])

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Create Full Purchase Structure From Mapped and Unmapped data

# COMMAND ----------

full_data_for_PS = pd.concat([good_data_for_PS, unmapped_data_for_PS],ignore_index=True)
full_PS =pd.concat([PS, PS_unmapped], ignore_index=True)

# COMMAND ----------

full_PS.drop_duplicates()

# COMMAND ----------

full_PS['W3'] = full_PS['W3'].fillna('-')
full_PS.drop_duplicates()

# COMMAND ----------

write_obj(full_PS, 
          catalog_ps['ps_data']['filepath'],
          catalog_ps['ps_data']['filename'], 
          catalog_ps['ps_data']['format'] )

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Plot Full Tree Results

# COMMAND ----------

tree_df =create_tree_df(full_PS,params_ps_vtm)

# COMMAND ----------

plot_tree_function(tree_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Volume Transfer Matrix Steps

# COMMAND ----------

# MAGIC %md
# MAGIC #### Prepare VTM Data

# COMMAND ----------

# prepares input data by grouping columns based on levels of aggregation
# check vtm_input_data.shape[0] for total unique UIDs in data and make sure the number is correct
vtm_input_data = prepare_vtm_data(x = full_data_for_PS, PS = full_PS, params = params_ps_vtm)
vtm_input_data.head(2)

# COMMAND ----------

# drop duplicate columns if any
#vtm_input_data = vtm_input_data.iloc[:, :-1]

# COMMAND ----------

# save input data if needed to FileStore
write_obj(vtm_input_data, 
       catalog_ps['vtm_input_data']['filepath'],
    catalog_ps['vtm_input_data']['filename'], 
          catalog_ps['vtm_input_data']['format'] )

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
# MAGIC #### Step by Step VTM

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create Sku Matrix

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
# subcatmat = pd.DataFrame([['core', 1, 0, 0],['daily', 0,1,0],['otros', 1,1,1]], columns=['product_sub_category','core', 'daily', 'otros'])
# subcatmat

# COMMAND ----------

# X_sku_MT_selective = get_selective_interaction(vtm_input_data, subcatmat, X_sku_MT, params_ps_vtm)
# X_sku_MT_selective

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
flow_combos_dict['X2_X3']

# COMMAND ----------

flow_combos_dict['X2']

# COMMAND ----------

# save all flow combinations in one excel file as different sheets
save_to_excel(flow_combos_dict,catalog_ps,"vtm_flow_combinations")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create RGMX Output files

# COMMAND ----------

#create endnode column using correct number of levels in structure - update X column values
vtm_input_data['END_NODE'] = ""
vtm_input_data['END_NODE'] = vtm_input_data.apply(lambda row: row['X1':'X6'].str.cat(sep=" - "), axis=1)
#create UID column
vtm_input_data['UID'] = vtm_input_data.apply(lambda row: row[['RETAILER_ID', 'PPG_NM']].str.cat(sep=" | "), axis=1)
vtm_input_data.display()

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

# MAGIC %md
# MAGIC ##### END OF PIPELINE
