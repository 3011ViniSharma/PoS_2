# Databricks notebook source
!pip install rpy2==3.5.4

# COMMAND ----------

!pip install pyspark==3.2.1
!pip install pandas==1.2.4

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
name_of_ewb = 'rgmx-econometrics_gitlab' # econometrics directory
name_of_ps = 'rgmx-pos_psvtm' # purchase structure directory rgmx_pos_ps_vtm rgmx_ewb-econometrics_clone
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
from src_psvtm.models.test_pos_ps_vtm import *

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
name_of_ps = 'rgmx-pos_psvtm'
base_ps = (str(PROJECT_DIR / name_of_ps)) + '/' 
params_ps = load_params(base_ps)
params_ps_vtm = params_ps['ps_vtm']
params_ps = load_params(base_ps)
params_ps_vtm = params_ps['ps_vtm']

# COMMAND ----------

catalog_ps = load_catalog(
    base_ewb,
    params_ewb["ewb"]["data_management"]["dir_path_spark"],
    params_ewb["ewb"]["data_management"]["dir_path"],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Date Preparation From Econometrics Module

# COMMAND ----------

## Do not run only needed
# read inputs
input_data = load_spark_data(catalog_ewb["input_data_prep"])
columns = StructType([])
covid_flag_1_df = spark.createDataFrame(data = [],
                           schema = columns)
#load_spark_data(catalog["input_covid_flag_1"])
covid_flag_2_df = covid_flag_1_df = spark.createDataFrame(data = [],
                           schema = columns)
#load_spark_data(catalog["input_covid_flag_2"])
logger.info("Running data preprocessing pipeline...")
standardised_df = standardise_batch(input_data, params_ewb["schema"])


# COMMAND ----------

standardised_df = standardised_df.withColumn("acv_any_promo", lit(1))

# COMMAND ----------

display(standardised_df)

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

# write_obj(raw_loess_all, 
#           catalog_ps['input_data']['filepath'],
#           catalog_ps['input_data']['filename'], 
#           catalog_ps['input_data']['format'] )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Structure Steps

# COMMAND ----------

# Used saved raw loess below
#raw_loess_all = pd.read_csv("/dbfs/mnt/upload/pos_purchase_structure_data/data/input/raw_loess_all.csv")

# COMMAND ----------

#set the ewb parameters and catalog in ps structure for run loess
set_params_catalog_ewb(params_ewb,catalog_ewb) # add additional commentary here

# COMMAND ----------

base_loess = raw_loess_all
base_loess

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Data Prep For Create Structure

# COMMAND ----------

## FLAG ATTTRIBUTE BASED ON PARAM FILE
input_data_flags_df = get_prod_attr_flags_df(base_loess, params_ps_vtm, dataframe= True)
input_data_flags_df =flag_items(input_data_flags_df, params_ps_vtm)
data_for_PS = input_data_flags_df

# COMMAND ----------


##Verify all attributes exist in columns of input_data_flags_Df
verify_attributes_prod_attr_flags(input_data_flags_df, params = params_ps_vtm)

 
##Verify OPI XPI
input_data_flags_df_testing = input_data_flags_df.rename(columns = {"sku": "product_id",  "volume_ltr_000": "net_sales_volume"})
opi_xpi_df = calc_opi_xpi(input_data_flags_df_testing, attr = 'brand_nm', filt_index = pd.Series([True for i in range(len(input_data_flags_df))]), params = params_ps_vtm)
verify_opi_xpi_output(opi_xpi_df, input_data_flags_df_testing)

# COMMAND ----------

### ONLY USE WHEN NEEDED ###
# we can decide to drop certain pair summaries as needed based on the drop_thresh in parameters file
#pair_summary_drop = get_pair_summary_drop(input_data_flags_df, params = params_ps_vtm )
#data_for_PS = drop_pairs(input_data_flags_df, pair_summary_drop, params_ps_vtm )

# COMMAND ----------

# data_for_PS.to_csv("/dbfs/mnt/upload/pos_purchase_structure_data/data/input/data_for_PS.csv", index=False)



# COMMAND ----------

# Get the attribute list
attrlist = np.array(params_ps_vtm["default_vars"]["attrlist"])

# COMMAND ----------

# Subset for general flag that meet the about the criteria set in data prep as input
good_data_for_PS = data_for_PS[data_for_PS ['general_flag'] == 'GOOD'].reset_index(drop=True) 

# COMMAND ----------

fx = good_data_for_PS ['general_flag'] == 'GOOD'
fx = fx.reset_index(drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Purchase Structure

# COMMAND ----------

res = create_structure(attr_df= good_data_for_PS, attrlist=attrlist, filt_idx=fx,nlevels= params_ps_vtm["default_vars"]["nlevels"], params=params_ps_vtm )
## get purchase structure
PS = res[0]
interim_res = res[1]

# COMMAND ----------

PS

# COMMAND ----------

interim_res

# COMMAND ----------

# write_obj(PS, 
#           catalog_ps['ps_data']['filepath'],
#           catalog_ps['ps_data']['filename'], 
#           catalog_ps['ps_data']['format'] )

# COMMAND ----------


# write_obj(interim_res, 
#        catalog_ps['intermediate_data']['filepath'],
#     catalog_ps['intermediate_data']['filename'], 
#           catalog_ps['intermediate_data']['format'] )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Volume Transfer Matrix Steps

# COMMAND ----------

# MAGIC %md
# MAGIC #### Prepare VTM Data

# COMMAND ----------


vtm_input_data = prepare_vtm_data(x = good_data_for_PS, PS = PS, params = params_ps_vtm)
vtm_input_data

# COMMAND ----------

vtm_input_data

# COMMAND ----------

#test_proper_x_w_output(vtm_input_data, parameters = params_ps_vtm)
#test_item_col(vtm_input_data = vtm_input_data, parameters = params_ps_vtm)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create VTM Initial Object

# COMMAND ----------


skuM = create_vtm(vtm_data = vtm_input_data, params =  params_ps_vtm)
skuM

# COMMAND ----------

test_rows_cols_create_vtm(skuM)
test_create_vtm_diagonal_vtm_initial(skuM)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Tone Down R Batch

# COMMAND ----------

X_sku_MT = tone_down_r_batch(skuM = skuM, vtm_data = vtm_input_data, params = params_ps_vtm )
X_sku_MT

# COMMAND ----------

test_tone_down_diagonal(X_sku_MT)
assert_proper_tone_down_shape(X_sku_MT)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Calculate Walk Rate Using VTM Input Data

# COMMAND ----------

I = X_sku_MT
sku_cols = X_sku_MT.columns
wr = calc_walkrate(I, vtm_input_data, params = params_ps_vtm )
wr

# COMMAND ----------

# MAGIC %md
# MAGIC #### Build Volume Transfer Matrix Using Walk Rate 

# COMMAND ----------

VTM = vtm(X_sku_MT, vtm_input_data, wr, params = params_ps_vtm )
VTM

# COMMAND ----------

VTM

# COMMAND ----------

test_rowsums(VTM)
test_vtm_rows_cols(VTM)
test_VTM_zero_diag(VTM)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run Full VTM

# COMMAND ----------

VTM = run_full_vtm(vtm_data = vtm_input_data, params = params_ps_vtm )
VTM

# COMMAND ----------

VTM.to_csv("/dbfs/mnt/upload/pos_purchase_structure_data/data/input/VTM_sku_retailer55.csv", index=True)

# COMMAND ----------

# write_obj(VTM, 
#           catalog_ps['vtm_data']['filepath'],
#           catalog_ps['vtm_data']['filename'], 
#           catalog_ps['vtm_data']['format'] )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Calculate Flow Combinations In Batch

# COMMAND ----------

a = batch_calc_flow_combos(X_sku_MT = X_sku_MT, vtm_data = vtm_input_data, wr = wr,params = params_ps_vtm )
a

# COMMAND ----------

assert_share_wr_decimals(a)
test_calc_flow_Xcolnames(a)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Purchase Structure (Multi Attribute)

# COMMAND ----------

# Test with multiple atributes
attrlist_comb = np.array(["brand_nm", "packaging", "TYPE8","TYPE4","TYPE6","TYPE7","BRAND16","BRAND13","BRAND75"])
fx_comb = good_data_for_PS ['general_flag'] == 'GOOD'
fx_comb = fx_comb.reset_index(drop=True)

# COMMAND ----------

data_for_PS_comb =good_data_for_PS.copy()

# COMMAND ----------

res_comb = create_structure(attr_df= good_data_for_PS, attrlist=attrlist_comb, filt_idx=fx,nlevels=params_ps_vtm["default_vars"]["nlevels"], params=params_ps_vtm )
## get purchase structure
PS_comb = res_comb[0]
interim_res_comb = res_comb[1]

# COMMAND ----------

PS_comb

# COMMAND ----------

interim_res_comb

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Purchase Structure (6 levels and Multi Attribute)

# COMMAND ----------

res_comb_6 = create_structure(attr_df= good_data_for_PS, attrlist=attrlist_comb, filt_idx=fx,nlevels=6, params=params_ps_vtm )
## get purchase structure
PS_comb_6 = res_comb_6[0]
interim_res_comb_6 = res_comb_6[1]

# COMMAND ----------

PS_comb_6

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Create Structure (Force Atrributes)

# COMMAND ----------

PS_f =PS.copy()

# COMMAND ----------

interim_res_f = interim_res.copy()

# COMMAND ----------

PS_f

# COMMAND ----------

interim_res_f

# COMMAND ----------

len(PS)

# COMMAND ----------

PS = pd.read_csv("/dbfs/mnt/upload/pos_purchase_structure_data/data/output/ps_data/ps_data_20230111-161605.csv")

# COMMAND ----------

interim_res = pd.read_csv("/dbfs/mnt/upload/pos_purchase_structure_data/data/output/interim_data/interim_data_20230110-143710.csv")

# COMMAND ----------

attrlist_btw = np.array(["MULTI","BRAND29"])
force_res =create_structure_between(attr_df= good_data_for_PS, price_struct_df =  PS_f, interim_result= interim_res_f,attrlist= attrlist_btw, filt_idx=fx,lvl=2,nlevels = 4, params= params_ps_vtm ) 

# COMMAND ----------

## get purchase structure
PS_forced = force_res[0]
interim_res_forced = force_res[1]

# COMMAND ----------

PS_forced

# COMMAND ----------

interim_res_forced 
