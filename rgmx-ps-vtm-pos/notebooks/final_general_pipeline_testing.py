# Databricks notebook source
!pip install pyspark==3.2.1
!pip install pandas==1.2.4
!pip install treelib
!pip install rpy2==3.5.4

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
from src_psvtm.utils import *

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

# MAGIC %md
# MAGIC ### Date Preparation From Econometrics Module

# COMMAND ----------

# ## Do not run only needed
# # read inputs
# input_data = load_spark_data(catalog_ewb["input_data_prep"])
# columns = StructType([])
# covid_flag_1_df = spark.createDataFrame(data = [],
#                            schema = columns)
# #load_spark_data(catalog["input_covid_flag_1"])
# covid_flag_2_df = covid_flag_1_df = spark.createDataFrame(data = [],
#                            schema = columns)
# #load_spark_data(catalog["input_covid_flag_2"])
# logger.info("Running data preprocessing pipeline...")
# standardised_df = standardise_batch(input_data, params_ewb["schema"])


# COMMAND ----------

# standardised_df = standardised_df.withColumn("acv_any_promo", lit(1)) 

# COMMAND ----------

# raw_loess_all, raw_data_good, raw_data_drop, unique_count_df = data_preparation(
#     standardised_df,
#     params_ewb,
#     params_ewb["holidays_config"],
#     covid_flag_1_df,
#     covid_flag_2_df,
#     catalog_ewb,
# )

# write_obj(raw_loess_all, 
#           catalog_ps['input_data']['filepath'],
#           catalog_ps['input_data']['filename'], 
#           catalog_ps['input_data']['format'] )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Structure Steps

# COMMAND ----------

# Used saved raw loess below
raw_loess_all = read_obj( catalog_ps['input_data']['filepath'],
          catalog_ps['input_data']['filename'], 
          catalog_ps['input_data']['format'])

# COMMAND ----------

#set the ewb parameters and catalog in ps structure for run loess
set_params_catalog_ewb(params_ewb,catalog_ewb) # add additional commentary here

# COMMAND ----------

base_loess = raw_loess_all
#base_loess

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Data Prep For Create Structure

# COMMAND ----------

## FLAG ATTTRIBUTE BASED ON PARAM FILE
#input_data_flags_df = get_prod_attr_flags_df(base_loess, params_ps_vtm, dataframe= True)
#input_data_flags_df =flag_items(input_data_flags_df, params_ps_vtm)
#data_for_PS = input_data_flags_df

# COMMAND ----------

# def custom_group_columns_df (input_data: pd.DataFrame, params: dict,  case = 'lower') -> pd.DataFrame:
#     """
#     This function creates product attribute flags that the user may either define or use default parameters for 
#     Args:
#         input_data: input data given from initial POS data
#         parameters: parameter dict
#             These parameters are used to determine which attributes (values) in which columns to make dummy variables for
#     Returns:
#         input_data: dataframe with selected/computer-chosen variables as dummy variables 
#     """
#     if case.upper() == 'lower':
#         input_data.columns = [x.lower() for x in input_data.columns]
#     vars_selection = params["pos_purchase_structures"]['data']["variable_selection"]
#     sales_col = params["default_vars"]['value_col']
#     cust_vars = params["pos_purchase_structures"]['data']["variable_transformation_def"]
#     for key in cust_vars.keys():
#         for transforms in cust_vars[key].keys():
#             col_name = key
#             name_for_others = cust_vars[col_name][transforms]['name_to_use_for_others']
#             column_name_to_use = cust_vars[col_name][transforms]['column_name_to_use']
#             print(column_name_to_use)
#             if cust_vars[col_name][transforms]['how'] not in ['', 'default']:
#                 if cust_vars[col_name][transforms]['how'] == 'top':
#                     sort_vars = input_data.groupby(col_name).agg({
#                          cust_vars[col_name][transforms]['sort_on']: 'sum'
#                      }).reset_index()
#                     sort_vars.columns = [col_name, 'to_order']
#                     sort_vars = sort_vars.sort_values('to_order', ascending = False).reset_index(drop = True)
#                     num_included = cust_vars[col_name][transforms]['top']
#                     attrs_included = list(sort_vars[col_name][:num_included])        
#                 elif cust_vars[col_name][transforms]['how'] == 'predefined':
#                     attrs_included = cust_vars[col_name][transforms]['what']
#             else:
#                 sort_vars = input_data.groupby(col_name).agg({
#                          sales_col: 'sum'
#                      }).reset_index()
#                 sort_vars.columns = [col_name, 'to_order']
#                 sort_vars = sort_vars.sort_values('to_order', ascending = False).reset_index(drop = True)
#                 attrs_included = list(sort_vars[col_name][:params["default_vars"]['top']])
        
#             input_data[column_name_to_use]=np.where(input_data[col_name].isin(attrs_included),input_data[col_name],name_for_others)
        
#     return input_data


# COMMAND ----------

# data_for_PS=custom_group_columns_df(base_loess, params_ps_vtm)
# data_for_PS.head()

# COMMAND ----------

### ONLY USE WHEN NEEDED ###
# we can decide to drop certain pair summaries as needed based on the drop_thresh in parameters file
#pair_summary_drop = get_pair_summary_drop(input_data_flags_df, params = params_ps_vtm )
#data_for_PS = drop_pairs(input_data_flags_df, pair_summary_drop, params_ps_vtm )

# COMMAND ----------

# #data_for_PS.to_csv("/dbfs/mnt/upload/pos_psvtm_testing/data/ps_input//data_for_PS.csv", index=False)
# write_obj(data_for_PS, 
#           catalog_ps['data_for_ps']['filepath'],
#           catalog_ps['data_for_ps']['filename'], 
#           catalog_ps['data_for_ps']['format'] )

# COMMAND ----------

#Used saved raw loess below
data_for_PS = read_obj( catalog_ps['data_for_ps']['filepath'],
          catalog_ps['data_for_ps']['filename'], 
          catalog_ps['data_for_ps']['format'])

# COMMAND ----------

# Get the attribute list
attrlist = np.array(params_ps_vtm["default_vars"]["attrlist"])
attrlist

# COMMAND ----------

# Subset for general flag that meet the about the criteria set in data prep as input
good_data_for_PS = data_for_PS[data_for_PS ['general_flag'] == 'GOOD'].reset_index(drop=True) 

# COMMAND ----------

fx = good_data_for_PS ['general_flag'] == 'GOOD'
fx = fx.reset_index(drop=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Purchase Structure for Mapped ("GOOD") Data

# COMMAND ----------

# res = create_structure(attr_df= good_data_for_PS, attrlist=attrlist, filt_idx=fx,nlevels= params_ps_vtm["default_vars"]["nlevels"], params=params_ps_vtm )
# ## get purchase structure
# PS = res[0]
# interim_res = res[1]

# COMMAND ----------

# write_obj(interim_res, 
#           catalog_ps['intermediate_data']['filepath'],
#           catalog_ps['intermediate_data']['filename'], 
#           catalog_ps['intermediate_data']['format'] )

# write_obj(PS, 
#           catalog_ps['ps_data']['filepath'],
#           catalog_ps['ps_data']['filename'], 
#           catalog_ps['ps_data']['format'] )

# COMMAND ----------

interim_res = read_obj( 
          catalog_ps['intermediate_data']['filepath'],
          catalog_ps['intermediate_data']['filename'], 
          catalog_ps['intermediate_data']['format'] )

PS = read_obj( 
          catalog_ps['ps_data']['filepath'],
          catalog_ps['ps_data']['filename'], 
          catalog_ps['ps_data']['format'] )

# COMMAND ----------

PS.drop_duplicates()

# COMMAND ----------

tree_df =create_tree_df(PS,params_ps_vtm)
plot_tree_function(tree_df)

# COMMAND ----------

interim_res.head()

# COMMAND ----------

PS_f =PS.copy()
interim_res_f = interim_res.copy()
PS_f.drop_duplicates()

# COMMAND ----------

good_data_for_PS[(good_data_for_PS.packtype=="Cans") & (good_data_for_PS.brand_nm!="BRAND29")]
node_rows=(good_data_for_PS.packtype=="Cans") & (good_data_for_PS.brand_nm!="BRAND29")

#Force the attribute at level 4

PS_f.loc[node_rows.values,"X4"]=good_data_for_PS[node_rows.values]["origin"]
PS_f.drop_duplicates()


# COMMAND ----------

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
    
    if len(filt_idx) <= 1:
        filt_idx = pd.Series([True for i in range(len(attr_df))])

    interim_res = []
    assert len(attr_df) == len(purchase_struct_df) # ensure data frame and price structure are the same
    lvl = lvl - 1 # subtract to move index up
    f = pd.Series([ True for i in range(len(attr_df))])
    if purchase_struct_df is None or lvl <= 0: 
        # Create dummy result df
        fixed_cols = ["X" + str(i + 1) for i in range(nlevels)]
        purchase_struct_df= pd.DataFrame(data="-", columns=fixed_cols, index=attr_df.index)
        purchase_struct_df.iloc[:, 0] = "Total"  # first column is Total
            
    else:
        last_value =int(list(purchase_struct_df.columns[-1])[-1]) # get the last value and subtract our current value from it
        purchase_struct_df.iloc[:,lvl:last_value] = "-" # update everything in current level and below        
        weight_cols = ["W" + str(i+1) for i in range(lvl, nlevels)] # create clean column
        purchase_struct_df.loc[:, weight_cols] = "-"
        
        # Get new weight for current level
        filt_lvl = lvl + 1
        interim_result =interim_result[interim_result['L'] == filt_lvl]
        
        # get winner in new level
        winner_result =interim_result[['TOT_INDEX', 'attributes' ]]
        winner_result = winner_result.sort_values(by=['TOT_INDEX'], ascending = False)
        filt_winner_result = winner_result[winner_result.attributes.isin(attrlist)]
        
        if len(filt_winner_result) > 0:
            logger.info("Getting updated weight coefficeint in force")
            filt_winner_result = filt_winner_result.sort_values(by=['TOT_INDEX'], ascending = False).reset_index()
            filt_winner = filt_winner_result.loc[:, 'attributes'][0]
            logger.info("Here is the new winner %s",filt_winner)
        else:
            logger.info("Selected forced attributes not in interim results, going to select the first in interim")
            filt_winner = winner_result.loc[:, 'attributes'][0]
            logger.info("Here is the new winner %s",filt_winner)
   
        
        interim_result = interim_result.set_index('attributes')
        interim_result['winner'] = filt_winner # forced winner
        new_weight = level_weight(interim_output = interim_result, winner=filt_winner, input_df=attr_df, params= params)
        
        logger.info("This is current index level %d to be updated ", lvl+1)
        logger.info("This is the current weight W%s",str(lvl+1) )
        purchase_struct_df.iloc[:,lvl] = filt_winner # update everything in current level and below  
        purchase_struct_df.loc[:, "W"+str(lvl+1)] = new_weight
        interim_result['attributes'] = interim_result.index
    for curr_lvl in range(lvl,nlevels-1):  # Python is not inclusive
        phrase = purchase_struct_df.iloc[:, 0:curr_lvl + 1].apply(lambda x: _concat(x), axis=1).to_frame()
        # check whether the share of level are above the threshold
        phrase.loc[phrase.index, "lowshare_filt"] = attr_df.loc[phrase.index, vol_col].fillna(0).sum() / \
                                                    attr_df[vol_col].fillna(0).sum()
        phrase.loc[phrase.index, "lowshare_filt_filter"] = phrase.loc[phrase.index, "lowshare_filt"] <= thresh
        phrase = phrase[phrase["lowshare_filt_filter"] != True]
        phrase = phrase.drop(columns=["lowshare_filt", "lowshare_filt_filter"])
        unique_phrases = phrase.drop_duplicates().values
        logger.info("\n In create structure loop\n")
        logger.info("INFO: level number: %d ", curr_lvl + 2)

        dashes_sum = (purchase_struct_df.iloc[:, curr_lvl - 1] == "-").sum()
        if curr_lvl >= 2:
            print(dashes_sum)
            print(len(purchase_struct_df))

        if curr_lvl >= 2 and dashes_sum == len(purchase_struct_df):
            logger.info("Only running once")
            new_result = pd.concat(interim_res, ignore_index = True)
            interim_result = pd.concat([interim_result, new_result], ignore_index= True)
            return [purchase_struct_df, interim_result]

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
                if len(struc) > 1:
                    purchase_struct_df = struc[0]
                    interim = struc[1]
                    if isinstance(interim, pd.DataFrame):
                        interim.loc[:, "_id"] = str(unique_phrase[0])
                        interim_res.append(interim)
                else:
                    purchase_struct_df = struc
            else:
                logger.info("WARNING: No attributes to test")
    if len(interim_res) > 0:
        new_result = pd.concat(interim_res, ignore_index = True)
        interim_result = pd.concat([interim_result, new_result], ignore_index= True)
    return [purchase_struct_df,interim_result]

# COMMAND ----------

import pandas as pd
from typing import Dict,List
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

# COMMAND ----------


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
                        interim_result = interim_result.append(interim)
                else:
                    purchase_struct_df = struc
            else:
                logger.info("WARNING: No attributes to test")
#     if len(interim_result) > 0:
#         interim_result = pd.concat(interim_result, ignore_index = True)
#         interim_result = pd.concat([interim_result, new_result], ignore_index= True)
    return [purchase_struct_df,interim_result]

# COMMAND ----------

attr_df= good_data_for_PS.copy()
purchase_struct_df = PS_f.copy()
interim_result = interim_res_f.copy()
attrlist = attrlist_btw = np.array(['packaging_group', 'type_aggregate', 'packsize_group','brand_16_flag', 'brand_13_flag'])
filt_idx = node_rows
lvl=4
nlevels =5
params = params_ps_vtm
    
force_res =create_structure_between(attr_df= good_data_for_PS, purchase_struct_df =  PS_f, interim_result= interim_res_f,attrlist= attrlist_btw, filt_idx=node_rows,lvl=4,nlevels = 5, params= params_ps_vtm ) 

# COMMAND ----------

purchase_struct = force_res[0]
interim_result_df = force_res[1]


# COMMAND ----------

interim_result_df[interim_result_df["L"]==5]._id.unique()

# COMMAND ----------

tree_df =create_tree_df(purchase_struct,params_ps_vtm)
plot_tree_function(tree_df)

# COMMAND ----------

purchase_struct[purchase_struct['X3']=="Stubbies"].drop_duplicates()#[purchase_struct['X3']]
purchase_struct.drop_duplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Create Purchase Structure For Unmapped Data

# COMMAND ----------

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
            selected_structure = purchase_struct_df[["X"+str(i) for i in range(1,curr_lvl+1)]+["W"+str(i) for i in range(1,curr_lvl+1)]].drop_duplicates()
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

# COMMAND ----------

unmapped_data_for_ps =data_for_PS[data_for_PS['general_flag'].str.contains("DROP")].reset_index(drop=True)
unmapped_data_for_ps.shape

# COMMAND ----------

int_res=interim_result_df.copy()
purchase_struct_df=purchase_struct.copy()

# COMMAND ----------

PS_unmapped = create_structure_unmapped_data (unmapped_data_for_ps,purchase_struct, nlevels)
PS_unmapped.drop_duplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Create Full Purchase Structure From Mapped and Unmapped data

# COMMAND ----------

full_data_for_PS = pd.concat([good_data_for_PS, unmapped_data_for_PS],ignore_index=True)

# COMMAND ----------

full_PS =pd.concat([PS, PS_unmapped], ignore_index=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Plot Full Tree Results

# COMMAND ----------

tree_df =create_tree_df(full_PS,params_ps_vtm)

# COMMAND ----------

plot_tree_function(tree_df)

# COMMAND ----------

write_obj(full_PS, 
          catalog_ps['ps_data']['filepath'],
          catalog_ps['ps_data']['filename'], 
          catalog_ps['ps_data']['format'] )

# COMMAND ----------


write_obj(interim_res, 
       catalog_ps['intermediate_data']['filepath'],
    catalog_ps['intermediate_data']['filename'], 
          catalog_ps['intermediate_data']['format'] )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Volume Transfer Matrix Steps

# COMMAND ----------

# MAGIC %md
# MAGIC #### Prepare VTM Data

# COMMAND ----------

vtm_input_data = prepare_vtm_data(x = full_data_for_PS, PS = full_PS, params = params_ps_vtm)
vtm_input_data

# COMMAND ----------

vtm_input_data.to_csv("/dbfs/mnt/upload/pos_purchase_structure_data/data/output/prepare_for_VTM_save.csv", index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create VTM Initial Object

# COMMAND ----------

skuM = create_vtm(vtm_data = vtm_input_data, params =  params_ps_vtm)
skuM

# COMMAND ----------

# MAGIC %md
# MAGIC #### Tone Down R Batch

# COMMAND ----------

X_sku_MT = tone_down_r_batch(skuM = skuM, vtm_data = vtm_input_data, params = params_ps_vtm )
X_sku_MT

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

# MAGIC %md
# MAGIC #### Run Full VTM

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
# MAGIC #### Calculate Flow Combinations In Batch

# COMMAND ----------

a = batch_calc_flow_combos(X_sku_MT = X_sku_MT, vtm_data = vtm_input_data, wr = wr,params = params_ps_vtm )
a

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Purchase Structure (Multi Attribute)

# COMMAND ----------

fx_comb = good_data_for_PS ['general_flag'] == 'GOOD'
fx_comb = fx_comb.reset_index(drop=True)


data_for_PS_comb =good_data_for_PS.copy()

res_comb_6 = create_structure(attr_df= good_data_for_PS, attrlist=attrlist_comb, filt_idx=fx_comb,nlevels=6, params=params_ps_vtm )
## get purchase structure
PS_comb_6 = res_comb_6[0]
interim_res_comb_6 = res_comb_6[1]

# COMMAND ----------

# Test with multiple atributes
attrlist_comb = np.array(["brand_nm", "packaging", "TYPE8","TYPE4","TYPE6","TYPE7","BRAND16","BRAND13","BRAND75"])
fx_comb = good_data_for_PS ['general_flag'] == 'GOOD'
fx_comb = fx_comb.reset_index(drop=True)

# COMMAND ----------


data_for_PS_comb =good_data_for_PS.copy()

# COMMAND ----------

res_comb = create_structure(attr_df= good_data_for_PS, attrlist=attrlist_comb, filt_idx=fx_comb,nlevels=params_ps_vtm["default_vars"]["nlevels"], params=params_ps_vtm )
## get purchase structure
PS_comb = res_comb[0]
interim_res_comb = res_comb[1]

PS_f =PS_comb_6.copy()
interim_res_f = interim_res.copy()

attrlist_btw = np.array(["MULTI","BRAND29"])
force_res =create_structure_between(attr_df= good_data_for_PS, purchase_struct_df =  PS_f, interim_result= interim_res_f,attrlist= attrlist_btw, filt_idx=fx_comb,lvl=2,nlevels = 4, params= params_ps_vtm ) 

# COMMAND ----------

PS_comb

# COMMAND ----------

interim_res_comb

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Purchase Structure (6 levels and Multi Attribute)

# COMMAND ----------

res_comb_6 = create_structure(attr_df= good_data_for_PS, attrlist=attrlist_comb, filt_idx=fx_comb,nlevels=6, params=params_ps_vtm )
## get purchase structure
PS_comb_6 = res_comb_6[0]
interim_res_comb_6 = res_comb_6[1]

# COMMAND ----------

PS_comb_6

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Create Structure (Force Atrributes)

# COMMAND ----------

PS_f =PS_comb_6.copy()

# COMMAND ----------

interim_res_f = interim_res.copy()

# COMMAND ----------

attrlist_btw = np.array(["MULTI","BRAND29"])
force_res =create_structure_between(attr_df= good_data_for_PS, purchase_struct_df =  PS_f, interim_result= interim_res_f,attrlist= attrlist_btw, filt_idx=fx_comb,lvl=2,nlevels = 4, params= params_ps_vtm ) 

# COMMAND ----------

## get purchase structure
PS_forced = force_res[0]
interim_res_forced = force_res[1]

# COMMAND ----------

PS_forced

# COMMAND ----------

interim_res_forced 
