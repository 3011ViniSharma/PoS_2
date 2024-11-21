# Databricks notebook source
!pip install pyspark==3.2.1
!pip install pandas==1.2.4
!pip install treelib
!pip install rpy2==3.5.4

# COMMAND ----------

# MAGIC %md
# MAGIC # Pipeline to create purchase structure & volume transfer matrix

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
# MAGIC ## Date Preparation From Econometrics Module

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

# base_loess = raw_loess_all.copy()


# write_obj(raw_loess_all, 
#           catalog_ps['input_data']['filepath'],
#           catalog_ps['input_data']['filename'], 
#           catalog_ps['input_data']['format'] )

# COMMAND ----------

# Used saved raw loess below
raw_loess_all = read_obj( catalog_ps['input_data']['filepath'],
          catalog_ps['input_data']['filename'], 
          catalog_ps['input_data']['format'])
base_loess = raw_loess_all.copy()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Structure Steps

# COMMAND ----------

#set the ewb parameters and catalog in ps structure for run loess
set_params_catalog_ewb(params_ewb,catalog_ewb) 

# COMMAND ----------

# MAGIC %md
# MAGIC ####Data Prep For Create Structure

# COMMAND ----------

# data_for_PS=custom_group_columns_df(base_loess, params_ps_vtm)
# data_for_PS.head()

# COMMAND ----------

# #write data for ps to storage
# write_obj(data_for_PS, 
#           catalog_ps['data_for_ps']['filepath'],
#           catalog_ps['data_for_ps']['filename'], 
#           catalog_ps['data_for_ps']['format'] )

# COMMAND ----------

#Used saved data for PS
data_for_PS = read_obj( catalog_ps['data_for_ps']['filepath'],
          catalog_ps['data_for_ps']['filename'], 
          catalog_ps['data_for_ps']['format'])

# COMMAND ----------

# Get the attribute list
attrlist = np.array(params_ps_vtm["default_vars"]["attrlist"])
attrlist

# COMMAND ----------

#Random sample 150 SKUS to for the Python and R comparison
import random
data_for_PS_all = data_for_PS.copy()
sku_list = data_for_PS_all.product_id.unique().tolist()
print(len(sku_list))
random_skus = random.sample(sku_list, 1000)
data_for_PS= data_for_PS_all[data_for_PS_all.product_id.isin(random_skus)]
data_for_PS.shape

# COMMAND ----------

# Subset for general flag that meet the about the criteria set in data prep as input
good_data_for_PS = data_for_PS[data_for_PS ['general_flag'] == 'GOOD'].reset_index(drop=True) 
fx = good_data_for_PS ['general_flag'] == 'GOOD'
fx = fx.reset_index(drop=True)
good_data_for_PS.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ####Create Purchase Structure for Mapped ("GOOD") Data

# COMMAND ----------

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

# COMMAND ----------

attr_df= good_data_for_PS; attrlist=np.array(params_ps_vtm["default_vars"]["attrlist"])[:3]; filt_idx=fx;nlevels= 2; params=params_ps_vtm 

# COMMAND ----------


parameters = params_ewb
CATALOG = catalog_ewb

# COMMAND ----------

attr_df= good_data_for_PS; attrlist=np.array(params_ps_vtm["default_vars"]["attrlist"])[:1]; filt_idx=fx;nlevels= 2; params=params_ps_vtm 
master_res_df=pd.DataFrame()

logger.info("\n Starting create structure \n")
thresh = params['default_vars']['create_struct_thresh']
vol_col = params['default_vars']['volume']
# Create dummy result df
fixed_cols = ["X" + str(i + 1) for i in range(nlevels)]
purchase_struct_df = pd.DataFrame(data="-", columns=fixed_cols, index=attr_df.index)
purchase_struct_df.iloc[:, 0] = "Total"  # first column is Total
if len(filt_idx) <= 1:
    filt_idx = pd.Series([True for i in range(len(attr_df))])

interim_res = []
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
        struc = [purchase_struct_df, interim_res]

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
            #struc = ps_break(input_df=attr_df, attrlist=attrlistx,filter_index=sltd_filt_idx, purchase_structure=purchase_struct_df,current_level=curr_lvl, params=params)
            ##############################################################################
            input_df=attr_df; attrlist=attrlistx;filter_index=sltd_filt_idx; purchase_structure=purchase_struct_df;current_level=curr_lvl; params=params
            
            logger.info("\n In price structure break \n")
            unq_ps_value = purchase_structure.iloc[:, 0:current_level+1].loc[filter_index.index, :].drop_duplicates()
            unq_ps_level = len(unq_ps_value)

            if unq_ps_level > 1:
                logger.info("\n not a unique purchase_structure level\n")
                logger.info(unq_ps_value)
                struc =  [purchase_structure, "not a unique purchase_structure level"]

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
            
            #ATTR IMP
            ############################################################################################################################################################
            input_df = input_filt_df; attrlist=attrlistx; input_filt=filt_flag_index; size_validated=0.075;useshare=True; params=params
            
            
            logger.info("\n Getting Attributes Importance Summary \n")
            weight_arr = list(params['default_vars']['attrsumm_wt']) #default [10, 3, 1, 2, 2, 2, 0.75])
            weight = np.array(weight_arr)
            
            #RES DF
            ############################################################################################################################################################
            attr_list = attrlist
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
                res_df =  None
            resbatch_rows = []
            resbatch_row_names = []

            for attr in filt_list:
                #perform attribute imputation and select summary result
                ############################################################################################################################################################
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
                data_for_loess_sh = data_for_loess_sh.replace([np.inf, -np.inf], np.nan).fillna(0)
                data_for_loess_sh[loess_target_col] = ownvol.fillna(0)
                data_for_loess_sh = pd.get_dummies(data_for_loess_sh, columns=[seasons_var], prefix='', prefix_sep='') #Get dummies for seasons
                data_for_loess_sh = data_for_loess_sh.replace([np.inf, -np.inf], np.nan).fillna(0)
                catalog = CATALOG
                # run_loess is from src/models
                loess_sh = run_loess(data_for_loess_raw=data_for_loess_sh, params=parameters, catalog=catalog)
                ############################################################################################################################################################
                target=res_calc_df; input_df=input_df; loess_sh=loess_sh; attr=attr; params=params; useshare=True; filt_index=outlier_filt
                
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
                # rename res_new to att imp df
                res_new = pd.DataFrame(res_rows)
                res_new[item_col] = res_rownames
                res_new = res_new.set_index(res_new[item_col])
                res_new = res_new.drop([item_col], axis=1)
                res_new['attr'] = attr_res

                if res_new.empty == True:
                    logger.info("ERROR: Empty df for %s in get_res_df ", attr)
                    res_df =  None
                res_new.columns = ["index_opi", "index_xpi", "index_odi", "index_xdi", "price_gap", "trend", "season",
                                   "elasticity_calc", "tot_vol", 'attr']
                res_df = res_new
                master_res_df = master_res_df.append(res_df)
                #END OF RES DF
                ############################################################################################################################################################
                
                #res_df = get_res_df(target=res_calc_df, input_df=input_df, loess_sh=loess_sh, attr=attr, params=params, useshare=True, filt_index=outlier_filt)
                if res_df is None:
                    att_df = [None, None]
                summary_res_df = get_agg_elasticities(res_df, attr=attr, capp=2, params=params)
                att_df = summary_res_df
                #END OF ATTR IMP
                ############################################################################################################################################################
#                 att_df = attr_imp(input_df=input_df, attr = attr, input_filt = input_filt,
#                                   params = params, useshare= useshare)[0]
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
                batch_attr_imp = None
            elif len(resbatch_rows) == 1:
                resbatch = pd.DataFrame(resbatch_rows[0])
            else:
                resbatch = pd.concat(resbatch_rows)

            resbatch['to_index'] = resbatch_row_names
            resbatch = resbatch.set_index('to_index')
            resbatch.columns = ['index_opi', 'index_xpi', 'index_odi', 'index_xdi', 'price_gap', 'trend', 'season',
                                'elasticity', 'ratio_price', 'ratio_dist', 'ratio_price_min', 'ratio_dist_min', 'ProdShare']
            batch_attr_imp = resbatch[resbatch['ProdShare'].notna()]
            
            #END OF ATTR BATCH
            ############################################################################################################################################################
            # Get attribute importance batch results
#             batch_attr_imp = attr_imp_batch(input_df = input_df, attr_list = attrlist,
#                                 input_filt = input_filt, params=params, useshare=useshare)

            #return batch_attr_imp
            if batch_attr_imp is None:
                attr_summ = None
            if len(batch_attr_imp) == 0:
                attr_summ = None

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
                attr_summ = [score]

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
            
            attr_summ = [score_table, batch_attr_imp]
            #END OF ATTR SUMMARY
            ############################################################################################################################################################
            
            if attr_summ is None:
                logger.info("No valid attributes for this break")
                struc = [purchase_structure, "No valid attributes for this break"]

            if len(attr_summ) == 1:
                logger.info("No valid attributes for this break")
                struc=  [purchase_structure, "No valid attributes for this break"]

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
            
            struc = [purchase_structure, interim_scoretable]
            
            #end of PS BREAK
            ##############################################################################

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

interim_result = pd.concat(interim_res, ignore_index=True)

# COMMAND ----------

testt = pd.concat([target,input_df[vol_col]],axis=1)
print(testt)
testt['own_vol'] = testt.net_sales_volume /( testt.v_opi + testt.v_xpi)

print(set(testt['own_vol'] == own_vol))

testt[testt['own_vol'].isna()]

# COMMAND ----------

set(loess_sh['log_net_sales_volume'] == own_vol)

# COMMAND ----------

set(loess_sh['log_net_sales_volume'] == np.log(loess_sh['net_sales_volume']))

# COMMAND ----------

print("coeffs" , cf_pv)

set(np.log(own_vol_df.loc[non_zero_idx.index, 'own_vol']) == log_own_vol)
#len(log_own_vol) - len(own_vol)
own_vol_df.loc[non_zero_idx.index, 'own_vol'].unique()

# COMMAND ----------

set(y_cf_ts_own == np.log(input_df.loc[non_zero_idx.index, vol_col]))

# COMMAND ----------

purchase_struct_df

# COMMAND ----------

interim_result

# COMMAND ----------

score_table


# COMMAND ----------

batch_attr_imp

# COMMAND ----------

master_res_df.attr.unique()
master_res_df[master_res_df.attr=="TYPE5"]

# COMMAND ----------

loess_sh.log_net_sales_volume

# COMMAND ----------

res = create_structure(attr_df= good_data_for_PS, attrlist=attrlist, filt_idx=fx,nlevels= params_ps_vtm["default_vars"]["nlevels"], params=params_ps_vtm )

## get purchase structure
PS = res[0]
interim_res = res[1]

