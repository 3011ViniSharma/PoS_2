import numpy as np
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from pyspark.sql.types import *
from pyspark.sql.functions import *
from src.utils.logger import logger
from src_psvtm.models.vtm import *
from src_psvtm.models.create_structure import *


EWB_PROJECT_DIR = Path(Path.cwd()).parents[1] #change 0 to 1 if you have this notebook in a folder
name_of_ewb = 'rgmx-econometrics_gitlab' # econometrics directory
name_of_ps = 'rgmx_ewb-econometrics_clone' # purchase structure directory
sys.path.append(str(EWB_PROJECT_DIR / name_of_ewb )) # append econometrics directory to system directory
sys.path.append(str(EWB_PROJECT_DIR / name_of_ps )) # append purchase structure directory to system directory


######PS STUFF############
######Inputs
def check_default_vars_prod_attr_flags(df, params):
    try:
        df = df.toPandas()
    except:
        df = df
    vol = params['default_vars']['volume']
    sales = params['default_vars']['value_col']
    rev = params['default_vars']['value_col']
    item = params['default_vars']['item']
    check_cols = 0

    for col in [vol, sales, rev, item]:
        print(col)
        if col in df.columns:
            print(col)
            check_cols += 1
    assert check_cols == 4
    
def check_vars_exist_prod_attr_flags(df, params):
    var_select_vars = params['pos_purchase_structures']['data']['variable_selection']
    group_col_vars = params['pos_purchase_structures']['data']['grouping_cols']
    vtd_vars = list(params['pos_purchase_structures']['data']['variable_transformation_def'].keys())
    vars_to_check = list(set(var_select_vars + group_col_vars + vtd_vars))

    check_num = 0
    for var in vars_to_check:
        if var in df.columns:
            check_num += 1
        else:
            print(var)
    assert check_num == len(vars_to_check)
    
def verify_attributes_prod_attr_flags(df, params):
    try:
        df = df.toPandas()
    except:
        df = df
    check_num = 0
    full_list = []

    for prod_attr in params['pos_purchase_structures']['data']['variable_transformation_def'].keys():
        if 'what' in params['pos_purchase_structures']['data']['variable_transformation_def'][prod_attr]:
            full_list.extend(params['pos_purchase_structures']['data']['variable_transformation_def'][prod_attr]['what'])
            for attr in params['pos_purchase_structures']['data']['variable_transformation_def'][prod_attr]['what']:
                if attr.upper() in list(df[prod_attr]):
                    check_num += 1
        else:
            print(attr)
    assert check_num == len(full_list)
    
######Outputs
def verify_predefined_dummies(df, params):
    print("verify predef dumms")
    try:
        df = df.toPandas()
    except:
        df = df
    check_num = 0
    full_list = []
    for prod_attr in params['pos_purchase_structures']['data']['variable_transformation_def'].keys():
        if params['pos_purchase_structures']['data']['variable_transformation_def'][prod_attr]['how'] == 'predefined':
            print(params['pos_purchase_structures']['data']['variable_transformation_def'][prod_attr]['what'])
            full_list.extend(params['pos_purchase_structures']['data']['variable_transformation_def'][prod_attr]['what'])
            #full_list.append(prod_attr + "_other")
    for attr in full_list:
        if attr.upper() in df.columns:
            check_num += 1
        else:
            print(attr + " under column name " + prod_attr + " is not present")

    assert check_num == len(full_list)
    

def check_default_vals(df, params):
    print("check_default_vals")
    result_df = get_prod_attr_flags_df(input_data = df, params = params, dataframe=True)
    attrs_to_check = []
    transformed_vars = params['pos_purchase_structures']['data']['variable_transformation_def'].keys()
    print(transformed_vars)
    for var in transformed_vars:
        if params['pos_purchase_structures']['data']['variable_transformation_def'][var]['how'] in ["", "default"]:
            attrs_to_check.append(var + "_other")
            attrs_to_check.extend(list(result_df.groupby([var]).agg({
            params['default_vars']['value_col']: 'sum'
                }).sort_values(params['default_vars']['value_col'], ascending = False).reset_index()[:params['default_vars']['top']][var]))
            print("see extended list below")
            print(list(result_df.groupby([var]).agg({
            params['default_vars']['value_col']: 'sum'
                }).sort_values(params['default_vars']['value_col'], ascending = False).reset_index()[:params['default_vars']['top']][var]))
    cols_in = 0
    for attr in attrs_to_check:
        if attr in result_df.columns:
            cols_in += 1
        else:
            print(attr + " not in df cols")
    assert cols_in == len(attrs_to_check)
    
def check_custom_vals(df, params):
    result_df = get_prod_attr_flags_df(input_data = df, params = params, dataframe=True)
    attrs_to_check = []
    transformed_vars = params['pos_purchase_structures']['data']['variable_transformation_def'].keys()
    for var in transformed_vars:
        if params['pos_purchase_structures']['data']['variable_transformation_def'][var]['how'] in ["top"]:
                attrs_to_check.append(var + "_other")
                agg_var = params['pos_purchase_structures']['data']['variable_transformation_def'][var]['on']
                how_many = params['pos_purchase_structures']['data']['variable_transformation_def'][var]['top']
                attrs_to_check.extend(list(result_df.groupby([var]).agg({
                    agg_var: 'sum'
                }).sort_values(agg_var, ascending = False)[:how_many].reset_index()[var]))
    check_num = 0
    for attr in attrs_to_check:
        if attr in result_df.columns:
            check_num += 1
    assert check_num == len(attrs_to_check)

def check_pd_input_correct(input_data_flags_df, params): 
    cols_to_check = [
        params['default_vars']['volume'],
    ] + params['pos_purchase_structures']['data']['variable_selection'] + params['pos_purchase_structures']['data']['grouping_cols']
    drop_thresh = params['default_vars']['drop_thresh']

    cols_to_check = list(set(cols_to_check))
    assert drop_thresh > 0 and drop_thresh <= 1
    check_num = 0
    for col in cols_to_check:
        if col in input_data_flags_df.columns: 
            check_num += 1
    assert check_num == len(cols_to_check)
    


def test_pair_drops(input_data_flags_df, params, drop_cols = ['packaging', 'product_id'], pair_drops = [
    {'packaging': 'SINGLES', 'product_id': 'SKU954'},
    {'packaging': 'FULL CASE', 'product_id': 'SKU 1297'}
]):
    ind_pair_dfs = []
    num_to_drop = 0
    for pair in pair_drops: 
        #print(pair[drop_cols[0]], pair[drop_cols[1]])
        drop_part = input_data_flags_df[(input_data_flags_df[drop_cols[0]] == pair[drop_cols[0]] ) & ((input_data_flags_df[drop_cols[1]] == pair[drop_cols[1]]))][drop_cols]
        ind_pair_dfs.append(drop_part)
        num_to_drop += len(drop_part)

    test_pair_summary_drop = pd.concat(ind_pair_dfs).reset_index(drop = True)
    data_for_PS_test = drop_pairs(input_data_flags_df, test_pair_summary_drop, params)
    assert len(data_for_PS_test) == len(input_data_flags_df) - num_to_drop

    
###Inputs
def verify_opi_xpi_attr(input_data_flags_df, attr):
    in_df = 0
    if attr in input_data_flags_df.columns: 
        in_df +=1 
    assert in_df == 1
    
###Outputs
def verify_opi_xpi_output(opi_xpi_df, input_attr_df):
    assert len(opi_xpi_df.dropna()) == len(opi_xpi_df)
    cols_present = 0
    cols_right_type = 0
    cols_to_check = ['p_opi',	'p_xpi',	'd_opi'	,'d_xpi',	'v_opi',	'v_xpi']
    for col in cols_to_check:
        if col in opi_xpi_df.columns:
            cols_present += 1
        if opi_xpi_df[col].dtype == 'float64':
            cols_right_type += 1
    assert len(cols_to_check) == cols_present
    assert len(cols_to_check) == cols_right_type
    assert len(opi_xpi_df) == len(input_attr_df)
    
    
def verify_interim_res_dataframes(interim_res, params):
    nlevels = params['default_vars']['nlevels']
    colnames_needed = ['winner', 'pricevolume', 'distvolume', 'elasticity', 'price_gap', 'trend', 'season', 'share', 'TOT_INDEX', 'attributes', '_id']
    cols_present = 0
    for col in interim_res.columns:
        if col in colnames_needed:
            cols_present += 1
    assert cols_present == len(colnames_needed)
    logger.info(" interim_res columns all present")
    
def verify_PS_w_dataframes(PS, params):
    nlevels = params['default_vars']['nlevels']
    w_cols = ["W" + str(lvl) for lvl in range(1, nlevels + 1) ]
    subset_w_df = PS[w_cols]
    num_rows_valid = 0
    for col in subset_w_df.columns:
        for i in range(len(subset_w_df)):
            try:
                if type(subset_w_df[col][i]) == np.float64:
                    num_rows_valid += 1
                else:
                    float(subset_w_df[col][i])
                    num_rows_valid += 1
            except:
                if subset_w_df[col][i] == "-":
                    num_rows_valid += 1
            #if subset_w_df[col][i] == "-" or type(subset_w_df[col][i]) == np.float64:
                #num_rows_valid += 1
    #print(num_rows_valid)
    assert num_rows_valid == nlevels * len(PS)
    logger.info("all weight rows in PS valid")

def verify_PS_x_dataframes(PS, data_for_PS, params ):
    nlevels = params['default_vars']['nlevels']
    x_cols = ["X" + str(lvl) for lvl in range(1, nlevels + 1) ]
    PS_x_only = PS[x_cols]
    cols_satisfied = 0
    cand_attrs = []
    for var in params['pos_purchase_structures']['data']['variable_selection']:
        cand_attrs.extend(list(set(data_for_PS[var])))
    cand_attrs = [x.upper() for x in cand_attrs]
    for col in PS_x_only.columns:
        if col == "X1":
            if len(set(PS_x_only[col])) == 1 and PS_x_only[col][0] == 'Total':
                cols_satisfied += 1
                print("x1s")
        else:
            valid_rows = 0
            print(set(PS_x_only[col]))
            for i in range(len(PS_x_only)):
                #print(set(PS_x_only[col]))
                term_to_see = PS_x_only[col][i].upper()
                term_to_see = term_to_see.replace("NOT_", "")
                x_val_as_list = term_to_see.split("_")
                #print(x_val_as_list)
                #print(x_val_as_list)
                x_val_as_list = list(set(x_val_as_list))
                if "-" in x_val_as_list:
                    x_val_as_list.remove("-")
                check_attr_list_num = 0
                for attr in x_val_as_list:
                    if attr.upper() in cand_attrs:
                        check_attr_list_num += 1
                if check_attr_list_num == len(x_val_as_list):
                    valid_rows += 1
                elif len(x_val_as_list) == 1 and x_val_as_list[0] == "-":
                    valid_rows+=1
            if valid_rows == len(PS_x_only):
                cols_satisfied += 1
            else:
                print(x_val_as_list)
    print(cols_satisfied)
    assert cols_satisfied == len(x_cols)
    logger.info("all PS (X) rows in PS valid")
    
    
def test_interim_res_levels(interim_res, params):
    nlevels = params['default_vars']['nlevels']
    bad_ls = 0
    for l in set(interim_res['L']):
        if l > nlevels:
            bad_ls += 1
    assert bad_ls == 0
    
def verify_interim_res_winners(interim_res, data_for_PS, params):
    rows_good = 0
    cand_attrs = []
    for var in params['pos_purchase_structures']['data']['variable_selection']:
        cand_attrs.extend(list(set(data_for_PS[var])))
    print(cand_attrs)
    cand_attrs = [x.upper() for x in cand_attrs]
    for i in range(len(interim_res)):
        winner_as_list = interim_res['winner'][i].split("_")
        valid_items = 0
        for item in winner_as_list:
            if item.upper() in cand_attrs:
                valid_items += 1
        if valid_items == len(winner_as_list):
            rows_good += 1
    assert rows_good == len(interim_res)
#######VTM STUFF##########
#####Input Stuff
def test_PS_nlevels(PS, parameters):
    logger.info("starting test_PS_nlevels test")
    num_levels_PS = 0
    nlevels = parameters['default_vars']['nlevels']
    for i in range(1, nlevels+1):
        if "X" + str(i) in PS.columns:
            num_levels_PS += 1
    #print(num_levels_PS == nlevels)
    assert num_levels_PS == nlevels
    logger.info("correct number of PS and Weights given Number of levels")
    
def test_dps_vars(data_for_PS, parameters):
    logger.info("starting test_dps_vars test")
    vol_col = parameters['default_vars']['volume']
    item_col = parameters['default_vars']['item']
    value_col = parameters['default_vars']['value_col']
    lvl1 =  parameters['default_vars']['lvl1']
    lvl2 =  parameters['default_vars']['lvl2']
    type_col =  parameters['default_vars']['type_col']
    attrlist = parameters['default_vars']['attrlist']
    vars_to_check = attrlist + [vol_col, item_col, value_col, lvl1, lvl2, type_col, attrlist]
    check_num_vars = len(attrlist) + 6
    num_included = 0
    for col in vars_to_check:
        if col in list(data_for_PS.columns):
            num_included += 1
            print(col)

    assert check_num_vars == num_included
    logger.info("All specified Variables are Included")
    
def test_non_neg_vol(data_for_PS, parameters):
    logger.info("starting test_non_neg_vol test")
    vol_col_name = parameters['default_vars']['volume']
    vol_col = data_for_PS[vol_col_name]
    non_pos_vol = vol_col[vol_col <= 0]
    assert len(non_pos_vol) == 0
    logger.info("All vol values are positive")
    
def test_tone_down_vars(data_for_PS, PS, parameters):
    logger.info("starting test_tone_down_vars test")
    all_cols = list(data_for_PS.columns) + list(PS.columns)

    check_var = 0 
    tone_down_vars = parameters['vtm']['tone_down_vars']
    for var in tone_down_vars:
        if var in all_cols:
            check_var += 1
    print(check_var == len(tone_down_vars))
    assert check_var == len(tone_down_vars)
    logger.info("All tone down vars are included ")
    
def test_calc_flow_vars(data_for_PS, PS, parameters):
    logger.info("starting test_calc_flow_vars test")
    all_cols = list(data_for_PS.columns) + list(PS.columns)
    calc_flow_combos_list = parameters['vtm']['calc_flow_combos']['combos']
    calc_flow_vars_list = list(set([var for combo in calc_flow_combos_list for var in combo]))
    check_num = 0
    for var in calc_flow_vars_list:
        if var in all_cols:
            check_num += 1
    #print(check_num == len(calc_flow_vars_list))
    assert check_num == len(calc_flow_vars_list)
    logger.info("All calc flow vars are included ")
#####Output Stuff 
def test_proper_x_w_output(vtm_input_data, parameters):
    logger.info("starting test_proper_x_w_output test")
    ps_w_cols = [col for col in vtm_input_data.columns if len(col) == 2 and 'X' in col or 'W' in col]
    same_w_x = 0
    for i in range(1, parameters['default_vars']['nlevels']+1):
        if "X" + str(i) in ps_w_cols and "W" + str(i) in ps_w_cols:
            same_w_x += 1
    assert len(ps_w_cols) == parameters['default_vars']['nlevels'] * 2
    assert same_w_x == parameters['default_vars']['nlevels'] 
    logger.info("starting correct number of PS (X) and W columns")
    
def test_item_col(vtm_input_data, parameters):
    logger.info("Starting test_item_col fxn")
    item_col = parameters['default_vars']['item']
    print(len(list(vtm_input_data[item_col])) == len(set(vtm_input_data[item_col])) )
    assert len(list(vtm_input_data[item_col])) == len(set(vtm_input_data[item_col])) 
    logger.info("all unique skus in item col")
    
def test_no_neg_volume(vtm_input_data, parameters):
    logger.info("Starting test_no_neg_volume fxn")
    vol_list = list(vtm_input_data[parameters['default_vars']['volume']])
    non_pos_vols = [vol for vol in vol_list if vol <= 0]
    assert len(non_pos_vols) == 0
    logger.info("No negative volume")
    
    
def test_rows_cols_create_vtm(skuM): 
    logger.info("Starting test_rows_cols_create_vtm fxn ")
    assert len(list(skuM.index)) == len(set(skuM.index)) ##assert rows have unique skus
    logger.info("Rows all have unique skus")
    assert len(list(skuM.columns)) == len(set(skuM.columns)) ##assert cols have unique skus
    logger.info("Cols all have unique skus")
    assert list(skuM.index) == list(skuM.columns) #test order
    logger.info("Row length and order matches Col length and order")
    assert skuM.shape[0] == skuM.shape[1] #test num_rows = num_cols
    logger.info("Col and Row lengths match")
    
def test_create_vtm_diagonal_vtm_initial(skuM):
    logger.info("Starting test_create_vtm_diagonal_vtm_initial fxn ")
    col_row_match = 0
    for col in skuM.columns:
        if list(skuM.loc[col]) == list(skuM[col]):
            col_row_match += 1
    assert col_row_match == len(skuM.columns)
    logger.info("VTM initial is diagonally symmetrical")
    
def test_no_neg_values(skuM):
    assert len(skuM[skuM <= 0].dropna()) == 0
    logger.info("No negative values")
    
def assert_proper_tone_down_shape(X_sku_MT):
    logger.info("Starting assert_proper_tone_down_shape fxn")
    assert X_sku_MT.shape[0] == X_sku_MT.shape[1]  #num cols must match num rows
    logger.info("Num Rows Equals Num Columns")
    assert list(X_sku_MT.columns) == list(X_sku_MT.index) #same values and order
    logger.info("Rows and Columns have same value and order")
    assert len(list(X_sku_MT.columns)) == len(set(X_sku_MT.columns)) #assert columns have unique skus
    logger.info("Columns have unique skus")
    assert len(list(X_sku_MT.index)) == len(set(X_sku_MT.index))#assert rows have unique skus
    logger.info("Rows have unique skus")
    
def test_tone_down_diagonal(X_sku_MT):
    logger.info("Starting test_tone_down_diagonal fxn")
    col_row_match = 0
    for col in X_sku_MT.columns:
        if list(X_sku_MT.loc[col]) == list(X_sku_MT[col]):
            col_row_match += 1
    assert col_row_match == len(X_sku_MT.columns)
    logger.info("Diagonal Condition met")
    
def test_wr(wr):
    logger.info("starting test_wr fxn")
    assert len(list(wr.index)) == len(set(wr.index)) #each rowname must be a unique sku
    logger.info("Each rowname has a unique sku")
    assert len(wr[(wr < 0) & (wr > 1)].dropna()) == 0 ###no values less than 0 or greater than 1 
    logger.info("Every value is less than 0 but greater than 1")
    
    
def assert_share_wr_decimals(calc_flow_combos):
    logger.info("Starting assert_share_wr_decimals test")
    share_met = 0
    wr_met = 0 
    all_decimals = 0

    for df in calc_flow_combos:
        if 'share' in df.columns:
            share_met += 1
        if 'wr' in df.columns:
            wr_met += 1
        df_numeric = df._get_numeric_data()
        if len(df_numeric[(df_numeric < 0) & (df_numeric > 1)].dropna()) == 0:
            all_decimals += 1

    assert share_met == len(calc_flow_combos) ##test there's a share column
    logger.info("share col exists")
    assert wr_met == len(calc_flow_combos)###test there's a wr column 
    logger.info("walk rate col exists")
    assert all_decimals == len(calc_flow_combos) ##test all values between 0 and 1 inclusive
    logger.info("All vals between 0 and 1")
    
def test_calc_flow_Xcolnames(calc_flow_combos):
    logger.info("Starting test_calc_flow_Xcolnames test")
    df_good = 0
    for df in calc_flow_combos:
        Xcols = [col for col  in df.columns if col[0] == "X" and len(col) <= 3]

        concat_xs = []
        redux_col = []

        for col in df.columns:
            if type(col) == str:
                redux_col.append(col)
            elif type(col) == tuple:
                new_colname = ""
                for i in col:
                    new_colname += i + "_"
                redux_col.append(new_colname)
        for i in range(len(df)): 
            if len(Xcols) > 1:
                concat_x = ""
                for col in Xcols:
                    concat_x += df[col][i] + "_"
                concat_xs.append(concat_x)
            else:
                concat_xs.append(df[Xcols[0]][i])


        matches = 0

        for x in concat_xs:
            if x in redux_col:
                matches += 1

        if matches == len(df):
            df_good += 1
    assert df_good == len(calc_flow_combos)
    logger.info("all concatenated X's are colnames")
    
    
def test_vtm_rows_cols(VTM):
    logger.info("Starting test_vtm_rows_cols test")
    assert VTM.shape[0] +1  == VTM.shape[1] ##dimensions correct 
    logger.info("Dimenstions Correct")
    assert list(VTM.columns[1:]) == list(VTM.index) ##excluding first column (walk rates): column names match row names
    logger.info("Colnames match row names")
    assert VTM.columns[0] == 'wr' ##first col must be walk rates
    logger.info("Walk Rates are in first column")
    
def test_vtm_percentages(VTM):
    logger.info("Starting test_vtm_percentages test")
    assert len(VTM[(VTM < 0) & (VTM > 1)].dropna()) == 0
    logger.info("All values are between 0 and 100%")
    
def test_rowsums(VTM):
    logger.info("Starting test_rowsums test")
    one_test = 0
    for item in VTM.index:
        if VTM.loc[item].sum() >= 0.9999 or VTM.loc[item].sum() <= 1.0001:
            one_test += 1
    assert one_test == len(VTM)
    logger.info("All row Sums are Equal to 1")
    
def test_VTM_zero_diag(VTM):
    logger.info("Starting test_VTM_zero_diag test")
    VTM_check_df = VTM.drop(['wr'], axis = 1)
    bads = 0
    for item in VTM_check_df.columns:
        if VTM_check_df.loc[item][item] != 0:
            bads += 1
    assert bads == 0
    logger.info("Diagonal excluding wr column is all 0")