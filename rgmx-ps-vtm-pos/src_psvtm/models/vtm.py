import numpy as np
import pandas as pd

def prepare_vtm_data(x, PS,  params):
    
    """
    Description: used to create initial VTM data
    
    Inputs:
    x: dataframe (likely to be data_for_PS) containing skus with their attributes across dates, volumes, binary vars,  probabilities, and elasticity columns
    PS: dataframe containing the X and W's in addition to other product attributes
    params: dict containing at least VTM parameters
   
    Outputs:
    x: dataframe used for VTM creation in the later functions
    """
    
    ##start config parameters
    lvl1 = params['default_vars']['lvl1']
    lvl2 = params['default_vars']['lvl2']
    lvl3 = params['default_vars']['lvl3']
    value_col = params['default_vars']['value_col']
    item_col = params['default_vars']['item']
    vol_col = params['default_vars']['volume']
    #type_col = params['default_vars']['type_col']
    attrlist = params['default_vars']['attrlist']
    ##end config parameters
    #cols_for_VTM = attrlist + [lvl1, lvl2, item_col, vol_col, value_col, type_col]
    if lvl3 == "":
        lvls_cols = [lvl1, lvl2]
    else:
        lvls_cols = [lvl1, lvl2, lvl3]
    cols_for_VTM = attrlist + lvls_cols + [item_col, vol_col, value_col]
    cols_for_VTM = list(set(cols_for_VTM))


    x = pd.concat([x, PS], axis =1)
    cols_keep = list(PS.columns) + cols_for_VTM
    cols_keep = [col for col in cols_keep if col in x.columns]
    #x = x[cols_keep]

    #create filter to identify data for last 52 weeks and anything greater than 52 weeks set vol to 0. This is implemented to retain all ppgs
    x['period_id'] = pd.to_datetime(x['period_id'])
    x["timeperiod_rank"] = x['period_id'].rank(method='dense', ascending=False)
    #x = x.loc[x['timeperiod_rank'] <= 52].reset_index()
    x.loc[x['timeperiod_rank']>52, vol_col] = 0.00001
   
    agged_lvl_vols = x.groupby(lvls_cols).agg({
        vol_col: 'sum'
    }).reset_index()
    agged_lvl_vols.columns = lvls_cols + [ vol_col + "_copy"]
    x = x.drop([value_col, vol_col], axis = 1)
    merged_df = pd.merge(x, agged_lvl_vols, how = 'left', on = lvls_cols)
    merged_df = merged_df.drop_duplicates().reset_index(drop = True).replace("-", np.nan)
    merged_df = merged_df.rename(columns={vol_col + "_copy": vol_col})
    merged_df = merged_df.sort_values(lvls_cols+["W1"], ascending= [True, True, True]).groupby(lvls_cols).first().reset_index() 
    return(merged_df)

def create_vtm(vtm_data: pd.DataFrame, params: dict):
    """
    Fxn Description: used to create Initial VTM object based on purchase structures from input data and weighted proportionally based on volume and initial weights 
    
    Inputs:
    vtm_data: dataframe with X (PS's) and W's (used for weightage) and features of skus
    params: dict (that is the config) containing for at least vtm  
    
    Outputs:
    I: dataframe that's the initial VTM structure 
    """
    
    #####params start
    vol_col = params['default_vars']['volume']
    item_col = params['default_vars']['item']
    nlevels = params['default_vars']['nlevels']
    lvl1 = params['default_vars']['lvl1']
    lvl2 = params['default_vars']['lvl2']
    lvl3 = params['default_vars']['lvl3']
    #####params end
    
    ## additional checks to remove vol col duplicates
    vtm_data = vtm_data.loc[:, ~vtm_data.columns.duplicated()].copy()

    uid_col_values = []
    for i in range(len(vtm_data)):
        if lvl3 == "":
            uid_val = vtm_data[lvl1][i] + " | " + vtm_data[lvl2][i]
        else:
            uid_val = vtm_data[lvl1][i] + " | " + vtm_data[lvl2][i]  + "_" + vtm_data[lvl3][i]
        uid_col_values.append(uid_val)
    PS_cols = []
    wtd_cols = []
    weight_columns = []
    x_columns = []
    assert len(x_columns) == len(weight_columns)
    all_cols_included = x_columns + weight_columns
    for i in range(1, nlevels + 1):
        if "X" + str(i) in vtm_data.columns:
            x_columns.append("X" + str(i))
        if "W" + str(i) in vtm_data.columns:
            weight_columns.append("W"+str(i))
    PS = vtm_data[x_columns]
    wtd = vtm_data[weight_columns].astype(np.float64).fillna(1)

    vol_df = pd.DataFrame(vtm_data[vol_col], columns = [vol_col])
    I = pd.DataFrame(np.ones((len(PS), len(PS))))
    I.columns = uid_col_values
    I['index'] = uid_col_values
    I = I.set_index('index')
    PS = pd.DataFrame(PS)
    ps = PS.iloc[:, 0]

    for L in range(len(PS.columns) - 1):
        ps = ps + " - "+ PS.iloc[:, L + 1].astype(str)
        ####for each level at each possible purchase structure, see where the 
        for u in list(set(ps[PS.iloc[:, L + 1] != "-"])):
            #find indexes where 
            ps_np = np.array(ps)
            if u in list(ps):
                #to_keep = list(ps).index(u)
                #print(to_keep)
                to_keep = [i for i, x in enumerate(ps_np) if x == u]

                I.iloc[to_keep, to_keep] = np.array(I.iloc[to_keep, to_keep]) * np.array(wtd.iloc[to_keep, L+1])

    fairshare = np.outer(vol_df[vol_col], vol_df[vol_col])
    I = I / (I * fairshare).sum().sum() * fairshare.sum()  * 100  
    I = I.round(2)


    return(I)
            
def tone_down_r(Msku, aggr, ratio = 0.9):
    """
    Description: used to caliberate the switching interaction. 
    Generally used when we check the VTMs and we find the volume flows should be controlled for certain attributes b/c of consumer preference OR transfers are different because of Volume/Value shares 
    
    
    Inputs: 
    Msku: dataframe that is the same diagonally with num_rows = num_cols = num_skus involved  
    aggr: dataframe column (series) that mentions what we aggregate by for determining what to keep and what to "tone-down by with" ratio
    ratio: ratio with what we tone down by for the given variables in the aggr series above 
    
    Outputs:
    MskuNew: dataframe containing adjusted Mskus with 
    """
    
    aggr = aggr.fillna("")
    u = list(set(aggr))
    MskuNew = Msku * ratio
    for a in range(len(u)):
        to_keep = [i for i, x in enumerate(aggr) if x == u[a]]
        MskuNew.iloc[to_keep, to_keep] = Msku.iloc[to_keep,to_keep]
    return MskuNew 



def calc_walkrate(I, vtm_data, params):
    """
    Description: this function is used to calculate walkrates 
    
    Inputs:
    I: dataframe with num cols = num rows = num unique skus (must be diagonally parallel)
    vtm_data: dataframe containing data of PS and Weights along with attributes
    params: dict (that is the config) containing for at least vtm 
    
    Outputs:
    wr: dataframe containing walkrates by sku (sku is index)
    """
    ###params start
    vol_col = params['default_vars']['volume']
    sim_share = vtm_data[vol_col].fillna(0.001)
    base_share = vtm_data[vol_col].fillna(0.001)
    wr_avg_ = params['vtm']['walkrate']['wr_avg_gbl_var']
    wr_size_factor_ = params['vtm']['walkrate']['wr_size_gbl_var']
    wr_shape_factor_ = params['vtm']['walkrate']['wr_shape_gbl_var']
    wr_index_factor_ = params['vtm']['walkrate']['wr_index_gbl_var']
    rescale =  params['vtm']['walkrate']['rescale']
    ###params end

    sku_cols = I.columns
    I = np.array(I)
    np.fill_diagonal(I, 0)
    I = pd.DataFrame(I, columns = sku_cols)
    I['index'] = sku_cols
    I = I.set_index('index')
    if rescale:
        sim_share = sim_share / sim_share.sum()
        base_share = base_share / base_share.sum()
    if wr_index_factor_ != 1:
        I = (I/100) ** wr_index_factor_ * 100
    base_share = pd.DataFrame([bs if bs >= 0 else 0 for bs in list(base_share)], columns = [vol_col])
    size_index = np.array((base_share * I.shape[0]) ** wr_size_factor_)
    per_sims = pd.DataFrame(np.multiply(I, np.array(sim_share))).T.sum() / 100
    filter_0 = [i for i, v in enumerate(per_sims) if v == 0]
    per_sims[filter_0] = size_index[filter_0].reshape(-1)
    loy_index = 1 / np.array(per_sims).T * size_index.T
    tr_loy_index = (loy_index / ( 1 + loy_index)) * wr_shape_factor_
    tr_median = np.median(tr_loy_index)
    log_factor = np.log(wr_avg_) / np.log(tr_median)
    wr = tr_loy_index ** log_factor
    if type(wr[0]) == np.ndarray: #additional check as original expectation was 1D array but observed as 2D causing errors
        wr[0][filter_0] = 1
    else:
        wr[filter_0] = 1
    wr = pd.DataFrame(wr).T
    wr['sku'] = sku_cols
    wr= wr.set_index("sku")
    wr.columns = ['wr']
    return wr 

def vtm(index, vtm_data, wr, params):
    """
    Description: generates  final output used to create tree
    
    Inputs:
    index: dataframe  with num cols = num rows = num unique skus
    vtm_data: dataframe containing data of PS and Weights along with attributes
    wr: dataframe series containing walk rates by sku (sku is index)
    
    Output:
    zz: dataframe containing VTM output between attributes plus walk rate by sku as the first column
    """
    
    ####params start
    item_col = params['default_vars']['item']
    vol_col = params['default_vars']['volume']
    Volume = vtm_data[vol_col]
    vtm_factor = params['vtm']['structure']['vtm_factor']
    ####params end
    
    sku_cols = index.columns
    assert np.shape(index)[0] == np.shape(index)[1]
    index = np.array(index)
    di = np.diag_indices(np.shape(index)[0])
    index[di] = 0
    size = np.outer(Volume, Volume)
    size = size/size.sum()
    
    refactored_index = (100 * (index/100) ** vtm_factor) * size

    refactored_index = pd.DataFrame((np.array(pd.DataFrame(refactored_index)) / np.array(pd.DataFrame(refactored_index)).sum(axis = 0)).T) * (1 - np.array(wr))
    refactored_index.columns = sku_cols
    #print(refactored_index)
    #refactored_index = pd.concat([wr.reset_index(), refactored_index], axis = 1).set_index(item_col)
    refactored_index = pd.concat([wr.reset_index(), refactored_index], axis = 1)
    #zz = zz_test.set_index(item_col)
    #refactored_index[item_col]  = sku_cols
    # refactored_index = refactored_index.set_index(item_col)
    #refactored_index = refactored_index.rename(columns={ refactored_index.columns[0]: "wr" })
    if "sku" in refactored_index.columns:
        refactored_index = refactored_index.rename(columns = {'sku': item_col})
    refactored_index = refactored_index.set_index(item_col)
    return refactored_index


def tone_down_r_batch(skuM, vtm_data, params):
    """
    Description: used to calculate multiple tonedown variables at once -- see params file for how to input this in 
    
    Inputs:
    skuM: dataframe object that contains initial VTM 
    vtm_data: dataframe containing data of PS and Weights along with attributes 
    params: dict containing parameters with vtm attributes 
    
    Outputs: 
    X_sku_MT_current: dataframe object that has all the toned down variables if applicable 
    """
    ###start_params
    tone_down = params['vtm']['tone_down_vars']
    ###end_params
    
    if len(tone_down) == 0:
        return(skuM)
    X_sku_MT_current = tone_down_r(Msku = skuM, aggr = vtm_data[list(tone_down.keys())[0]], ratio = tone_down[list(tone_down.keys())[0]])
    if len(tone_down) == 1:
        return(X_sku_MT_current)
    for var in [var for var,ratio in tone_down.items()][1:]:
        X_sku_MT_next = tone_down_r(Msku = X_sku_MT_current, aggr = vtm_data[var], ratio = tone_down[var])
        X_sku_MT_current = X_sku_MT_next
    return X_sku_MT_current

def run_full_vtm(vtm_data, params): 
    """
    Description: runs full VTM pipeline from initial creation of VTM object to final output used to create tree 
    
    Inputs:
    vtm_data: dataframe containing data of PS and Weights along with attributes
    params: dict (that is the config) containing for at least vtm 
    
    Outputs:
    VTM: dataframe that is final VTM object with SKUX-SKUY VTM numbers 
    
    """
    ####params_start
    vol_col = params['default_vars']['volume']
    fill_na_var = params['vtm']['structure']['fill_vol_na']
    ####params_end
    
    ## extra checks to ensure no duplicated vol col
    vtm_data = vtm_data.loc[:, ~vtm_data.columns.duplicated()].copy()

    skuM = create_vtm(vtm_data, params = params)
    X_sku_MT = tone_down_r_batch(skuM = skuM, vtm_data = vtm_data, params = params)
    I = X_sku_MT
    sku_cols = I.columns
    vtm_data[vol_col] = vtm_data[vol_col].fillna(fill_na_var)
    sku_cols = X_sku_MT.columns
    X_sku_MT = np.array(X_sku_MT)
    np.fill_diagonal(X_sku_MT, 0)
    X_sku_MT = pd.DataFrame(X_sku_MT, columns = sku_cols)
    X_sku_MT['index'] = sku_cols
    X_sku_MT = X_sku_MT.set_index('index')
    
    wr = calc_walkrate(I, vtm_data, params = params)
    VTM = vtm(X_sku_MT, vtm_data, wr, params = params)
    return VTM


def calc_flow_between(Msku,vtm_data, wr, agg_cols, params):
    """
    Description: 
    
    Inputs: 
    Msku: dataframe object that contains initial VTM and any intermediate tonedown variables 
    vtm_data: dataframe containing data of PS and Weights along with attributes 
    wr: dataframe containing walkrates by sku 
    agg_cols: list of cols we are aggregating by for determining calc_flow_between 
    
    Outputs:
    mat: dataframe object containing share calculations for the 1 attribute in the agg_cols list 

    """
    ####start parameters
    vol_col = params['default_vars']['volume']
    factor = params['vtm']['structure']['vtm_factor']
    item_col =  params['default_vars']['item']
    volume = vtm_data[vol_col]
    ####end parameters 
    
    agging_df = pd.DataFrame()
    for col in agg_cols:
        agging_df[col] = vtm_data[col]
    
    sku_cols = Msku.columns
    Msku = np.array(Msku)
    volume = [vol if vol >= 0.0001 else 0.0001 for vol in list(volume)]
    np.fill_diagonal(Msku, 0)
    sizeM = np.outer(np.array(volume), np.array(volume))
    np.fill_diagonal(sizeM, 0)
    vtm = sizeM * Msku ** factor
    vtm = np.array(pd.DataFrame((vtm / vtm.sum(axis = 1)) * np.array(1 - wr)).T)
    vtm_for_agging = pd.DataFrame(vtm).T
    vtm_for_agging = pd.concat([vtm_for_agging, agging_df], axis = 1)
    vtm_for_agging[item_col] = sku_cols
    vtmaggr = pd.DataFrame(vtm_for_agging).groupby(agg_cols, dropna=False).sum()
   
    vtmaggr.columns = sku_cols
    vol_df = pd.DataFrame(volume, columns = ["vol"])
    vol_df = pd.concat([vol_df, agging_df], axis = 1)
    tot_vol = vol_df.vol.sum()
    share = vol_df.groupby(agg_cols, dropna=False).sum() / tot_vol
    

    new_wr = wr.T
    new_wr['wr'] = "wr"
    new_wr = new_wr.set_index("wr")
    x = pd.concat([vtmaggr,new_wr])

    ##for new vtmaggr
    denom = vol_df.groupby(agg_cols, dropna=False).agg({
        'vol': 'sum'
    })
    x = pd.concat([vtmaggr,new_wr])
    x_times_v = x * volume
    x_times_v2 = x_times_v.T.reset_index()
    x_times_v2 = pd.concat([x_times_v2, agging_df], axis =1)
    x_times_v2 =  x_times_v2.rename(columns={'index': item_col}) 
    x_times_v2= x_times_v2.set_index(item_col)
    new_vtm_aggr = x_times_v2.groupby(agg_cols, dropna=False).sum() / np.array(denom)
    mat = pd.concat([new_vtm_aggr, share], axis = 1)
    #mat.columns = [*mat.columns[:-1], 'share'] #continue to retain vol% column
    mat = mat.reset_index()

    # Add index value to compare within group transfer
    non_flow_cols_excl_wr = np.append(agg_cols, ['wr', 'vol']) 
    index_col = (np.diag(mat.iloc[:, ~mat.columns.isin(non_flow_cols_excl_wr)])) / mat['vol']
    mat['index'] = index_col
    return(mat)

def batch_calc_flow_combos(X_sku_MT, vtm_data, wr, params):
    
    """
    Description: used to calculate flow between in batch 
  
    Inputs:
    X_sku_MT: dataframe object that has all the toned down variables if applicable 
    vtm_data: dataframe containing data of PS and Weights along with attributes  
    wr: dataframe containing walkrates by sku
    params: dict containing at least relevant VTM parameters 
    
    Outputs:
    calc_flows: dict containing dataframes of calc_flow_between calculations specified in the calc_flow_combos list
    """
    
    ####parameters start####
    combos = params['vtm']['calc_flow_combos']['combos']
    ####parameters end####
    calc_flows = {}
    for agg_cols in combos:
        a = calc_flow_between(Msku = X_sku_MT, vtm_data = vtm_data, wr = wr, agg_cols = agg_cols,  params = params)
        calc_flows['_'.join(agg_cols)] = a
    return(calc_flows)

def get_selective_interaction(vtm_data, subcat_mat, mat, params):    
    """
    Description: gets original subcategory matrix and takes into account interactions based on overlapping subcategories 
    
    Inputs: vtm_input_data: data used for creating vtm sku matrix
    mat: original sku matrix
    subcat_mat: dataframe containing subcategories and whether (1) or not (0) they belong to another subcategory (overlapping)
    params: dict containing parameters
    
    Outputs: mat: dataframe containing multiplied interactions incorporated into initial mat input 
    """
    
    assert subcat_mat.columns[0] in vtm_data.columns, "Column name invalid."
    
    lvl1 = params['default_vars']['lvl1']
    lvl2 = params['default_vars']['lvl2']
    lvl3 = params['default_vars']['lvl3']

    if lvl3 == "":
        lvls_cols = [lvl2, lvl1]
    else:
        lvls_cols = [lvl3, lvl2, lvl1]

    
    subcat_col = subcat_mat.columns[0]
    vtm_data["id_vtm"] = vtm_data[lvls_cols].agg("_".join, axis=1)
    subcategory_list = vtm_data[["id_vtm",subcat_col]].drop_duplicates().sort_values(by = "id_vtm")[subcat_col].to_list()

    subcats = subcat_mat[subcat_col]
    subcat_mat = subcat_mat.drop([subcat_col], axis = 1)
    subcat_mat['index'] = subcats
    subcat_mat = subcat_mat.set_index('index')

    new_mat = mat.copy(deep=True)
    
    for i in range(len(subcategory_list)):
        for j in range(len(subcategory_list)):
            new_mat.iloc[i,j] = new_mat.iloc[i,j] * subcat_mat.loc[subcategory_list[i],subcategory_list[j]]
    
    return(new_mat)