# Databricks notebook source
import pandas as pd
import numpy as np

###parameters
subcat_col = "subcategory"
subcategory_list = ["Disinfectant","Disinfectant","Bathroom","Bathroom","Bathroom","Kitchen","Kitchen","Flooor","Wipes","Wipes"]
subcat_mat = pd.read_csv("/dbfs/mnt/upload/pos_purchase_structure_data/data/input/subcategory_interaction.csv")
mat = pd.read_csv("/dbfs/mnt/upload/pos_purchase_structure_data/data/input/SKU_mat.csv")


##set up sku matrix
def get_selective_interaction(mat, subcat_mat, subcategory_list, subcat_col = 'subcategory'):
    
    """
    Description: gets original subcategory matrix and takes into account interactions based on overlapping subcategories 
    
    Inputs: mat: original matrix containing 
    subcat_mat: dataframe containing subcategories and whether (1) or not (0) they belong to another subcategory (overlapping)
    subcategory list: list of subcategories
    subcat_col: column name of subcategories in the subcat_matrix 
    
    Outputs: mat: dataframe containing multiplied interactions incorporated into initial mat input 
    """
    mat = mat.iloc[: , 1:]
    mat['index'] = mat.columns 
    mat = mat.set_index("index")
    np.fill_diagonal(mat.values, 0)

    subcats = subcat_mat[subcat_col]
    subcat_mat = subcat_mat.drop([subcat_col], axis = 1)
    subcat_mat['index'] = subcats
    subcat_mat = subcat_mat.set_index('index')

    new_mat = pd.DataFrame()
    for i in range(len(subcategory_list)):
        new_col_vals = []
        for j in range(len(subcategory_list)):
            mat.iloc[i,j] = mat.iloc[i,j] * subcat_mat.loc[subcategory_list[i],subcategory_list[j]]
    return(mat)


get_selective_interaction(
 mat = mat, 
 subcat_mat = subcat_mat,
 subcategory_list = subcategory_list,
 subcat_col = 'subcategory'
)

# COMMAND ----------

subcat_mat

# COMMAND ----------


