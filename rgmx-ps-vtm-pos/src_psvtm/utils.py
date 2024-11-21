import treelib
import pandas as pd
from pandas import DataFrame
from typing import Dict
import shutil
import time as time
from src.utils.logger import logger


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

def create_tree_df(purchase_struct:pd.DataFrame , params_ps_vtm:Dict):
    """
    Create a tree dataframe that can be proceessed by treelib to plot tree
    Args
        ps_output: purchase_structure_output
        params: parameter dictionary
    """
    parent_child_df = pd.DataFrame()
    nlevels = params_ps_vtm['default_vars']['nlevels']
    for curr_lvl in range(1,nlevels):
        selected_structure = purchase_struct[["X"+str(i) for i in range(1,curr_lvl+2)]].drop_duplicates()
        selected_structure['phrase'] = selected_structure[["X"+str(i) for i in range(1,curr_lvl+1)] ].apply(lambda x: _concat(x), axis=1)
        selected_structure['child_phrase'] = selected_structure[["X"+str(i) for i in range(1,curr_lvl+2)] ].apply(lambda x: _concat(x), axis=1)
        unique_phrases = selected_structure['phrase'].unique()

        for iter_phrase in unique_phrases:
            struct_to_split= selected_structure[selected_structure.phrase==iter_phrase]
            child_list = struct_to_split["X"+str(curr_lvl+1)].unique().tolist()
            for child in child_list:
                parent_child_df=parent_child_df.append(
                    pd.DataFrame({'parent':[struct_to_split["X"+str(curr_lvl)].unique()[0]],
                              'child':child,
                                 "phrase":iter_phrase,
                                 "child_phrase": struct_to_split[struct_to_split["X"+str(curr_lvl+1)]==child]["child_phrase"].unique()[0]}))

    
    parent_child_df= parent_child_df.apply(lambda x: x.astype(str).str.upper())
    parent_child_df=parent_child_df[parent_child_df.parent!="-"]
    parent_child_df=parent_child_df[parent_child_df.child!="-"]
    parent_child_df=parent_child_df.drop_duplicates()
    
    return (parent_child_df)
                
    
def plot_tree_function(tree_df:pd.DataFrame):
    """Function to create the tree object and plot it on the screen
    Args
        tree_df: PS tree object as a dataframe """
    
    # Create the tree
    tree = treelib.Tree()

    # Add the root node
    tree.create_node("TOTAL", "TOTAL")

    # Iterate through the DataFrame to add the child nodes and their relationships
    for index, row in tree_df.iterrows():
        tree.create_node(row['child'], row['child_phrase'], parent=row['phrase'])

    # Plot the tree
    tree.show()
    
def combine_excels(dataframe_dict,file_name):
    
    #initialze the excel writer
    try:
        writer = pd.ExcelWriter(file_name,engine='xlsxwriter')
    except:
        logger.info("Excel write failed. Check if package xlsxwriter is installed")

    #now loop thru and put each on a specific sheet
    for sheet, frame in  dataframe_dict.items(): 
        frame.to_excel(writer, sheet_name=sheet, startrow=0 , startcol=0)
        
    writer.save()
    
def save_to_excel(dataframe_dict, catalog_params,catalog_name):
    # save to temp location
    temp_filepath = '/local_disk0/tmp/{0}.xlsx'.format(catalog_params[catalog_name]['filename'])
    combine_excels(dataframe_dict,temp_filepath)
    
    # load catalog path
    filepath = catalog_params[catalog_name]['filepath']
    filename = catalog_params[catalog_name]['filename']
    format_file = catalog_params[catalog_name]['format']
    time_version = time.strftime("%Y%m%d-%H%M%S")
    real_path = filepath+filename+"/"+filename+"_"+time_version+"."+format_file
    
    #copy file to dbfs
    shutil.copyfile(temp_filepath, real_path)
    