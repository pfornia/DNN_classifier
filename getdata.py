#getdata.py

from __future__ import division # so that 1/2 = 0.5 and not 0
import urllib.request
import random
import copy
import numpy as np
import pandas as pd
import helpers
import re
import sys


# Dataset #1
def get_cancer_data(file_in):
    """
    Fetch the UCI data set on breast cancer characteristics 
    """

    data_text = open(file_in, 'r').read()
    data_rows = data_text.split('\n')
    data_rows = data_rows[0:-1] #last line is blank

    x_headers = ['id',
        'clump_thickness',
        'unif_cell_size',
        'unif_cell_shape',
        'marginal_adhesion',
        'single_epithelial_cell_size',
        'bare_nuclei',
        'bland_chrmatin',
        'normal_nucleoli',
        'mitoses',
        'target']

    cat_variables = ['target']
    
    data_all = [row.split(',') for row in data_rows]

    output = pd.DataFrame(data_all, columns = x_headers)

    output = output.drop(['id'], axis = 1)
    
    output = helpers.replace_missing_mode(output)

    for col in output: 
        if col not in cat_variables: output[col] = [float(x) for x in output[col]]

    output = helpers.one_hot_encode(output, exclude = [])
    output = helpers.normalize(output)
    
    return(output)


# Dataset #2
def get_glass_data(file_in):
    """
    Fetch the UCI data set on age of chemical characteristics of glass.
    """

    data_text = open(file_in, 'r').read()
    data_rows = data_text.split('\n')
    data_rows = data_rows[0:-1] #last line is blank

    x_headers = ['id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'target']

    cat_variables = ['target']
    
    data_all = [row.split(',') for row in data_rows]

    output = pd.DataFrame(data_all, columns = x_headers)

    output = helpers.replace_missing_mode(output)

    for col in output: 
        print(col)
        if col not in cat_variables: output[col] = [float(x) for x in output[col]]

    output = output.drop(['id'], axis = 1)
    
    output = helpers.one_hot_encode(output, exclude = [])
    output = helpers.normalize(output)
    
    return(output)


# Dataset #3
def get_iris_data(file_in):
    """
    Fetch the UCI data set on physical characteristics of Iris species.
    """

    data_text = open(file_in, 'r').read()
    data_rows = data_text.split('\n')
    data_rows = data_rows[0:-2] #last two lines are blank

    x_headers = ['sepal_length',
                 'sepal_width',
                 'petal_length',
                 'petal_width',
                 'target']

    cat_variables = ['target']
    
    data_all = [row.split(',') for row in data_rows]

    output = pd.DataFrame(data_all, columns = x_headers)

    output = helpers.replace_missing_mode(output)

    for col in output: 
        print(col)
        if col not in cat_variables: output[col] = [float(x) for x in output[col]]

    output = helpers.one_hot_encode(output, exclude = [])
    output = helpers.normalize(output)
    
    return(output)


# Dataset #4
def get_soy_data(file_in):
    """
    Fetch the UCI data set on diseases of soybean samples.
    """

    data_text = open(file_in, 'r').read()
    data_rows = data_text.split('\n')
    data_rows = data_rows[0:-1] #last line is blank

    x_headers = [
        'date',
        'plant-stand',
        'precip',
        'temp',
        'hail',
        'crop-hist',
        'area-damaged',
        'severity',
        'seed-tmt',
        'germination',
        'plant-growth',
        'leaves',
        'leafspots-halo',
        'leafspots-marg',
        'leafspot-size',
        'leaf-shread',
        'leaf-malf',
        'leaf-mild',
        'stem',
        'lodging',
        'stem-cankers',
        'canker-lesion',
        'fruiting-bodies',
        'external decay',
        'mycelium',
        'int-discolor',
        'sclerotia',
        'fruit-pods',
        'fruit spots',
        'seed',
        'mold-growth',
        'seed-discolor',
        'seed-size',
        'shriveling',
        'roots',
        'target'
    ]

    cat_variables = ['target']
    
    data_all = [row.split(',') for row in data_rows]

    output = pd.DataFrame(data_all, columns = x_headers)

    output = helpers.replace_missing_mode(output)

    output = helpers.one_hot_encode(output, exclude = [])
    output = helpers.normalize(output)

    return(output)


# Dataset #5
def get_vote_data(file_in):
    """
    Fetch and clean the UCI data set on US Representative vote records
    """

    data_text = open(file_in, 'r').read()
    data_rows = data_text.split('\n')
    data_rows = data_rows[0:-1] #last line is blank

    x_headers = ['target',
                 'handicapped-infants',
                 'water-project-cost-sharing',
                 'adoption-of-the-budget-resolution',
                 'physician-fee-freeze',
                 'el-salvador-aid',
                 'religious-groups-in-schools',
                 'anti-satellite-test-ban',
                 'aid-to-nicaraguan-contras',
                 'mx-missile',
                 'immigration',
                 'synfuels-corporation-cutback',
                 'education-spending',
                 'superfund-right-to-sue',
                 'crime',
                 'duty-free-exports',
                 'export-administration-act-south-africa'
                ]

    cat_variables = ['target']
    
    data_all = [row.split(',') for row in data_rows]

    output = pd.DataFrame(data_all, columns = x_headers)

    output = helpers.replace_missing_mode(output)

    output = helpers.one_hot_encode(output, exclude = [])
    output = helpers.normalize(output)

    return(output)

#If user provides all 4 arguments, skip the input steps.
if len(sys.argv) == 4:
    selection = int(sys.argv[1])
    filepath_in = sys.argv[2]
    filepath_out = sys.argv[3]
else:
    print("Please select dataset to download...")
    print("1. Breast Cancer")
    print("2. Glass")
    print("3. Iris")
    print("4. Soybean")
    print("5. Vote")
    selection = int(input("Selection: "))

#Select the appropriate cleaning function
if selection == 1:
    name = "cancer"
    get_data_func = get_cancer_data    
elif selection == 2:
    name = "glass"
    get_data_func = get_glass_data
elif selection == 3:
    name = "iris"
    get_data_func = get_iris_data
elif selection == 4:
    name = "soy"
    get_data_func = get_soy_data
elif selection == 5:
    name = "vote"
    get_data_func = get_vote_data

if len(sys.argv) != 4:
    filepath_in = input("Enter source path for raw %s data (plain text file): " % name)
    filepath_out = input("Enter destination path for %s data (.pkl file): " % name)

print("Fetching %s data..." % name)

#Perform Data cleaning.
this_data = get_data_func(filepath_in)
print(this_data)

#Save results to appropriate file location
this_data.to_pickle(filepath_out)


