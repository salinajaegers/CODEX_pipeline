
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as clr

import argparse
import base64
import io
import json
import os
import random
import re
import warnings
from copy import deepcopy
from functools import reduce

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import torch
import torch.nn.functional as F
from dash.dependencies import Input, Output, State
from skimage.filters import threshold_li, threshold_mean
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from class_dataset import RandomCrop, Subtract, ToTensor, myDataset
from load_data import DataProcesser
from utils_app import frange, model_output_app


import sys
import seaborn as sns
import pandas as pd
import numpy as np
import pytorch_lightning as pl

import results_model
from utils import model_output

def parseArguments_overlay():
    parser = argparse.ArgumentParser(description='Project the CNN features with tSNE and browse interactively.')
    parser.add_argument('-m', '--model', help='str, path to the model file', type=str)
    parser.add_argument('-do', '--dataoriginal', help='str, path to the data file', type=str)
    parser.add_argument('-dn', '--datanew', help='str, path to the data file', type=str)
    parser.add_argument('-s', '--set', help='str, set to project. Must be one of ["all", "train", "validation", "test"]. Default to "all".',
                        type=str, default='all')
    parser.add_argument('-b', '--batch', help='int, size of the batch when passing data to the model. '
                                              'Increase as high as GPU memory allows for speed up.'
                                              ' Must be smaller than the number of series selected,'
                                              ' Default is set to 1/10 of the dataset size.',
                        type=int, default=-1)
    parser.add_argument('--measurement', help='list of str, names of the measurement variables. In DataProcesser convention,'
                                              ' this is the prefix in a column name that contains a measurement'
                                              ' (time being the suffix). Pay attention to the order since this is'
                                              ' how the dimensions of a sample of data will be ordered (i.e. 1st in'
                                              ' the list will form 1st row of measurements in the sample,'
                                              ' 2nd is the 2nd, etc...). Leave empty for automatic detection.',
                        type=str, default='', nargs='*')
    parser.add_argument('--seed', help='int, seed for random, ensures reproducibility. Default to 7.',
                        type=int, default=7)
    parser.add_argument('--start', help='int, start time range for selecting data. Useful to ignore part of the '
                                        'data were irrelevant measurement were acquired. Set to -1 for automatic detection.',
                        type=int, default=-1)
    parser.add_argument('--end', help='int, end time range for selecting data. Useful to ignore part of the '
                                        'data were irrelevant measurement were acquired. Set to -1 for automatic detection.',
                        type=int, default=-1)
    parser.add_argument('-p', '--port', help='int, port on which to start the application.', type=int, default=8050)
    parser.add_argument('--host', help='str, host name for the application. Default to "127.0.0.1".', type=str, default='127.0.0.1')
    args = parser.parse_args()
    return(args)

args = parseArguments_overlay()
# ----------------------------------------------------------------------------------------------------------------------
# Inputs
myseed = args.seed
np.random.seed(myseed); random.seed(myseed); torch.manual_seed(myseed)
# Parameters
original_data = args.dataoriginal
new_data = args.datanew

start_time = None if args.start==-1 else args.start
end_time = None if args.end==-1 else args.end
measurement = None if args.measurement=='' else args.measurement
selected_classes = None
perc_selected_ids = 1  # Select only percentile of all trajectories, not always useful to project them all and slow
batch_sz = args.batch  # set as high as GPU memory can handle for speed up

rand_crop = True
set_to_project = args.set  # one of ['all', 'train', 'validation', 'test']

n_pca = 2
# ----------------------------------------------------------------------------------------------------------------------
# Model Loading
model_file = args.model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load(model_file) if torch.cuda.is_available() else torch.load(model_file, map_location='cpu')
net.eval()
net.double()
net = net.to(device)
length = net.length
print('LENGTH IS ', length)

# ----------------------------------------------------------------------------------------------------------------------
data_file = original_data

# Data Loading, Subsetting, Preprocessing
data = DataProcesser(data_file, datatable=False)
measurement = data.detect_groups_times()['groups'] if measurement is None else measurement
start_time = data.detect_groups_times()['times'][0] if start_time is None else start_time
end_time = data.detect_groups_times()['times'][1] if end_time is None else end_time
data.subset(sel_groups=measurement, start_time=start_time, end_time=end_time)

# Check that the measurements columns are numeric, if not try to convert to float64
cols_to_check = '^(?:{})'.format('|'.join(measurement))  # ?: for non-capturing group
cols_to_check = data.dataset.columns.values[data.dataset.columns.str.contains(cols_to_check)]
cols_to_change = [(s,t) for s,t in zip(cols_to_check, data.dataset.dtypes[cols_to_check]) if not pd.api.types.is_numeric_dtype(data.dataset[s])]
if len(cols_to_change) > 0:
    warnings.warn('Some measurements columns are not of numeric type. Attempting to convert the columns to float64 type. List of problematic columns: {}'.format(cols_to_change))
    try:
        cols_dict = {s[0]:'float64' for s in cols_to_change}
        data.dataset = data.dataset.astype(cols_dict)
    except ValueError:
        warnings.warn('Conversion to float failed for at least one column.')

data.get_stats()
if selected_classes is not None:
    data.dataset = data.dataset[data.dataset[data.col_class].isin(selected_classes)]
# Suppress the warning that data were not processed, irrelevant for the app
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    data.split_sets(which='dataset')

print('Start time: {}; End time: {}; Measurement: {}'.format(start_time, end_time, measurement))

# ----------------------------------------------------------------------------------------------------------------------
# df used to plot measurements
assert set_to_project in ['all', 'train', 'validation', 'test']
if set_to_project == 'all':
    df = deepcopy(data.dataset)
elif set_to_project == 'train':
    df = deepcopy(data.train_set)
elif set_to_project == 'validation':
    df = deepcopy(data.validation_set)
elif set_to_project == 'test':
    df = deepcopy(data.test_set)
print('Size of raw dataframe: {}'.format(df.shape))
df.rename(columns={data.col_id: 'ID', data.col_class: 'Class'}, inplace = True)
selected_ids = np.random.choice(df.loc[:,'ID'].unique(), round(perc_selected_ids * df.shape[0]), replace=False)
df = df[df['ID'].isin(selected_ids)]
print('Size of selected dataframe: {}'.format(df.shape))

if batch_sz == -1:
    batch_sz = round(df.shape[0]/10)
print('Batch size: {}'.format(batch_sz))
assert batch_sz <= df.shape[0]
# Split for each measurement, melt and append.
ldf = []
for meas in measurement:
    col_meas = [i for i in df.columns if re.match('^{}_'.format(meas), i)]
    temp = df[['ID', 'Class'] + col_meas].melt(['ID', 'Class'])
    temp['Time'] = temp['variable'].str.extract('([0-9]+$)')
    temp['variable'] = temp['variable'].str.replace('_[0-9]+$', '', regex=True)
    ldf.append(temp)
df = pd.concat(ldf)
del temp
del ldf
df.sort_values(['ID', 'Time'])

# ----------------------------------------------------------------------------------------------------------------------
# Prepare dataloader for t-SNE, set high batch_size to speed up
subtract_numbers = [data.stats['mu'][meas]['train'] for meas in measurement]
if rand_crop:
    ls_transforms = transforms.Compose([
        Subtract(subtract_numbers),
        RandomCrop(output_size=length, ignore_na_tails=True, export_crop_pos=True),
        ToTensor()])
else:
    ls_transforms = transforms.Compose([
        Subtract(subtract_numbers),
        ToTensor()])

mydataset = myDataset(dataset=data.dataset[data.dataset['ID'].isin(selected_ids)], transform=ls_transforms)
mydataloader = DataLoader(dataset=mydataset,
                          batch_size=batch_sz,
                          shuffle=False,
                          drop_last=False)

# ----------------------------------------------------------------------------------------------------------------------
# Classes object definition
classes = tuple(data.classes[data.col_classname])
classes_col = data.classes[data.col_classname]
if selected_classes is not None:
    classes = [i for i in classes if i
               in list(data.classes[data.classes[data.col_class].isin(selected_classes)][data.col_classname])]
    classes = tuple(classes)
    classes_dict = data.classes[data.classes[data.col_class].isin(selected_classes)].to_dict()[data.col_classname]
else:
    classes_dict = data.classes.to_dict()[data.col_classname]

net.batch_size = batch_sz  # Learn representations over the whole dataset at once if equal to dataset length

# ----------------------------------------------------------------------------------------------------------------------

model = net
dataloader = mydataloader

df_out = model_output_app(model, dataloader, export_prob=True, export_feat=True, device=device, export_crop_pos=rand_crop)
df_out['Class'].replace(classes_col, inplace=True)
feat_cols = [i for i in df_out.columns if i.startswith('Feat_')]
feature_blobs_array = np.array(df_out[feat_cols])
pca = PCA(n_components=n_pca)
pca_original = pca.fit_transform(feature_blobs_array)


#label = np.array(df_out['Class'])
identifier = np.array(df_out['ID'])
df_raw = pd.DataFrame(pca_original)
#df['Class'] = pd.DataFrame(label)
#df = np.column_stack((a,b))
df_original = df_raw.join(df_out)
df_original.to_csv('/data/users/sjaegers/research_project/scripts/pca_original.csv', index=False)
#df_cut_original = df_original.sample(5000)
df_cut_original = df_original

#["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00"]
colors = {'A':"#3B9AB2",'B':"#F21A00"}



#plt.scatter(df_cut_original[0], df_cut_original[1], c=df_cut_original['ID'].astype(str).str[0].map(colors), alpha=0.1)
plt.scatter(df_cut_original[0], df_cut_original[1], alpha=0.1, c=df_cut_original['ID'].astype(str).str[0].map(colors))
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA Plot')
custom_lines = [Line2D([0], [0], color=colors['A'], lw=4),
                Line2D([0], [0], color=colors['B'], lw=4)]
plt.legend(custom_lines, ['WT', 'H1047R'], loc='upper left', bbox_to_anchor=(1.04, 1))

plt.savefig('pca_plot_original.pdf', format='pdf', bbox_inches="tight")

# Close the plot
plt.close()


#Create interactive plot
#df_cut = df_original.sample(1000)
#colors = {'H':'red', 'L':'yellow','A':'blue','B':'green'}
#
#
#components = df_cut[0:n_pca]
#labels = {
#    str(i): f"PC {i+1} ({var:.1f}%)"
#    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
#}
#
#fig2 = px.scatter_matrix(
#    components,
#    labels=labels,
#    dimensions=range(n_pca),
#    color=df['ID'].astype(str).str[0].map(colors), 
#    alpha=0.3
#)
#fig2.update_traces(diagonal_visible=False)
#fig2.show()
















# ----------------------------------------------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load(model_file) if torch.cuda.is_available() else torch.load(model_file, map_location='cpu')
net.eval()
net.double()
net = net.to(device)
length = net.length


data_file = new_data
# Data Loading, Subsetting, Preprocessing
data = DataProcesser(data_file, datatable=False)
measurement = data.detect_groups_times()['groups'] if measurement is None else measurement
start_time = data.detect_groups_times()['times'][0] if start_time is None else start_time
end_time = data.detect_groups_times()['times'][1] if end_time is None else end_time
data.subset(sel_groups=measurement, start_time=start_time, end_time=end_time)

# Check that the measurements columns are numeric, if not try to convert to float64
cols_to_check = '^(?:{})'.format('|'.join(measurement))  # ?: for non-capturing group
cols_to_check = data.dataset.columns.values[data.dataset.columns.str.contains(cols_to_check)]
cols_to_change = [(s,t) for s,t in zip(cols_to_check, data.dataset.dtypes[cols_to_check]) if not pd.api.types.is_numeric_dtype(data.dataset[s])]
if len(cols_to_change) > 0:
    warnings.warn('Some measurements columns are not of numeric type. Attempting to convert the columns to float64 type. List of problematic columns: {}'.format(cols_to_change))
    try:
        cols_dict = {s[0]:'float64' for s in cols_to_change}
        data.dataset = data.dataset.astype(cols_dict)
    except ValueError:
        warnings.warn('Conversion to float failed for at least one column.')

data.get_stats()
if selected_classes is not None:
    data.dataset = data.dataset[data.dataset[data.col_class].isin(selected_classes)]
# Suppress the warning that data were not processed, irrelevant for the app
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    data.split_sets(which='dataset')

print('Start time: {}; End time: {}; Measurement: {}'.format(start_time, end_time, measurement))

# ----------------------------------------------------------------------------------------------------------------------
# df used to plot measurements
assert set_to_project in ['all', 'train', 'validation', 'test']
if set_to_project == 'all':
    df = deepcopy(data.dataset)
elif set_to_project == 'train':
    df = deepcopy(data.train_set)
elif set_to_project == 'validation':
    df = deepcopy(data.validation_set)
elif set_to_project == 'test':
    df = deepcopy(data.test_set)
print('Size of raw dataframe: {}'.format(df.shape))
df.rename(columns={data.col_id: 'ID', data.col_class: 'Class'}, inplace = True)
selected_ids = np.random.choice(df.loc[:,'ID'].unique(), round(perc_selected_ids * df.shape[0]), replace=False)
df = df[df['ID'].isin(selected_ids)]
print('Size of selected dataframe: {}'.format(df.shape))

if batch_sz == -1:
    batch_sz = round(df.shape[0]/10)
print('Batch size: {}'.format(batch_sz))
assert batch_sz <= df.shape[0]
# Split for each measurement, melt and append.
ldf = []
for meas in measurement:
    col_meas = [i for i in df.columns if re.match('^{}_'.format(meas), i)]
    temp = df[['ID', 'Class'] + col_meas].melt(['ID', 'Class'])
    temp['Time'] = temp['variable'].str.extract('([0-9]+$)')
    temp['variable'] = temp['variable'].str.replace('_[0-9]+$', '', regex=True)
    ldf.append(temp)
df = pd.concat(ldf)
del temp
del ldf
df.sort_values(['ID', 'Time'])

# ----------------------------------------------------------------------------------------------------------------------
# Prepare dataloader for t-SNE, set high batch_size to speed up
subtract_numbers = [data.stats['mu'][meas]['train'] for meas in measurement]
if rand_crop:
    ls_transforms = transforms.Compose([
        Subtract(subtract_numbers),
        RandomCrop(output_size=length, ignore_na_tails=True, export_crop_pos=True),
        ToTensor()])
else:
    ls_transforms = transforms.Compose([
        Subtract(subtract_numbers),
        ToTensor()])

mydataset = myDataset(dataset=data.dataset[data.dataset['ID'].isin(selected_ids)], transform=ls_transforms)
mydataloader = DataLoader(dataset=mydataset,
                          batch_size=batch_sz,
                          shuffle=False,
                          drop_last=False)

# ----------------------------------------------------------------------------------------------------------------------
# Classes object definition
classes = tuple(data.classes[data.col_classname])
classes_col = data.classes[data.col_classname]
if selected_classes is not None:
    classes = [i for i in classes if i
               in list(data.classes[data.classes[data.col_class].isin(selected_classes)][data.col_classname])]
    classes = tuple(classes)
    classes_dict = data.classes[data.classes[data.col_class].isin(selected_classes)].to_dict()[data.col_classname]
else:
    classes_dict = data.classes.to_dict()[data.col_classname]

net.batch_size = batch_sz  # Learn representations over the whole dataset at once if equal to dataset length

# ----------------------------------------------------------------------------------------------------------------------

model = net
dataloader = mydataloader

df_out = model_output_app(model, dataloader, export_prob=True, export_feat=True, device=device, export_crop_pos=rand_crop)
df_out['Class'].replace(classes_col, inplace=True)
feat_cols = [i for i in df_out.columns if i.startswith('Feat_')]
feature_blobs_array = np.array(df_out[feat_cols])
new_pca = pca.transform(feature_blobs_array)


#label = np.array(df_out['Class'])
identifier = np.array(df_out['ID'])
df_raw = pd.DataFrame(new_pca)
#df['Class'] = pd.DataFrame(label)
#df = np.column_stack((a,b))
df_new = df_raw.join(df_out)
df_new.to_csv('/data/users/sjaegers/research_project/scripts/pca_new.csv', index=False)
#df_cut_new = df_new.sample(5000)
df_cut_new = df_new

#["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00"]
colors = {'H':"#EBCC2A", 'L':"#E1AF00"}



#plt.scatter(df_cut_new[0], df_cut_new[1], c=df_cut_new['ID'].astype(str).str[0].map(colors), alpha=0.1)
plt.scatter(df_cut_new[0], df_cut_new[1], alpha=0.1, c=df_cut_new['ID'].astype(str).str[0].map(colors))
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA Plot')
custom_lines = [Line2D([0], [0], color=colors['H'], lw=4),
                Line2D([0], [0], color=colors['L'], lw=4)]
plt.legend(custom_lines, ['High Dose', 'Low Dose'], loc='upper left', bbox_to_anchor=(1.04, 1))

plt.savefig('pca_plot_new.pdf', format='pdf', bbox_inches="tight")

# Close the plot
plt.close()



#label = np.array(df_out['Class'])
#identifier = np.array(df_out['ID'])
#df_raw = pd.DataFrame(new_pca)
##df['Class'] = pd.DataFrame(label)
##df = np.column_stack((a,b))
#
#
##put together the pcas and the rest of the data information
#df = df_raw.join(df_out)
#df.to_csv('./pca.csv', index=False)
#
#
##Create interactive plot
#df_cut = df.sample(1000)
#colors = {'H':'red', 'L':'yellow','A':'blue','B':'green'}
#
#
#components = df_cut[0:n_pca]
#labels = {
#    str(i): f"PC {i+1} ({var:.1f}%)"
#    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
#}
#
#fig2 = px.scatter_matrix(
#    components,
#    labels=labels,
#    dimensions=range(n_pca),
#    color=df['ID'].astype(str).str[0].map(colors), 
#    alpha=0.3
#)
#fig2.update_traces(diagonal_visible=False)
#fig2.show()




df_cut_all = df_cut_original.append(df_cut_new)

#["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00"]
colors = {'H':"#EBCC2A", 'L':"#E1AF00",'A':"#3B9AB2",'B':"#F21A00"}



#plt.scatter(df_cut_all[0], df_cut_all[1], c=df_cut_all['ID'].astype(str).str[0].map(colors), alpha=0.1)
plt.scatter(df_cut_all[0], df_cut_all[1], alpha=0.1, c=df_cut_all['ID'].astype(str).str[0].map(colors))
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA Plot')
custom_lines = [Line2D([0], [0], color=colors['H'], lw=4),
                Line2D([0], [0], color=colors['L'], lw=4),
                Line2D([0], [0], color=colors['A'], lw=4),
                Line2D([0], [0], color=colors['B'], lw=4)]
plt.legend(custom_lines, ['High Dose', 'Low Dose', 'WT', 'H1047R'], loc='upper left', bbox_to_anchor=(1.04, 1))

plt.savefig('pca_plot_all.pdf', format='pdf', bbox_inches="tight")

# Close the plot
plt.close()
