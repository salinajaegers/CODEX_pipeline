from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as clr


import torch
import sys
from torch.utils.data import DataLoader
from torchvision import transforms
import seaborn as sns
import pandas as pd
import os
import numpy as np
import plotly.express as px
import pytorch_lightning as pl

import yaml

import re
import warnings
from copy import deepcopy

# Custom functions/classes
path_to_module = snakemake.params.scripts  # Path where all the .py files are, relative to the notebook folder
#path_to_module = './'
sys.path.append(path_to_module)

from class_dataset import RandomCrop, Subtract, ToTensor, myDataset
from load_data import DataProcesser
from utils_app import frange, model_output_app
import results_model
from utils import model_output

with open('./config.yml', 'r') as file:
    config_file = yaml.safe_load(file)

config_training = config_file['training']
config_pca = config_file['pca']
name = str(config_file['name'])

# ----------------------------------------------------------------------------------------------------------------------
# Inputs
myseed = config_pca['seed']
torch.manual_seed(myseed)
torch.cuda.manual_seed(myseed)
np.random.seed(myseed)

# Parameters
data_file = snakemake.params.zip
model_file = snakemake.params.model
#data_file = './ERKH/ERKH.zip'
#model_file = './model/ERKKTR_model.pytorch'


start_time = None
end_time = None
measurement = None if config_training['measurement']=='' else config_training['measurement']
selected_classes = None
perc_selected_ids = 1  # Select only percentile of all trajectories, not always useful to project them all and slow
batch_sz = config_pca['batch']  # set as high as GPU memory can handle for speed up

rand_crop = True
set_to_project = 'all'  # one of ['all', 'train', 'validation', 'test']

#n_pca = config_pca['n_pca']
n_pca = 2
#---------------------------------------------------------------------------------------------------
# Model Loading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load(model_file) if torch.cuda.is_available() else torch.load(model_file, map_location='cpu')
net.eval()
net.double()
net = net.to(device)
length = net.length
print('LENGTH IS ', length)

# ----------------------------------------------------------------------------------------------------------------------

# Data Loading, Subsetting, Preprocessing
data = DataProcesser(data_file, datatable=False)
measurement = data.detect_groups_times()['groups'] if measurement is None else measurement
start_time = data.detect_groups_times()['times'][0] if start_time is None else start_time
end_time = data.detect_groups_times()['times'][1] if end_time is None else end_time
data.subset(sel_groups=measurement, start_time=start_time, end_time=end_time)

data.get_stats()
data.split_sets()
classes = tuple(data.classes.iloc[:,1])
dict_classes = data.classes[data.col_classname]

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

out_dir = './results_' + name + '/pca'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


df_raw = pd.DataFrame(pca_original)
df_original = df_raw.join(df_out)
df_original.to_csv(out_dir + '/pca_all.csv', index=False)
df_original = pd.DataFrame(df_original)

Nclasses = len(np.unique(df_original['Class']))
#cmap = plt.cm.get_cmap('hsv')
cmap = clr.LinearSegmentedColormap.from_list('Zissou1', ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00"], N=100)



label_color_dict = {label: cmap(np.linspace(0,1,Nclasses))[idx] for idx, label in enumerate(np.unique(df_original['Class']))}
#colors = [cmap(label_color_dict[label]) for label in df_original['Class']]

colors = [label_color_dict[label] for label in df_original['Class']]
#colors = df_original['Class'].astype(str).map(colors)


plt.scatter(df_original[0], df_original[1], alpha=0.1, c=colors)
custom_lines = [Line2D([0], [0], color=cmap(np.linspace(0,1,Nclasses))[i], lw=4) for i, cl in enumerate(cmap(np.linspace(0,1,Nclasses)))]
#custom_lines = [Line2D([0], [0], color=label_color_dict[cl], lw=4) for cl in label_color_dict.keys()]
plt.legend(custom_lines, label_color_dict.keys(), loc='upper left', bbox_to_anchor=(1.04, 1))

# Add the axis labels
plt.xlabel('PC 1 (%.2f%%)' % (pca.explained_variance_ratio_[0]*100))
plt.ylabel('PC 2 (%.2f%%)' % (pca.explained_variance_ratio_[1]*100))
plt.title('PCA-Plot')
plt.savefig(out_dir + '/pca_all.pdf', format='pdf', bbox_inches="tight")
# Close the plot
plt.close()

# Helper functions for plotting
col_id = data.col_id
col_class = data.col_class
col_classname = data.col_classname

# Minimal confidence for uncorrelated prototypes
perc_selected_ids = config_pca['perc_selected_ids']  # Select only percentile of all trajectories, not always useful to project them all and slow
threshold_confidence = config_pca['threshold_confidence']
length_data = len(pd.DataFrame(df_original))
npoint = round(perc_selected_ids * length_data)
ntop = npoint
nworst = npoint
nuncor = npoint
nrandom = npoint

# ### Top trajectories per class
tops = results_model.top_confidence_perclass(model, dataloader, n=ntop, labels_classes=dict_classes)

df_tops = df_original.loc[df_original['ID'].isin(tops['ID'])]
df_tops.to_csv(out_dir + '/pca_tops_' + str(config_pca['perc_selected_ids']) + '.csv', index=False)


colors = [label_color_dict[label] for label in df_tops['Class']]
plt.scatter(df_tops[0], df_tops[1], alpha=0.1, c=colors)

custom_lines = [Line2D([0], [0], color=cmap(np.linspace(0,1,Nclasses))[i], lw=4) for i, cl in enumerate(cmap(np.linspace(0,1,Nclasses)))]
plt.legend(custom_lines, label_color_dict.keys(), loc='upper left', bbox_to_anchor=(1.04, 1))

# Add the axis labels
plt.xlabel('PC 1 (%.2f%%)' % (pca.explained_variance_ratio_[0]*100))
plt.ylabel('PC 2 (%.2f%%)' % (pca.explained_variance_ratio_[1]*100))
plt.title('PCA-Plot Tops')
plt.savefig(out_dir + '/pca_tops_' + str(config_pca['perc_selected_ids']) + '.pdf', format='pdf', bbox_inches="tight")
# Close the plot
plt.close()


# ### Least correlated set per class
# When choosing uncorrelated curves, pick only from curves for which the model confidence in the input class is at least "threshold_confidence".
uncorr = results_model.least_correlated_set(model, dataloader, n=nuncor, labels_classes=dict_classes, threshold_confidence=threshold_confidence)

df_uncorr = df_original.loc[df_original['ID'].isin(uncorr['ID'])]
df_uncorr.to_csv(out_dir + '/pca_uncorr_' + str(config_pca['perc_selected_ids']) + '.csv', index=False)

colors = [label_color_dict[label] for label in df_uncorr['Class']]
plt.scatter(df_uncorr[0], df_uncorr[1], alpha=0.1, c=colors)

custom_lines = [Line2D([0], [0], color=cmap(np.linspace(0,1,Nclasses))[i], lw=4) for i, cl in enumerate(cmap(np.linspace(0,1,Nclasses)))]
plt.legend(custom_lines, label_color_dict.keys(), loc='upper left', bbox_to_anchor=(1.04, 1))

# Add the axis labels
plt.xlabel('PC 1 (%.2f%%)' % (pca.explained_variance_ratio_[0]*100))
plt.ylabel('PC 2 (%.2f%%)' % (pca.explained_variance_ratio_[1]*100))
plt.title('PCA-Plot Uncorrelated')
plt.savefig(out_dir + '/pca_uncorr_' + str(config_pca['perc_selected_ids']) + '.pdf', format='pdf', bbox_inches="tight")
# Close the plot
plt.close()



# ### Worst trajectory per class
worsts = results_model.worst_classification_perclass(model, dataloader, n=nworst, labels_classes=dict_classes)

df_worsts = df_original.loc[df_original['ID'].isin(worsts['ID'])]
df_worsts.to_csv(out_dir + '/pca_worsts_' + str(perc_selected_ids) + '.csv', index=False)

colors = [label_color_dict[label] for label in df_worsts['Class']]
plt.scatter(df_worsts[0], df_worsts[1], alpha=0.1, c=colors)

custom_lines = [Line2D([0], [0], color=cmap(np.linspace(0,1,Nclasses))[i], lw=4) for i, cl in enumerate(cmap(np.linspace(0,1,Nclasses)))]
plt.legend(custom_lines, label_color_dict.keys(), loc='upper left', bbox_to_anchor=(1.04, 1))

# Add the axis labels
plt.xlabel('PC 1 (%.2f%%)' % (pca.explained_variance_ratio_[0]*100))
plt.ylabel('PC 2 (%.2f%%)' % (pca.explained_variance_ratio_[1]*100))
plt.title('PCA-Plot Worsts')
plt.savefig(out_dir + '/pca_worsts_' + str(perc_selected_ids) + '.pdf', format='pdf', bbox_inches="tight")
# Close the plot
plt.close()



# ### Random sample
# Get a random sample of trajectories.
randoms_ids = []
for classe in data.validation_set['class'].unique():
    randoms_ids += list(data.dataset.loc[data.dataset['class']==classe]['ID'].sample(nrandom))
randoms = model_output(model, dataloader)
randoms = randoms.loc[randoms['ID'].isin(randoms_ids)]

df_randoms = df_original.loc[df_original['ID'].isin(randoms['ID'])]
df_randoms.to_csv(out_dir + '/pca_randoms_' + str(perc_selected_ids) + '.csv', index=False)

colors = [label_color_dict[label] for label in df_randoms['Class']]
plt.scatter(df_randoms[0], df_randoms[1], alpha=0.1, c=colors)

custom_lines = [Line2D([0], [0], color=cmap(np.linspace(0,1,Nclasses))[i], lw=4) for i, cl in enumerate(cmap(np.linspace(0,1,Nclasses)))]
plt.legend(custom_lines, label_color_dict.keys(), loc='upper left', bbox_to_anchor=(1.04, 1))
# Add the axis labels
plt.xlabel('PC 1 (%.2f%%)' % (pca.explained_variance_ratio_[0]*100))
plt.ylabel('PC 2 (%.2f%%)' % (pca.explained_variance_ratio_[1]*100))
plt.title('PCA-Plot Randoms')
plt.savefig(out_dir + '/pca_randoms_' + str(perc_selected_ids) + '.pdf', format='pdf', bbox_inches="tight")
# Close the plot
plt.close()









#-----------------------------------------------------------------------------------------------------------
# Inputs
myseed = config_pca['seed']
torch.manual_seed(myseed)
torch.cuda.manual_seed(myseed)
np.random.seed(myseed)

# Parameters
data_file = snakemake.params.zip
model_file = snakemake.params.model
#data_file = './ERKH/ERKH.zip'
#model_file = './model/ERKKTR_model.pytorch'


start_time = None
end_time = None
measurement = None if config_training['measurement']=='' else config_training['measurement']
selected_classes = None
perc_selected_ids = 1  # Select only percentile of all trajectories, not always useful to project them all and slow
batch_sz = config_pca['batch']  # set as high as GPU memory can handle for speed up

rand_crop = True
set_to_project = 'test'  # one of ['all', 'train', 'validation', 'test']

#n_pca = config_pca['n_pca']
n_pca = 2
#---------------------------------------------------------------------------------------------------
# Model Loading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load(model_file) if torch.cuda.is_available() else torch.load(model_file, map_location='cpu')
net.eval()
net.double()
net = net.to(device)
length = net.length
print('LENGTH IS ', length)

# ----------------------------------------------------------------------------------------------------------------------

# Data Loading, Subsetting, Preprocessing
data = DataProcesser(data_file, datatable=False)
measurement = data.detect_groups_times()['groups'] if measurement is None else measurement
start_time = data.detect_groups_times()['times'][0] if start_time is None else start_time
end_time = data.detect_groups_times()['times'][1] if end_time is None else end_time
data.subset(sel_groups=measurement, start_time=start_time, end_time=end_time)

data.get_stats()
data.split_sets()
classes = tuple(data.classes.iloc[:,1])
dict_classes = data.classes[data.col_classname]

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

out_dir = './results_' + name + '/pca'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


df_raw = pd.DataFrame(pca_original)
df_original = df_raw.join(df_out)
df_original.to_csv(out_dir + '/pca_all_test.csv', index=False)
df_original = pd.DataFrame(df_original)

Nclasses = len(np.unique(df_original['Class']))
#cmap = plt.cm.get_cmap('hsv')
cmap = clr.LinearSegmentedColormap.from_list('Zissou1', ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00"], N=100)



label_color_dict = {label: cmap(np.linspace(0,1,Nclasses))[idx] for idx, label in enumerate(np.unique(df_original['Class']))}
#colors = [cmap(label_color_dict[label]) for label in df_original['Class']]

colors = [label_color_dict[label] for label in df_original['Class']]
#colors = df_original['Class'].astype(str).map(colors)


plt.scatter(df_original[0], df_original[1], alpha=0.1, c=colors)
custom_lines = [Line2D([0], [0], color=cmap(np.linspace(0,1,Nclasses))[i], lw=4) for i, cl in enumerate(cmap(np.linspace(0,1,Nclasses)))]
#custom_lines = [Line2D([0], [0], color=label_color_dict[cl], lw=4) for cl in label_color_dict.keys()]
plt.legend(custom_lines, label_color_dict.keys(), loc='upper left', bbox_to_anchor=(1.04, 1))

# Add the axis labels
plt.xlabel('PC 1 (%.2f%%)' % (pca.explained_variance_ratio_[0]*100))
plt.ylabel('PC 2 (%.2f%%)' % (pca.explained_variance_ratio_[1]*100))
plt.title('PCA-Plot Test')
plt.savefig(out_dir + '/pca_all_test.pdf', format='pdf', bbox_inches="tight")
# Close the plot
plt.close()









#-----------------------------------------------------------------------------------------------------------
# Inputs
myseed = config_pca['seed']
torch.manual_seed(myseed)
torch.cuda.manual_seed(myseed)
np.random.seed(myseed)

# Parameters
data_file = snakemake.params.zip
model_file = snakemake.params.model
#data_file = './ERKH/ERKH.zip'
#model_file = './model/ERKKTR_model.pytorch'


start_time = None
end_time = None
measurement = None if config_training['measurement']=='' else config_training['measurement']
selected_classes = None
perc_selected_ids = 1  # Select only percentile of all trajectories, not always useful to project them all and slow
batch_sz = config_pca['batch']  # set as high as GPU memory can handle for speed up

rand_crop = True
set_to_project = 'validation'  # one of ['all', 'train', 'validation', 'test']

#n_pca = config_pca['n_pca']
n_pca = 2
#---------------------------------------------------------------------------------------------------
# Model Loading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load(model_file) if torch.cuda.is_available() else torch.load(model_file, map_location='cpu')
net.eval()
net.double()
net = net.to(device)
length = net.length
print('LENGTH IS ', length)

# ----------------------------------------------------------------------------------------------------------------------

# Data Loading, Subsetting, Preprocessing
data = DataProcesser(data_file, datatable=False)
measurement = data.detect_groups_times()['groups'] if measurement is None else measurement
start_time = data.detect_groups_times()['times'][0] if start_time is None else start_time
end_time = data.detect_groups_times()['times'][1] if end_time is None else end_time
data.subset(sel_groups=measurement, start_time=start_time, end_time=end_time)

data.get_stats()
data.split_sets()
classes = tuple(data.classes.iloc[:,1])
dict_classes = data.classes[data.col_classname]

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

out_dir = './results_' + name + '/pca'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


df_raw = pd.DataFrame(pca_original)
df_original = df_raw.join(df_out)
df_original.to_csv(out_dir + '/pca_all_validation.csv', index=False)
df_original = pd.DataFrame(df_original)

Nclasses = len(np.unique(df_original['Class']))
#cmap = plt.cm.get_cmap('hsv')
cmap = clr.LinearSegmentedColormap.from_list('Zissou1', ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00"], N=100)



label_color_dict = {label: cmap(np.linspace(0,1,Nclasses))[idx] for idx, label in enumerate(np.unique(df_original['Class']))}
#colors = [cmap(label_color_dict[label]) for label in df_original['Class']]

colors = [label_color_dict[label] for label in df_original['Class']]
#colors = df_original['Class'].astype(str).map(colors)


plt.scatter(df_original[0], df_original[1], alpha=0.1, c=colors)
custom_lines = [Line2D([0], [0], color=cmap(np.linspace(0,1,Nclasses))[i], lw=4) for i, cl in enumerate(cmap(np.linspace(0,1,Nclasses)))]
#custom_lines = [Line2D([0], [0], color=label_color_dict[cl], lw=4) for cl in label_color_dict.keys()]
plt.legend(custom_lines, label_color_dict.keys(), loc='upper left', bbox_to_anchor=(1.04, 1))

# Add the axis labels
plt.xlabel('PC 1 (%.2f%%)' % (pca.explained_variance_ratio_[0]*100))
plt.ylabel('PC 2 (%.2f%%)' % (pca.explained_variance_ratio_[1]*100))
plt.title('PCA-Plot Validation')
plt.savefig(out_dir + '/pca_all_validation.pdf', format='pdf', bbox_inches="tight")
# Close the plot
plt.close()












#-----------------------------------------------------------------------------------------------------------
# Inputs
myseed = config_pca['seed']
torch.manual_seed(myseed)
torch.cuda.manual_seed(myseed)
np.random.seed(myseed)

# Parameters
data_file = snakemake.params.zip
model_file = snakemake.params.model
#data_file = './ERKH/ERKH.zip'
#model_file = './model/ERKKTR_model.pytorch'


start_time = None
end_time = None
measurement = None if config_training['measurement']=='' else config_training['measurement']
selected_classes = None
perc_selected_ids = 1  # Select only percentile of all trajectories, not always useful to project them all and slow
batch_sz = config_pca['batch']  # set as high as GPU memory can handle for speed up

rand_crop = True
set_to_project = 'train'  # one of ['all', 'train', 'validation', 'test']

#n_pca = config_pca['n_pca']
n_pca = 2
#---------------------------------------------------------------------------------------------------
# Model Loading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load(model_file) if torch.cuda.is_available() else torch.load(model_file, map_location='cpu')
net.eval()
net.double()
net = net.to(device)
length = net.length
print('LENGTH IS ', length)

# ----------------------------------------------------------------------------------------------------------------------

# Data Loading, Subsetting, Preprocessing
data = DataProcesser(data_file, datatable=False)
measurement = data.detect_groups_times()['groups'] if measurement is None else measurement
start_time = data.detect_groups_times()['times'][0] if start_time is None else start_time
end_time = data.detect_groups_times()['times'][1] if end_time is None else end_time
data.subset(sel_groups=measurement, start_time=start_time, end_time=end_time)

data.get_stats()
data.split_sets()
classes = tuple(data.classes.iloc[:,1])
dict_classes = data.classes[data.col_classname]

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

out_dir = './results_' + name + '/pca'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


df_raw = pd.DataFrame(pca_original)
df_original = df_raw.join(df_out)
df_original.to_csv(out_dir + '/pca_all_train.csv', index=False)
df_original = pd.DataFrame(df_original)

Nclasses = len(np.unique(df_original['Class']))
#cmap = plt.cm.get_cmap('hsv')
cmap = clr.LinearSegmentedColormap.from_list('Zissou1', ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00"], N=100)



label_color_dict = {label: cmap(np.linspace(0,1,Nclasses))[idx] for idx, label in enumerate(np.unique(df_original['Class']))}
#colors = [cmap(label_color_dict[label]) for label in df_original['Class']]

colors = [label_color_dict[label] for label in df_original['Class']]
#colors = df_original['Class'].astype(str).map(colors)


plt.scatter(df_original[0], df_original[1], alpha=0.1, c=colors)
custom_lines = [Line2D([0], [0], color=cmap(np.linspace(0,1,Nclasses))[i], lw=4) for i, cl in enumerate(cmap(np.linspace(0,1,Nclasses)))]
#custom_lines = [Line2D([0], [0], color=label_color_dict[cl], lw=4) for cl in label_color_dict.keys()]
plt.legend(custom_lines, label_color_dict.keys(), loc='upper left', bbox_to_anchor=(1.04, 1))

# Add the axis labels
plt.xlabel('PC 1 (%.2f%%)' % (pca.explained_variance_ratio_[0]*100))
plt.ylabel('PC 2 (%.2f%%)' % (pca.explained_variance_ratio_[1]*100))
plt.title('PCA-Plot Training')
plt.savefig(out_dir + '/pca_all_train.pdf', format='pdf', bbox_inches="tight")
# Close the plot
plt.close()




