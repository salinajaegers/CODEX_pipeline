
import torch
import sys
from torch.utils.data import DataLoader
from torchvision import transforms
import seaborn as sns
import matplotlib as plt
import pandas as pd
import os
import numpy as np
import plotly.express as px
import pytorch_lightning as pl


import yaml

# Custom functions/classes
path_to_module = snakemake.params.scripts  # Path where all the .py files are, relative to the notebook folder
sys.path.append(path_to_module)
import results_model
from class_dataset import myDataset, RandomCrop, Subtract, ToTensor
from utils import model_output
from load_data import DataProcesser

with open('./config.yml', 'r') as file:
    config_file = yaml.safe_load(file)


config_training = config_file['training']
config_prototypes = config_file['prototypes']
name = str(config_file['name'])


# For reproducibility
myseed = config_prototypes['seed']
torch.manual_seed(myseed)
torch.cuda.manual_seed(myseed)
np.random.seed(myseed)

cuda_available = torch.cuda.is_available()

# Set params
prototypes_set = 'all'
assert prototypes_set in ['train', 'validation', 'test', 'all']

data_file = snakemake.params.zip
model_file = snakemake.params.model
#data_file = './ERKH/ERKH.zip'
#model_file = './model/ERKKTR_model.pytorch'

batch_size = config_prototypes['batch'] # Set as high as your memory allows to speed up
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
meas_var = None  # Set to None for auto detection
start_time = None  # Set to None for auto detection
end_time = None  # Set to None for auto detection

model = torch.load(model_file) if cuda_available else torch.load(model_file, map_location='cpu')
model.eval()
model.double()
model.batch_size = 1  # Set it to 1!
model = model.to(device)

# Select measurements and times, subset classes and split the dataset
data = DataProcesser(data_file)
meas_var = data.detect_groups_times()['groups'] if meas_var is None else meas_var
start_time = data.detect_groups_times()['times'][0] if start_time is None else start_time
end_time = data.detect_groups_times()['times'][1] if end_time is None else end_time
data.subset(sel_groups=meas_var, start_time=start_time, end_time=end_time)
data.get_stats()
data.split_sets()
classes = tuple(data.classes.iloc[:,1])
dict_classes = data.classes[data.col_classname]

# Input preprocessing, this is done sequentially, on the fly when the input is passed to the network
subtract_numbers = [data.stats['mu'][meas]['train'] for meas in meas_var]
ls_transforms = transforms.Compose([
    Subtract(subtract_numbers),
    RandomCrop(output_size=model.length, ignore_na_tails=True),
    ToTensor()])

# Set the DataLoader with the selected set
if prototypes_set == 'train':
    data_toLoader = myDataset(dataset=data.train_set, transform=ls_transforms)
elif prototypes_set == 'validation':
    data_toLoader = myDataset(dataset=data.validation_set, transform=ls_transforms)
elif prototypes_set == 'test':
    data_toLoader = myDataset(dataset=data.test_set, transform=ls_transforms)
elif prototypes_set == 'all':
    data_toLoader = myDataset(dataset=data.dataset, transform=ls_transforms)

if batch_size > len(data_toLoader):
    raise ValueError('Batch size ({}) must be smaller than the number of trajectories in the selected set ({}).'.format(batch_size, len(data_toLoader)))
                     
myDataLoader = DataLoader(dataset=data_toLoader,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          drop_last=False)


# Plot some trajectories to check that the data loading and processing is properly done.
n_smpl = 6
indx_smpl = np.random.randint(0, len(data_toLoader), n_smpl)
length = model.length

col_ids = []
col_lab = []
col_mes = []
# Long format for seaborn grid, for loop to avoid multiple indexing
# This would triggers preprocessing multiple times and add randomness with some preprocessing steps
for i in indx_smpl:
    smpl = data_toLoader[i]
    col_ids.append(smpl['identifier'])
    col_lab.append(smpl['label'].item())
    col_mes.append(smpl['series'].numpy().transpose())
col_ids = pd.Series(np.hstack(np.repeat(col_ids, length)))
col_lab = pd.Series(np.hstack(np.repeat(col_lab, length)))
col_mes = pd.DataFrame(np.vstack(col_mes), columns=meas_var)
col_tim = pd.Series(np.tile(np.arange(0, length), n_smpl))

df_smpl = pd.concat([col_ids, col_lab, col_tim, col_mes], axis=1)
df_smpl.rename(columns={0: 'identifier', 1: 'label', 2:'time'}, inplace=True)
df_smpl = df_smpl.melt(id_vars=['identifier', 'label', 'time'], value_vars=meas_var)

sns.set_style('white')
sns.set_context('notebook')
grid = sns.FacetGrid(data=df_smpl, col='identifier', col_wrap=3, sharex=True)
grid.map_dataframe(sns.lineplot, x='time', y='value', hue='variable', ci=None)
grid.set(xlabel='Time', ylabel='Measurement Value')
grid.add_legend()


# ## Confusion matrix
accuracy = results_model.acc_per_class(model, myDataLoader, device, dict_classes)
conft = results_model.confusion_matrix(model, myDataLoader, device, dict_classes)
conft['Precision'] = accuracy
print(conft)

# Path where to export plots and table with prototypes
#out_dir = '/data/users/sjaegers/research_project/prototypes/AKT_50_m3/' + '_'.join(meas_var) + '/prototypes'
out_dir = './results_' + name + '/prototypes'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
out_file = out_dir + '/confusionTable_' + str(config_prototypes['n_prototypes']) + '.csv'
conft.to_csv(out_file)


# ## Prototypes extraction
# Number of prototypes from each category
ntop = config_prototypes['n_prototypes']
nworst = config_prototypes['n_prototypes']
nuncor = config_prototypes['n_prototypes']
nrandom = config_prototypes['n_prototypes']

# Minimal confidence for uncorrelated prototypes
threshold_confidence = config_prototypes['threshold_confidence']
    
# Plot and export options
plot_results = True
export_plot_results = True
export_table_prototypes = True

# Helper functions for plotting
col_id = data.col_id
col_class = data.col_class
col_classname = data.col_classname

def make_subset_forPlot(sel_ids, sel_id_vars = [col_id, col_class]):
    subset = data.dataset.loc[data.dataset[col_id].isin(sel_ids)]
    subset = subset.melt(id_vars=sel_id_vars)
    subset[['Meas','Time']]=subset['variable'].str.extract(r'^(?P<Meas>[A-Za-z0-9]+)_(?P<Time>\d+)$')
    subset['Time'] = subset['Time'].astype('int')
    subset.sort_values('Time', inplace = True)
    subset = pd.merge(subset, data.classes, left_on=col_class, right_on=col_class)
    return(subset)

#["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00"]
def make_grid_plot(in_data, ncol_plot, sharex=True, sharey=True, ylim=(None, None), yscale='linear', out_file=None):
    sns.set_style('ticks')
    sns.set_context('paper')
    grid = sns.FacetGrid(data=in_data, col=col_id, col_wrap=ntop, sharex=sharex, sharey=sharey, aspect=2)
    grid.map_dataframe(sns.lineplot, x="Time", y="value", hue="Meas", ci=None)
    grid.set(xlabel='Time', ylabel='Measurement', ylim=ylim, yscale = yscale)
    grid.add_legend()
    if out_file:
        grid.savefig(out_file)


# ### Top trajectories per class
tops = results_model.top_confidence_perclass(model, myDataLoader, n=ntop, labels_classes=dict_classes)

if plot_results:
    subset = make_subset_forPlot(tops[col_id])
    subset[col_id] = subset[col_id] + ': ' + subset[col_classname]
    subset.sort_values(by=[col_classname], inplace=True)
    out_file = out_dir + '/tops_' + str(config_prototypes['n_prototypes']) + '.pdf' if export_plot_results else None
    make_grid_plot(subset, ntop, sharey=True, yscale='linear', out_file=out_file)


# ### Least correlated set per class
# 
# When choosing uncorrelated curves, pick only from curves for which the model confidence in the input class is at least "threshold_confidence".
uncorr = results_model.least_correlated_set(model, myDataLoader, n=nuncor, labels_classes=dict_classes, threshold_confidence=threshold_confidence)

if plot_results:
    subset = make_subset_forPlot(uncorr[col_id])
    out_file = out_dir + '/uncorr_' + str(config_prototypes['n_prototypes']) + '.pdf'  if export_plot_results else None
    make_grid_plot(subset, nuncor, sharey=True, yscale='linear', out_file=out_file)


# ### Worst trajectory per class
worsts = results_model.worst_classification_perclass(model, myDataLoader, n=nworst, labels_classes=dict_classes)

if plot_results:
    subset = data.dataset.loc[data.dataset['ID'].isin(worsts['ID'])]
    subset = pd.merge(subset, worsts[['ID', 'Prediction']], on='ID')
    subset = subset.melt(id_vars=['ID', 'class', 'Prediction'])
    subset[['Meas','Time']]=subset['variable'].str.extract(r'^(?P<Meas>\w+)_(?P<Time>\d+)$')
    subset['Time'] = subset['Time'].astype('int')
    subset = pd.merge(subset, data.classes, left_on=col_class, right_on=col_class)
    subset['ID'] = subset['ID'] + '->' + subset['Prediction']

    out_file = out_dir + '/worsts_' + str(config_prototypes['n_prototypes']) + '.pdf' if export_plot_results else None
    make_grid_plot(subset, nworst, sharey=True, yscale='linear', out_file=out_file)

# ### Random sample
# 
# Get a random sample of trajectories.

randoms_ids = []
for classe in data.validation_set['class'].unique():
    randoms_ids += list(data.dataset.loc[data.dataset['class']==classe]['ID'].sample(nrandom))
randoms = model_output(model, myDataLoader)
randoms = randoms.loc[randoms['ID'].isin(randoms_ids)]

if plot_results:
    subset = make_subset_forPlot(randoms[col_id])
    out_file = out_dir + '/random_' + str(config_prototypes['n_prototypes']) + '.pdf' if export_plot_results else None
    make_grid_plot(subset, nrandom, sharey=True, yscale='linear', out_file=out_file)


# ### Table with all prototypes
tops['Category'] = 'Top'
worsts['Category'] = 'Worst'
uncorr['Category'] = 'Uncorr_' + str(threshold_confidence * 100)
randoms['Category'] = 'Random'
proto_table = pd.concat([tops, worsts, uncorr, randoms])

if export_table_prototypes:
    out_file = out_dir + '/protoTable_' + str(config_prototypes['n_prototypes']) + '.csv'
    proto_table.to_csv(out_file, index=False)

