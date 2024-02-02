import pandas as pd
from random import choices, seed
import yaml
import os
from zipfile import ZipFile
from collections import OrderedDict


# get dataset
with open(snakemake.params.scripts + '/config.yml', 'r') as file:
#with open('./config.yml', 'r') as file:
    config_file = yaml.safe_load(file)

config_prep = config_file['prep']
name = config_file['name']

dataset = pd.read_csv(snakemake.input.dataset)
classes = pd.read_csv(snakemake.input.classes)
#dataset = pd.read_csv('./ERKH/data/dataset.csv')
#classes = pd.read_csv('./ERKH/data/classes.csv')

colnames = list(dataset.columns.values)
colnames.remove('ID')
colnames.remove('class')
groups = list(OrderedDict.fromkeys([i.split('_')[0] for i in colnames]))


dataset_final = dataset.iloc[:, 0:2].copy()
dataset_intp = dataset.copy()
for i in groups:
    # interpolate
    dataset_intp = dataset_intp.filter(regex=i)
    dataset_intp = dataset_intp.interpolate(axis=1, method='linear', limit_area='inside')
    print(dataset_intp)
    dataset_final = pd.concat([dataset_final, dataset_intp], axis=1)
    print(dataset_final)

# create ID_set
seed(config_prep['seed'])
id_set = pd.DataFrame(dataset['ID'])
training, validation, test = config_prep['training'], config_prep['validation'], config_prep['test']
id_set['set'] = choices(['train', 'validation', 'test'], k=len(id_set), weights=[training, validation, test])
# save new dir and file
id_set.to_csv('./id_set.csv', sep=',', na_rep='', header=True, index=False)
dataset_final.to_csv('./dataset.csv', sep=',', na_rep='', header=True, index=False)
