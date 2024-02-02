
# Standard libraries
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
from skimage.filters import threshold_li, threshold_mean
import os
from itertools import chain
from tqdm import tqdm
import sys
import yaml

# Custom functions/classes
print('yes')
path_to_module = snakemake.params.scripts  # Path where all the .py files are, relative to the notebook folder
sys.path.append(path_to_module)
from load_data import DataProcesser
from results_model import top_confidence_perclass, least_correlated_set
from pattern_utils import extend_segments, create_cam, longest_segments, extract_pattern
from class_dataset import myDataset, ToTensor, RandomCrop

print('yes')
#with open(snakemake.params.scripts + '/config.yml', 'r') as file:
with open('./config.yml', 'r') as file:
    config_file = yaml.safe_load(file)

config_motif = config_file['motif analysis']
name = str(config_file['name'])

# For reproducibility
myseed = config_motif['seed']
torch.manual_seed(myseed)
torch.cuda.manual_seed(myseed)
np.random.seed(myseed)

cuda_available = torch.cuda.is_available()


# ## Parameters
# 
# Parameters for the motifs extraction:
# - selected_set: str one of ['all', 'training', 'validation', 'test'], from which set of trajectories should motifs be extracted? For this purprose, extracting from training data also makes sense.
# - n_series_perclass: int, maximum number of series, per class, on which motif extraction is attempted.
# - n_pattern_perseries: int, maximum number of motifs to extract out of a single trajectory.
# - mode_series_selection: str one of ['top_confidence', 'least_correlated']. Mode to select the trajectories from which to extract the motifs (see Prototype analysis). If top confidence, the motifs might be heavily biased towards a representative subpopulation of the class. Hence, the output might not reflect the whole diversity of motifs induced by the class.
# - extend_patt: int, by how many points to extend motifs? After binarization into 'relevant' and 'non-relevant time points', the motifs are usually fragmented because a few points in their middle are improperly classified as 'non-relevant'. This parameter allows to extend each fragment by a number of time points (in both time directions) before extracting the actual patterns.
# - min_len_patt/max_len_patt: int, set minimum/maximum size of a motif. **/!\ The size is given in number of time-points. This means that if the input has more than one channel, the actual length of the motifs will be divided across them.** For example, a motif that spans over 2 channels for 10 time points will be considered of length 20.
# 
# Parameters for the groups of motifs:
# - export_perClass: bool, whether to run the motif clustering class per class.
# - export_allPooled: bool, whether to pool all motifs across classes for clustering.
print('yes')
selected_set = 'all'
n_series_perclass = config_motif['n_series_perclass']
n_pattern_perseries = config_motif['n_pattern_perseries']
mode_series_selection = config_motif['mode_series_selection']
# mode_series_selection = 'least_correlated'
thresh_confidence = config_motif['thresh_confidence']  # used in least_correlated mode to choose set of series with minimal classification confidence
extend_patt = config_motif['extend_patt']
min_len_patt = config_motif['min_len_patt']
max_len_patt = config_motif['max_len_patt'] # length to divide by nchannel

export_perClass = True
export_allPooled = True

assert selected_set in ['all', 'training', 'validation', 'test']
assert mode_series_selection in ['top_confidence', 'least_correlated']


# ## Load model and data
# 
# - Pay attention to the order of 'meas_var', should be the same as for training the model!
# - Pay attention to trajectories preprocessing.
# - Set batch_size as high as memory allows for speed up.
print('yes')
data_file = snakemake.params.zip
model_file = snakemake.params.model
out_dir = './results_' + name + '/motifs'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
print('yes')
meas_var = None  # Set to None for auto detection
start_time = None  # Set to None for auto detection
end_time = None  # Set to None for auto detection

batch_size = config_motif['batch']  # Set as high as memory allows for speed up
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
model = torch.load(model_file) if cuda_available else torch.load(model_file, map_location='cpu')
model.eval()
model.double()
model.batch_size = batch_size
model = model.to(device)


# Pay attention that **data.process() is already centering the data**, so don't do a second time when loading the data in the DataLoader. The **random crop** should be performed before passing the trajectories to the model to ensure that the same crop is used as input and for extracting the patterns.

# Transformations to perform when loading data into the model
ls_transforms = transforms.Compose([RandomCrop(output_size=model.length, ignore_na_tails=True),
                                                            ToTensor()])
# Loading and PREPROCESSING
data = DataProcesser(data_file)
meas_var = data.detect_groups_times()['groups'] if meas_var is None else meas_var
start_time = data.detect_groups_times()['times'][0] if start_time is None else start_time
end_time = data.detect_groups_times()['times'][1] if end_time is None else end_time
# Path where to export tables with motifs
if out_dir == 'auto':
    out_dir = 'output/' + '_'.join(meas_var) + '/local_motifs/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

data.subset(sel_groups=meas_var, start_time=start_time, end_time=end_time)
cols_to_check=data.dataset.columns.values[data.dataset.columns.str.startswith('FGF')]
cols_dict={k:'float64' for k in cols_to_check}
data.dataset=data.dataset.astype(cols_dict)
data.get_stats()
data.process(method='center_train', independent_groups=True)  # do here and not in loader so can use in df
data.crop_random(model.length, ignore_na_tails=True)
data.split_sets(which='dataset')
classes = tuple(data.classes[data.col_classname])
dict_classes = data.classes[data.col_classname]

# Random crop before to keep the same in df as the ones passed in the model
if selected_set == 'validation':
    selected_data = myDataset(dataset=data.validation_set, transform=ls_transforms)
    df = data.validation_set
elif selected_set == 'training':
    selected_data = myDataset(dataset=data.train_set, transform=ls_transforms)
    df = data.train_set
elif selected_set == 'test':
    selected_data = myDataset(dataset=data.test_set, transform=ls_transforms)
    df = data.train_set
elif selected_set == 'all':
    try:
        selected_data = myDataset(dataset=data.dataset_cropped, transform=ls_transforms)
        df = data.dataset_cropped
    except:
        selected_data = myDataset(dataset=data.dataset, transform=ls_transforms)
        df = data.dataset

if batch_size > len(selected_data):
    raise ValueError('Batch size ({}) must be smaller than the number of trajectories in the selected set ({}).'.format(batch_size, len(selected_data)))        

data_loader = DataLoader(dataset=selected_data,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=4)
# Dataframe used for retrieving trajectories. wide_to_long() instead of melt() because can do melting per group of columns
df = pd.wide_to_long(df, stubnames=meas_var, i=[data.col_id, data.col_class], j='Time', sep='_', suffix='\d+')
df = df.reset_index()  # wide_to_long creates a multi-level Index, reset index to retrieve indexes in columns
df.rename(columns={data.col_id: 'ID', data.col_class: 'Class'}, inplace=True)
df['ID'] = df['ID'].astype('U32')
del data  # free memory


# ## Select trajectories from which to extract patterns

if mode_series_selection == 'least_correlated':
    set_trajectories = least_correlated_set(model, data_loader, threshold_confidence=thresh_confidence, device=device,
                                            n=n_series_perclass, labels_classes=dict_classes)
elif mode_series_selection == 'top_confidence':
    set_trajectories = top_confidence_perclass(model, data_loader, device=device, n=n_series_perclass,
                                               labels_classes=dict_classes)

# free some memory by keeping only relevant series
selected_trajectories = set_trajectories['ID']
df = df[df['ID'].isin(selected_trajectories)]
# Make sure that class is an integer (especially when 0 or 1, could be read as boolean)
df['Class'] = df['Class'].astype('int32')


# ## Extract patterns
# 
# ### Extract, extend and filter patterns. 
# 
# Outputs a report of how many trajectories were filtered out by size.
# Initialize dict to store the patterns and set progress bar
store_patts = {i:[] for i in classes}
model.batch_size = 1  # Leave it to 1!
report_filter = {'Total number of patterns': 0,
                 'Number of patterns above maximum length': 0,
                 'Number of patterns below minimum length': 0}
pbar = tqdm(total=len(selected_trajectories))

for id_trajectory in selected_trajectories:
    # Read and format the trajectories to numpy
    series_numpy = np.array(df.loc[df['ID'] == id_trajectory][meas_var]).astype('float').squeeze()
    # Row: measurement; Col: time
    if len(meas_var) >= 2:
        series_numpy = series_numpy.transpose()
    series_tensor = torch.tensor(series_numpy)
    class_trajectory = df.loc[df['ID']==id_trajectory]['Class'].iloc[0]  # repeated value through all series
    class_label = classes[class_trajectory]
    
    # Create and process the CAM for the trajectory
    cam = create_cam(model, array_series=series_tensor, feature_layer='features',
                         device=device, clip=0, target_class=class_trajectory)
    thresh = threshold_li(cam)
    bincam = np.where(cam >= thresh, 1, 0)
    bincam_ext = extend_segments(array=bincam, max_ext=extend_patt)
    patterns = longest_segments(array=bincam_ext, k=n_pattern_perseries)
    
    # Filter short/long patterns
    report_filter['Total number of patterns'] += len(patterns)
    report_filter['Number of patterns above maximum length'] += len([k for k in patterns.keys() if patterns[k] > max_len_patt])
    report_filter['Number of patterns below minimum length'] += len([k for k in patterns.keys() if patterns[k] < min_len_patt])
    patterns = {k: patterns[k] for k in patterns.keys() if (patterns[k] >= min_len_patt and
                                                            patterns[k] <= max_len_patt)}
    if len(patterns) > 0:
        for pattern_position in list(patterns.keys()):
            store_patts[class_label].append(extract_pattern(series_numpy, pattern_position, NA_fill=False))
    pbar.update(1)

print(report_filter)


# ### Dump patterns into csv
if export_allPooled:
    concat_patts_allPooled = np.full((sum(map(len, store_patts.values())), len(meas_var) * max_len_patt), np.nan)
    irow = 0
for classe in classes:
    concat_patts = np.full((len(store_patts[classe]), len(meas_var) * max_len_patt), np.nan)
    for i, patt in enumerate(store_patts[classe]):
        if len(meas_var) == 1:
            len_patt = len(patt)
            concat_patts[i, 0:len_patt] = patt
        if len(meas_var) >= 2:
            len_patt = patt.shape[1]
            for j in range(len(meas_var)):
                offset = j*max_len_patt
                concat_patts[i, (0+offset):(len_patt+offset)] = patt[j, :]
    if len(meas_var) == 1:
        headers = ','.join([meas_var[0] + '_' + str(k) for k in range(max_len_patt)])
        fout_patt = out_dir + 'motif_{}.csv.gz'.format(classe)
        if export_perClass:
            np.savetxt(fout_patt, concat_patts,
                       delimiter=',', header=headers, comments='')
    elif len(meas_var) >= 2:
        headers = ','.join([meas + '_' + str(k) for meas in meas_var for k in range(max_len_patt)])
        fout_patt = out_dir + 'motif_{}.csv.gz'.format(classe)
        if export_perClass:
            np.savetxt(fout_patt, concat_patts,
                       delimiter=',', header=headers, comments='')
    if export_allPooled:
        concat_patts_allPooled[irow:(irow+concat_patts.shape[0]), :] = concat_patts
        irow += concat_patts.shape[0]

if export_allPooled:
    concat_patts_allPooled = pd.DataFrame(concat_patts_allPooled)
    concat_patts_allPooled.columns = headers.split(',')
    pattID_col = [[classe] * len(store_patts[classe]) for classe in classes]
    concat_patts_allPooled['pattID'] = [j+'_'+str(i) for i,j in enumerate(list(chain.from_iterable(pattID_col)))]
    concat_patts_allPooled.set_index('pattID', inplace = True)
    fout_patt = out_dir + 'motif_allPooled.csv.gz'.format(classe)
    concat_patts_allPooled.to_csv(fout_patt, header=True, index=True, compression='gzip')

