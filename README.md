# Pipelining CODEX

This is a pipeline made for CODEX (https://github.com/pertzlab/CODEX) using snakemake. The Pipeline is using the CNN-training, the prototype selection, and the motif mining from CODEX. A PCA has been added as well as the option to have an interactive exploration of the PCA results in a notebook separate from the pipeline. The motif analysis part of CODEX which is done in R has not yet been integrated into the pipeline and needs to be set up locally. 

## Installing CODEX
Tthe environment for CODEX is created using a singularity image. To do so install singularity and run 'singularity build sing_CODEX_CPU_pipeline_motif.sif sing_CODEX_CPU_pipeline_motif.def' to creat the image. Currently the image is only woking with CPU. Alternatively, a GPU-compatible environment can be created in Conda using the instructions from the original CODEX documentation. In this case, Snakemake and pyyaml would need to be added to the environment manually. 

The pipeline can then be started from the main notebook 'CODEX pipeline.ipynb' within the singularity. In the notebook the parameters for the analysis need to be set, more details are provided within the notebook. 

Alternatively, the pipeline can be run from commandline with 'snakemake -s ./source/Snakefile.txt -d path/to/working/directory --configfile ./source/config.yaml'. The working directory is where the data is stored in. Snakemake will then create a subdirectory within this where the results will be stored. The name of the analysis and the general settings need to be adjusted in the configuration file in the source directory. It is only possible to run one analysis at a time per working directory, as snakemake does not allow for more. 

## Data preparation
The input to the pipeline is similar to what is required from CODEX. The class definition file and the dataset file need to be the same format as originally, except that interpolation is not necessary in the dataset as it is done within the pipeline. The set with the allocatement of training, validation, and test to the IDs is not required as input, it will be created with the percentages definined in the pipeline configurations. The data does not need to be zipped either as an archive will be created after creating the id allocation set. It does however need to be in a separate directory, which Snakemake can be run in. 

## Interactive PCA
All scripts that are needed for the interactive PCA can be found in the PCA folder. It is best to run this part locally, instead of on a cluster. To make it more accessible for local usage, the PCA is calculated as part of the pipeline and the results can then be referenced in the notebook. This way only the plots themselves have to be created locally. The plot can then be explored in more detail by clicking on the points within the plot itself. A tab should open in the browser where the class probabilities and the time series are shown. 



