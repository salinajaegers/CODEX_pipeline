# Pipelining CODEX

This is a pipeline made for CODEX (https://github.com/pertzlab/CODEX) using snakemake. 

## Installing CODEX
First, the environment for CODEX needs to be configured with singularity. The definition file 
'sing_CODEX_CPU_pipeline_motif.def' will be used to create the singularity image in which CODEX will be run. 
To do so install singularity and run 'singularity build sing_CODEX_CPU_pipeline_motif.sif /path/to/definition_file.def'. 
Alternatively, a GPU-compatible environment can be created in Conda using the instructions from the original CODEX 
documentation, and Snakemake needs to be added to this. 

The pipeline can then be started from the main notebook 'CODEX pipeline.ipynb' within the singularity using the instructions provided there. 
