# ALDA Official Repo 


## Setup Instructions

### Setting up DMControl Generalization Benchmark

In `dmcontrol_generalization_benchmark/setup/config.cfg` change `your/path/to/` to wherever you put this repository. 
Follow the instructions [here](https://github.com/nicklashansen/dmcontrol-generalization-benchmark/tree/main) under the 
`Datasets` header to download the DAVIS dataset for the distracting background environment.
Download and extract the dataset to `alda_official/dmcontrol_generalization_benchmark/datasets/`.
Finally, follow their setup instructions in `setup/install_envs.sh`. 
You do not need to create another conda env, just use the same one we set up earlier! 