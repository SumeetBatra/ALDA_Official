# ALDA Official Repo 


## Installation

### Setting up Conda Environment 
1. Create the conda environment and activate it 
```bash
conda create -n alda python=3.8
conda activate alda
```
2. run or manually install the packages in `dmcontrol_generalization_benchmark/setup/install_envs.sh`.
```bash
cd dmcontrol_generalization_benchmark
./setup/install_envs.sh
cd ..
```
3. Install the packages in `requirements.txt`:
```bash
pip install -r requirements.txt 
```


### Setting up DMControl Generalization Benchmark

In `dmcontrol_generalization_benchmark/setup/config.cfg` change `your/path/to/` to wherever you put this repository. 
Follow the instructions [here](https://github.com/nicklashansen/dmcontrol-generalization-benchmark/tree/main) under the 
`Datasets` header to download the DAVIS dataset for the distracting background environment.
Download and extract the dataset to `alda_official/dmcontrol_generalization_benchmark/datasets/`.