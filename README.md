# ALDA Official Repo 

![alda_walker_gif](assets/alda_walker.gif)

[[Paper]](https://arxiv.org/abs/2410.07441)
[[Project Website]](https://sumeetbatra.github.io/alda/)

Official Repo of ALDA! Zero-shot visual generalization of RL agents
on various DMControl tasks using disentangled representation learning with principles 
of associative memory. 


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

### Running Experiments
The `specs/` folder contains yaml configs for all tasks with the default hyperparameters used for the main paper results.
For example, to run ALDA on the Walker Walk task:

```python
python -m scripts.train --experiment_spec_file specs/train_alda_walker_walk.yaml
```

We use Weights and Biases to record results. To enable w&b, add the `--use_wandb` flag and specify your w&b entity and
project name with the `--wandb_entity` and `--wandb_project` flags:

```python
python -m scripts.train --experiment_spec_file specs/train_alda_walker_walk.yaml --wandb_entity xyz --wandb_project abc
```

By default, the code will not let you run two experiments with the same name so that you don't accidentally overwrite an 
existing result. To change this behavior, add the `--debug` flag:

```python
python -m scripts.train --experiment_spec_file specs/train_alda_walker_walk.yaml --debug
```

## Acknowledgements
- This SAC implementation is based off of [this repository (SVEA).](https://github.com/nicklashansen/dmcontrol-generalization-benchmark)

- ALDA builds directly on top of [QLAE](https://github.com/kylehkhsu/latent_quantization) for disentanglement. 