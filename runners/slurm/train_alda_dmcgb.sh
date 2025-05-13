#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c4
#SBATCH --mem-per-cpu=16G
#SBATCH --output=tmp/alda_dmcgb_%j.log

export MUJOCO_GL=egl

SEED=0
DOMAIN="finger"
TASK="spin"
RUN_NAME="alda_${DOMAIN}_${TASK}"
srun python -m scripts.train --experiment_spec_file=specs/train_alda_${DOMAIN}_${TASK}.yaml  \
                              --results_dir=./results \
                              --use_wandb=True \
                              --wandb_group=alda_${DOMAIN}_${TASK} \
                              --wandb_run_name=$RUN_NAME \
                              --spec_overrides \
                              --spec.name=$RUN_NAME \
                              --spec.trainer.config.seed=$SEED