#!/bin/bash
module load python
mamba activate ei_world_model
python main.py \
    cluster=fas_h200 \
    +requeue=ikswdu56 \
    name=libero_pose \
    algorithm=wan_at2v \
    dataset=libero \
    experiment.training.checkpointing.every_n_train_steps=100 \
    experiment.training.checkpointing.save_on_exception=true \
    algorithm.diffusion_forcing.cond_mode=concat \