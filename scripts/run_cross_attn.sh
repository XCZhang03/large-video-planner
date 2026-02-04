#!/bin/bash
module load python
mamba activate ei_world_model
python main.py \
    cluster=fas_h100 \
    load=uelxxm9w:model \
    name=libero_90_cross-attn \
    algorithm=wan_at2v_low_dim \
    experiment.training.checkpointing.every_n_train_steps=100 \
    experiment.training.checkpointing.save_on_exception=true \
    algorithm.diffusion_forcing.cond_mode=cross-attn \