#!/bin/bash
module load python
mamba activate ei_world_model
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py \
    cluster=fas_h100 \
    resume=yeand8xz \
    name=dexmimicgen_pose_ray-none \
    algorithm=wan_at2v_cam \
    algorithm.diffusion_forcing.cond_mode=concat+ray \
    algorithm.diffusion_forcing.normalization=none \
    experiment.training.checkpointing.every_n_train_steps=50 \
    experiment.training.checkpointing.save_on_exception=true \
    experiment.training.batch_size=2 \
    experiment.training.optim.accumulate_grad_batches=8 \