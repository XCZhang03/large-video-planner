#!/bin/bash
module load python
mamba activate ei_world_model
python main.py \
    load=null \
    name=test \
    dataset=dexmimicgen \
    algorithm=wan_at2v_cam \
    algorithm.diffusion_forcing.cond_mode=concat+ray \
    algorithm.diffusion_forcing.normalization=none \
    experiment.tasks=['training'] \
    experiment.training.data.num_workers=2 \
    experiment.validation.data.num_workers=2 \
    experiment.validation.batch_size=1 \
    experiment.validation.limit_batch=1 \
    dataset.width=64 \
    dataset.height=64 \