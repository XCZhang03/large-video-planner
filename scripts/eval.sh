#!/bin/bash
module load python
mamba activate ei_world_model
python main.py \
    resume=null \
    load=xbk1muqp:model \
    cluster=fas_eval \
    name=eval_long_dexmimicgen_pose \
    dataset.metadata_path=/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/DiffRL/dexmimicgen_dataset/dexmimicgen/datasets/replay_videos/metadata_20260130_065553.csv \
    +dataset.total_frames=52 \
    algorithm=wan_at2v \
    algorithm.diffusion_forcing.cond_mode=concat \
    experiment.tasks=['validation'] \
    experiment.validation.data.num_workers=8 \
    experiment.validation.batch_size=2 \
    experiment.validation.limit_batch=null \