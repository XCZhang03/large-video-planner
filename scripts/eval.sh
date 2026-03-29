#!/bin/bash
module load python
mamba activate ei_world_model
python main.py \
    load=jg9viims:model \
    cluster=fas_eval \
    name=eval_libero10_libero-object-ft \
    dataset=libero \
    wandb.project=wan_at2v_eval \
    dataset.metadata_path=/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR/env_repos/LIBERO/libero/datasets/metadata/metadata_20260227_011748.csv \
    algorithm=wan_at2v \
    algorithm.diffusion_forcing.cond_mode=concat \
    experiment.tasks=['validation'] \
    experiment.validation.data.num_workers=8 \
    experiment.validation.batch_size=2 \
    experiment.validation.limit_batch=null \
    dataset.n_frames=81 \
    dataset.total_frames=101 \
    algorithm.n_frames=41 \
    algorithm.hist_len=21 \
    +algorithm.max_frames=81 \
    +algorithm.hist_steps=[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60] \