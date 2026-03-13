#!/bin/bash
module load python
mamba activate ei_world_model
python main.py \
    load=d2rp721m:model \
    cluster=fas_eval \
    name=eval_libero10_pose \
    dataset=libero \
    wandb.project=wan_at2v_eval \
    dataset.metadata_path=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/SAILOR/env_repos/LIBERO/libero/datasets/metadata/metadata_20260126_232055.csv \
    algorithm=wan_at2v \
    algorithm.diffusion_forcing.cond_mode=concat \
    experiment.tasks=['validation'] \
    experiment.validation.data.num_workers=8 \
    experiment.validation.batch_size=2 \
    experiment.validation.limit_batch=null \