#!/bin/bash
module load python
mamba activate ei_world_model
python main.py \
    resume=9l71tu0f \
    name=test \
    algorithm=wan_at2v \
    dataset=libero \
    experiment.training.checkpointing.every_n_train_steps=100 \
    experiment.training.checkpointing.save_on_exception=false \
    experiment.validation.batch_size=1 \
    experiment.training.batch_size=1 \
    algorithm.diffusion_forcing.cond_mode=concat \
    dataset.n_frames=81 \
    dataset.total_frames=101 \
    dataset.metadata_path=/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/SAILOR/env_repos/LIBERO/libero/datasets/metadata/metadata_20260210_085632.csv \
    +algorithm.max_frames=81 \
    algorithm.n_frames=41 \
    algorithm.hist_len=21 \
    +algorithm.hist_steps=[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60] \
    experiment.tasks=['validation'] \