#!/bin/bash
module load python
mamba activate ei_world_model
python main.py \
    cluster=gpu_h200 \
    load=y6z4m3pp:model \
    +requeue=0 \
    name=libero_pose_sparse_mokapot \
    algorithm=wan_at2v \
    dataset=libero \
    experiment.training.checkpointing.every_n_train_steps=100 \
    experiment.training.checkpointing.save_on_exception=true \
    algorithm.diffusion_forcing.cond_mode=concat \
    experiment.training.batch_size=4 \
    experiment.training.optim.accumulate_grad_batches=2 \
    experiment.num_nodes=2 \
    dataset.n_frames=81 \
    dataset.total_frames=101 \
    dataset.metadata_path=/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR/env_repos/LIBERO/libero/datasets/metadata/metadata_20260308_221042.csv \
    algorithm.n_frames=41 \
    algorithm.hist_len=21 \
    +algorithm.max_frames=81 \
    +algorithm.hist_steps=[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60] \