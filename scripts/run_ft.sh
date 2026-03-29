#!/bin/bash
module load python
mamba activate ei_world_model
python main.py \
    cluster=fas_ft \
    load=9l71tu0f:model \
    name=libero_pose_sparse_object \
    algorithm=wan_at2v \
    dataset=libero \
    experiment.training.checkpointing.every_n_train_steps=50 \
    experiment.training.checkpointing.save_on_exception=false \
    experiment.training.max_epochs=10 \
    experiment.validation.val_every_n_epoch=1 \
    experiment.validation.val_every_n_step=null \
    algorithm.diffusion_forcing.cond_mode=concat \
    experiment.training.batch_size=4 \
    experiment.training.optim.accumulate_grad_batches=4 \
    experiment.num_nodes=1 \
    dataset.n_frames=81 \
    dataset.total_frames=81 \
    dataset.metadata_path=/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR/env_repos/LIBERO/libero/datasets/libero_object_replay/metadata.csv \
    algorithm.n_frames=41 \
    algorithm.hist_len=21 \
    +algorithm.max_frames=81 \
    +algorithm.hist_steps=[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60] \