CUDA_VIEIBLE_DEVICES=0
python main.py \
    load=/workspace/fan/xc/icml_wm/large-video-planner/outputs/checkpoint_links/awesome-wm/icml/ehwb3z4b/latest.ckpt:model \
    dataset.metadata_path=/workspace/fan/xc/icml_wm/large-video-planner/data/meta_data/libero_icml/libero_task4_eval_hard.csv \
    experiment.tasks=['validation'] \
    name=eval_libero_task4_hard \