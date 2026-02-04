"""Minimal test script for wm_client package."""

import imageio
import imageio.v3 as iio

from wm_client import WMClient, WMPredictionOutput

# Configure your server address
HOST = "10.31.146.126"
PORT = 7860


def main() -> None:
    # Example: load videos and construct frame arrays (replace with your own images)
    video_path = "/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/SAILOR/env_repos/LIBERO/libero/datasets/libero_90_std_0.2_224_224_chunk80_pos_len80/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo/demo_9/merged_seg1.mp4"
    pose_video_path = "/n/holylabs/ydu_lab/Lab/zhangxiangcheng/code/SAILOR/env_repos/LIBERO/libero/datasets/libero_90_std_0.2_224_224_chunk80_pos_len80/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo/demo_9/merged_seg1_pose.mp4"

    video_frames = iio.imread(video_path)  # (T, H, W, C)
    pose_frames = iio.imread(pose_video_path)

    history_frames = [video_frames[i] for i in range(23)]
    history_conds = [pose_frames[i] for i in range(23)]
    future_conds = [pose_frames[i] for i in range(23, 63)]

    client = WMClient(HOST, PORT)

    result = client.predict(history_frames, history_conds, future_conds)

    if isinstance(result, WMPredictionOutput):
        print("Prediction succeeded")
        print("full_video frames:", len(result.full_video))
        print("pred_frames:", len(result.pred_frames))
        print("pred_panels:", len(result.pred_panels))
        imageio.mimwrite("test_video.mp4", result.full_video, fps=16)
    else:
        print("Prediction failed:", result)

    client.close()



if __name__ == "__main__":
    main()