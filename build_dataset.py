import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import decord
from concurrent.futures import ThreadPoolExecutor, as_completed

# split data contains the video_path and split ["training" or "validation"]
split_data = "/workspace/fan/xc/icml_wm/openpi/examples/libero/top_100_balanced.csv"
split_data = pd.read_csv(split_data).to_dict(orient='records')
metadata_path = Path("/workspace/fan/xc/icml_wm/large-video-planner/data/meta_data/libero_icml/libero_task4_oracle_200.csv")

pairs = []
# You can modify the split to be training or validation
for row in split_data:
    video_path = Path(row["video_path"])
    action_path = Path(str(video_path).replace("agentview_chunk", "actions_chunk").replace('.mp4', '.npz'))
    # if row["split"] == "validation":
    #     pairs.append((video_path, action_path, "validation"))
    pairs.append((video_path, action_path, "training"))
    # pairs.append((video_path, action_path))


records = []
def process_pair(pair):
    video_path, action_path, split = pair
    # video_path, action_path = pair

    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        return None
    if not action_path.exists():
        print(f"Action file not found: {action_path}")
        return None

    try:
        vr = decord.VideoReader(str(video_path))
        n_frames = len(vr)
        del vr
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None

    try:
        action = np.load(action_path)
        n_action_frames = action['actions'].shape[0]
    except Exception as e:
        print(f"Error loading action file {action_path}: {e}")
        return None

    if n_frames != n_action_frames:
        print(
            f"Frame count mismatch: {video_path} has {n_frames} frames, "
            f"but {action_path} has {n_action_frames} frames."
        )
        return None

    return {
        "video_path": str(video_path.resolve()),
        "cond_path": str(action_path.resolve()),
        "fps": 20,
        "n_frames": n_frames,
        "width": 128,
        "height": 128,
        "split": split,
        }

max_workers = min(16, os.cpu_count() or 4)
records_parallel = []
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = {ex.submit(process_pair, pair): pair for pair in pairs}
    for f in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Building metadata (parallel)",
    ):
        rec = f.result()
        if rec is not None:
            records_parallel.append(rec)
records = records_parallel
import random
random.shuffle(records)

metadata_path.parent.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(records)
df.to_csv(metadata_path, index=False)
print(f"Created metadata CSV with {len(records)} videos at {metadata_path}")