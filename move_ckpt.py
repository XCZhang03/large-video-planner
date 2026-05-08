import os
from pathlib import Path
import sys

run_id = "jyec0fvq"
run_path = Path(f"/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR/large-video-planner/outputs/checkpoint_links/awesome-wm/wan_at2v/{run_id}")
new_run_path = Path(f"/net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/checkpoints/lvp/outputs/checkpoint_links/awesome-wm/wan_at2v/{run_id}/checkpoints")
new_run_path.mkdir(parents=True, exist_ok=True)
for ckpt_path in run_path.glob("*.ckpt"):
    if "latest" in ckpt_path.stem:
        new_ckpt_path = new_run_path / f"{ckpt_path.relative_to(run_path)}"
        ckpt_path = ckpt_path.resolve()
        config_path = ckpt_path.parent.parent/".hydra"
        if not ckpt_path.exists():
            print(f"Checkpoint {ckpt_path} does not exist, skipping.")
            continue
        if new_ckpt_path.exists():
            os.system(f"rm {new_ckpt_path}")
        print(f"Copying {ckpt_path} to {new_ckpt_path}...")
        os.system(f"cp {ckpt_path} {new_ckpt_path}")
        os.system(f"cp -r {config_path} {new_run_path.parent}")
        print(f"Copied {ckpt_path} to {new_ckpt_path}")