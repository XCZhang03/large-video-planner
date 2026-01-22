import pandas as pd
from tqdm import tqdm
from pathlib import Path
from tqdm import trange
import torch
import numpy as np
import decord

from .cond_video import CondVideoDataset

from concurrent.futures import ThreadPoolExecutor, as_completed
import os


def filter_path(path: Path) -> bool:
    # Implement your filtering logic here
    dir_allowed_patterns = [
        "icml"
    ]
    file_allowed_patterns = [
        ""
    ]
    if path.is_dir():
        # return True
        if any(pattern in str(path) + '/' for pattern in dir_allowed_patterns):
            return True
    elif path.is_file():
        if any(pattern in str(path) for pattern in file_allowed_patterns) and any(pattern in str(path.parent) + '/' for pattern in dir_allowed_patterns):
            return True
    return False

class RobosuiteDataset(CondVideoDataset):
    def __init__(self, cfg, split="training"):
        self.override_fps = cfg.download.override_fps
        super().__init__(cfg, split)
        # Additional initialization if needed

    # Override or add any methods specific to RobosuiteDataset if necessary
    def download(self):
        # video_dirs = list(self.data_root.glob("*icml*/"))
        # if len(video_dirs) == 0:
        #     raise ValueError(f"No dataset directories found in {self.data_root} for RobosuiteDataset.")
        # video_dirs = [ d for d in video_dirs if filter_path(d)]
        # print(f"Found {len(video_dirs)} dataset directories in {self.data_root}.")

        # mp4_files = []
        # for video_dir in video_dirs:
        #     mp4_files.extend(list(video_dir.rglob("*.mp4")))

        # pairs = []
        # mp4_set = set(mp4_files)
        # split_data = self.data_root / "random_1000.csv"
        split_data = "/workspace/fan/xc/icml_wm/openpi/examples/libero/top_100_balanced.csv"
        split_data = pd.read_csv(split_data).to_dict(orient='records')
        # video_path_list = Path("/workspace/fan/xc/icml_wm/openpi/examples/libero/libero_10_demo").rglob("*chunk*.mp4")
        pairs = []
        for row in split_data:
        # for video_path in video_path_list:
            video_path = self.data_root / row["video_path"]
            action_path = Path(str(video_path).replace("agentview_chunk", "actions_chunk").replace('.mp4', '.npz'))
            # if row["split"] == "validation":
            #     pairs.append((video_path, action_path, "validation"))
            pairs.append((video_path, action_path, "training"))
            # pairs.append((video_path, action_path))
        
        self.video_pairs = pairs

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
                "fps": self.override_fps,
                "n_frames": n_frames,
                "width": 128,
                "height": 128,
                "split": split,
                # "trim_start": 0,
                # "trim_end": 49,
                }

        max_workers = min(16, os.cpu_count() or 4)
        records_parallel = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(process_pair, pair): pair for pair in self.video_pairs}
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

        metadata_path = self.data_root / self.metadata_path
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(records)
        df.to_csv(metadata_path, index=False)
        print(f"Created metadata CSV with {len(records)} videos at {metadata_path}")
        
    def cache_prompt_embed(self, cfg, encode=False):
        prompt_embed_path = (self.data_root / self.metadata_path).parent.parent / "robosuite_default_prompt.pt"
        neg_prompt_embed_path = (self.data_root / self.metadata_path).parent.parent / "robosuite_default_neg_prompt.pt"
        if encode:
            from algorithms.cogvideo.t5 import T5Encoder
            from algorithms.wan.modules.t5 import umt5_xxl
            from algorithms.wan.modules.tokenizers import HuggingfaceTokenizer
            text_encoder = (
                umt5_xxl(
                    encoder_only=True,
                    return_tokenizer=False,
                    dtype=torch.bfloat16,
                    device=torch.device("cpu"),
                )
                .eval()
                .requires_grad_(False)
            )
            text_encoder.load_state_dict(
                torch.load(
                    cfg.text_encoder.ckpt_path,
                    map_location="cpu",
                    weights_only=True,
                    # mmap=True,
                )
            )
            text_encoder = text_encoder.cuda()
            self.text_encoder = text_encoder
            # Initialize tokenizer
            self.tokenizer = HuggingfaceTokenizer(
                name=cfg.text_encoder.name,
                seq_len=cfg.text_encoder.text_len,
                clean="whitespace",
            )
            prompt_embed = self.encode_text([self.id_token])[0]
            prompt_embed_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving prompt embed shape {prompt_embed.shape} to {prompt_embed_path}")
            torch.save(prompt_embed.clone(), prompt_embed_path)

            neg_prompt = cfg.neg_prompt if hasattr(cfg, 'neg_prompt') else ""
            neg_prompt_embed = self.encode_text([neg_prompt])[0]
            neg_prompt_embed_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving neg prompt embed shape {neg_prompt_embed.shape} to {neg_prompt_embed_path}")
            torch.save(neg_prompt_embed.clone(), neg_prompt_embed_path)

        records = self.records

        metadata_path = self.data_root / self.metadata_path
        new_records = []

        for i in trange(0, len(records)):
            record = records[i]
            new_record = record.copy()
            new_record["prompt_embed_path"] = str(prompt_embed_path.absolute())
            new_record["negative_prompt_embed_path"] = str(neg_prompt_embed_path.absolute())
            new_records.append(new_record)

        df = pd.DataFrame(new_records)
        df.to_csv(metadata_path, index=False)
        print(f"Updated metadata CSV with {len(new_records)} videos to {metadata_path}")

    def encode_text(self, texts):
        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        ids = ids.to("cuda")
        mask = mask.to("cuda")
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]
