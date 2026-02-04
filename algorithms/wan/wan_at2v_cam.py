import torch
from einops import rearrange
from .wan_at2v_base import WanActionTextToVideoBase


class WanActionTextToVideoCam(WanActionTextToVideoBase):
    def __init__(self, cfg):
        super().__init__(cfg)

    def check_cfg(self):
        super().check_cfg()
        self.robot_cond_mode, self.cam_cond_mode = self.diffusion_forcing.cond_mode.split("+")
        assert self.robot_cond_mode == "concat"
        assert self.cam_cond_mode in ['global', 'ray', 'ray-encoding', 'plucker']
    
    def configure_model(self):
        super().configure_model()
        from .modules.action_encoder import CameraPoseEncoder
        self.cam_encoder = CameraPoseEncoder(
            dim=self.model.dim,
            cond_mode=self.cam_cond_mode,
            normalization=self.diffusion_forcing.get("normalization", "none")
        )
        if self.cfg.model.tuned_ckpt_path is not None:
            incompatiable_keys = self.cam_encoder.load_state_dict(
                self._load_tuned_state_dict(prefix="cam_encoder."),
                strict=False,
            )
            print("WanActionTextToVideoCam: loaded cam_encoder with incompatiable keys:", incompatiable_keys)
        if not self.is_inference:
            self.cam_encoder.to(self.dtype).train()


    def prepare_embeds(self, batch, **kwargs):
        batch = super().prepare_embeds(batch, **kwargs)
        with torch.enable_grad():
            batch['cam_cond_lat'] = self.prepare_cam_embeds(batch, **kwargs)
        batch['cond_lat'] = {
            'robot_cond_lat': batch['robot_cond_lat'],
            'cam_cond_lat': batch['cam_cond_lat'],
        }
        return batch

    @torch.enable_grad()
    def prepare_cam_embeds(self, batch, **kwargs):
        raw_camera_poses = batch['camera_poses']  # (B, T, num_cams, pose_dim)
        # get sparse history frames
        indices = list(self.hist_steps) + list(range(self.max_frames - self.pred_len, self.max_frames))
        raw_camera_poses = raw_camera_poses[:, indices]
        video_metadata = batch['video_metadata']
        camera_hidden_states = self.cam_encoder(raw_camera_poses, video_metadata)
        return camera_hidden_states

    @torch.no_grad()
    def prepare_video_embeds(self, batch, **kwargs):
        videos = batch["videos"]
        conds = batch["conds"]
        batch_size, t, _, h, w = videos.shape

        if t != self.max_frames:
            raise ValueError(f"Number of frames in videos must be {self.max_frames}")
        if h != self.height or w != self.width:
            raise ValueError(
                f"Height and width of videos must be {self.height} and {self.width}"
            )

        # get sparse history frames
        indices = list(self.hist_steps) + list(range(self.max_frames - self.pred_len, self.max_frames))
        assert len(indices) == self.n_frames, \
            f"Total selected frames {len(indices)} not equal to model n_frames {self.n_frames}"
        videos = videos[:, indices]
        conds = conds[:, indices]

        video_lat = self.encode_video(rearrange(videos, "b t c h w -> b c t h w"))
        # video_lat ~ (b, lat_c, lat_t, lat_h, lat_w)
        batch["video_lat"] = video_lat

        cond_lat = self.encode_video(rearrange(conds, "b t c h w -> b c t h w"))        
        batch["robot_cond_lat"] = cond_lat

        return batch