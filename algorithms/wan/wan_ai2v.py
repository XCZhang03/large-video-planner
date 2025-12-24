import torch
import torch.nn as nn
from tqdm import tqdm
from einops import rearrange, repeat
from transformers import get_scheduler
from .modules.clip import clip_xlm_roberta_vit_h_14
from .wan_i2v import WanImageToVideo



class WanActionImageToVideo(WanImageToVideo):
    """
    Main class for WanActionToVideo, inheriting from WanImageToVideo
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        assert self.diffusion_forcing.cond_mode in ["seq", "channel"], \
            f"Unsupported cond_mode {self.diffusion_forcing.cond_mode} for WanAction"

    def configure_model(self):
        # Call parent's configure_model first
        super().configure_model()

        if self.cfg.model.tuned_ckpt_path is None:
            self.model.hack_embedding_ckpt()

        # Additionally initialize CLIP for image encoding
        clip, clip_transform = clip_xlm_roberta_vit_h_14(
            pretrained=False,
            return_transforms=True,
            return_tokenizer=False,
            dtype=torch.float16 if self.is_inference else self.dtype,
            device="cpu",
        )
        if self.cfg.clip.ckpt_path is not None:
            clip.load_state_dict(
                torch.load(
                    self.cfg.clip.ckpt_path, map_location="cpu", weights_only=True
                )
            )
        if self.cfg.clip.compile:
            clip = torch.compile(clip)
        self.clip = clip
        self.clip_normalize = clip_transform.transforms[-1]


    @torch.no_grad()
    def prepare_embeds(self, batch):
        batch = super().prepare_embeds(batch)

        # encode cond video
        batch['cond_lat'] = self.encode_video(rearrange(batch['conds'], "b t c h w -> b c t h w"))
        assert batch['cond_lat'].shape == batch['video_lat'].shape

        assert self.diffusion_forcing.enabled
        
        if self.cfg.diffusion_forcing.cond_mode == "channel":
            batch["image_embeds"][:, 4:] = batch["cond_lat"]

        return batch
    
    def training_step(self, batch, batch_idx=None):
        batch = self.prepare_embeds(batch)
        clip_embeds = batch["clip_embeds"]
        image_embeds = batch["image_embeds"]
        prompt_embeds = batch["prompt_embeds"]
        video_lat = batch["video_lat"]
        cond_lat = batch["cond_lat"]

        noisy_lat, noise, t = self.add_training_noise(video_lat)
        flow = noise - video_lat

        flow_pred = self.model(
            noisy_lat,
            t=t,
            context=prompt_embeds,
            clip_fea=clip_embeds,
            seq_len=self.max_tokens,
            y=image_embeds,
            cond=cond_lat if self.diffusion_forcing.cond_mode == "seq" else None,
        )
        loss = torch.nn.functional.mse_loss(flow_pred, flow)

        if self.global_step % self.cfg.logging.loss_freq == 0:
            self.log("train/loss", loss, sync_dist=True)

        return loss

    @torch.no_grad()
    def sample_seq(self, batch, hist_len=1, pbar=None):
        """
        Main sampling loop. Only first hist_len frames are used for conditioning
        batch: dict
            batch["videos"]: [B, T, C, H, W]
            batch["prompts"]: [B]
        """
        if (hist_len - 1) % self.vae_stride[0] != 0:
            raise ValueError(
                "hist_len - 1 must be a multiple of vae_stride[0] due to temporal vae. "
                f"Got {hist_len} and vae stride {self.vae_stride[0]}"
            )
        hist_len = (hist_len - 1) // self.vae_stride[0] + 1  #  length in latent

        self.inference_scheduler, self.inference_timesteps = self.build_scheduler(False)
        lang_guidance = self.lang_guidance if self.lang_guidance else 0
        hist_guidance = self.hist_guidance if self.hist_guidance else 0

        batch = self.prepare_embeds(batch)
        clip_embeds = batch["clip_embeds"]
        image_embeds = batch["image_embeds"]
        prompt_embeds = batch["prompt_embeds"]
        video_lat = batch["video_lat"]
        cond_lat = batch["cond_lat"]

        batch_size = video_lat.shape[0]

        video_pred_lat = torch.randn_like(video_lat)
        if self.lang_guidance:
            if self.load_prompt_embed:
                neg_prompt_embeds = batch["negative_prompt_embeds"]
            else:
                neg_prompt_embeds = self.encode_text(
                    [self.neg_prompt] * len(batch["prompts"])
                )
        if pbar is None:
            pbar = tqdm(range(len(self.inference_timesteps)), desc="Sampling")
        for t in self.inference_timesteps:
            if self.diffusion_forcing.enabled:
                video_pred_lat[:, :, :hist_len] = video_lat[:, :, :hist_len]
                t_expanded = torch.full((batch_size, self.lat_t), t, device=self.device)
                t_expanded[:, :hist_len] = self.inference_timesteps[-1]
            else:
                t_expanded = torch.full((batch_size,), t, device=self.device)

            # normal conditional sampling
            flow_pred = self.model(
                video_pred_lat,
                t=t_expanded,
                context=prompt_embeds,
                seq_len=self.max_tokens,
                clip_fea=clip_embeds,
                y=image_embeds,
                cond=cond_lat if self.diffusion_forcing.cond_mode == "seq" else None,
            )

            # language unconditional sampling
            if lang_guidance:
                no_lang_flow_pred = self.model(
                    video_pred_lat,
                    t=t_expanded,
                    context=neg_prompt_embeds,
                    seq_len=self.max_tokens,
                    clip_fea=clip_embeds,
                    y=image_embeds,
                    cond=cond_lat if self.diffusion_forcing.cond_mode == "seq" else None,
                )
            else:
                no_lang_flow_pred = torch.zeros_like(flow_pred)

            # history guidance sampling:
            if hist_guidance and self.diffusion_forcing.enabled:
                no_hist_video_pred_lat = video_pred_lat.clone()
                no_hist_video_pred_lat[:, :, :hist_len] = torch.randn_like(
                    no_hist_video_pred_lat[:, :, :hist_len]
                )
                t_expanded[:, :hist_len] = self.inference_timesteps[0]
                no_hist_flow_pred = self.model(
                    no_hist_video_pred_lat,
                    t=t_expanded,
                    context=prompt_embeds,
                    seq_len=self.max_tokens,
                    clip_fea=clip_embeds,
                    y=image_embeds,
                    cond=cond_lat if self.diffusion_forcing.cond_mode == "seq" else None,
                )
            else:
                no_hist_flow_pred = torch.zeros_like(flow_pred)

            flow_pred = flow_pred * (1 + lang_guidance + hist_guidance)
            flow_pred = (
                flow_pred
                - lang_guidance * no_lang_flow_pred
                - hist_guidance * no_hist_flow_pred
            )

            video_pred_lat = self.remove_noise(flow_pred, t, video_pred_lat)
            pbar.update(1)

        video_pred_lat[:, :, :hist_len] = video_lat[:, :, :hist_len]

        video_pred = self.decode_video(video_pred_lat)
        video_pred = rearrange(video_pred, "b c t h w -> b t c h w")

        return video_pred

