import torch
import torch.nn as nn
from einops import rearrange, repeat
from transformers import get_scheduler
from .modules.clip import clip_xlm_roberta_vit_h_14
from .wan_t2v import WanTextToVideo


class WanImageToVideo(WanTextToVideo):
    """
    Main class for WanImageToVideo, inheriting from WanTextToVideo
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg.model.in_dim = self.cfg.vae.z_dim * 2 + 4

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

    def build_metrics(self):
        self.metric = None
        return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.parameters(), "lr": self.cfg.lr},
                {"params": self.vae.parameters(), "lr": 0},
                {"params": self.clip.parameters(), "lr": 0},
            ],
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.betas,
        )
        # optimizer = torch.optim.AdamW(
        #     self.model.parameters(),
        #     lr=self.cfg.lr,
        #     weight_decay=self.cfg.weight_decay,
        #     betas=self.cfg.betas,
        # )
        lr_scheduler_config = {
            "scheduler": get_scheduler(
                optimizer=optimizer,
                **self.cfg.lr_scheduler,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

    def clip_features(self, videos):
        size = (self.clip.image_size,) * 2
        videos = rearrange(videos, "b t c h w -> (b t) c h w")
        videos = nn.functional.interpolate(
            videos, size=size, mode="bicubic", align_corners=False
        )
        videos = self.clip_normalize(videos.mul_(0.5).add_(0.5))
        return self.clip.visual(videos, use_31_block=True)

    @torch.no_grad()
    def prepare_embeds(self, batch, **kwargs):
        batch = super().prepare_embeds(batch, **kwargs)

        videos = batch["videos"]
        images = videos[:, :1]
        has_bbox = batch["has_bbox"]  # [B, 2]
        bbox_render = batch["bbox_render"]  # [B, 2, H, W]

        batch_size, t, _, h, w = videos.shape
        lat_c, lat_t, lat_h, lat_w = self.lat_c, self.lat_t, self.lat_h, self.lat_w

        clip_embeds = self.clip_features(images)
        batch["clip_embeds"] = clip_embeds

        mask = torch.zeros(
            batch_size,
            self.vae_stride[0],
            lat_t,
            lat_h,
            lat_w,
            device=self.device,
            dtype=self.dtype,
        )
        # after the ckpt hack, we repurpose the 4 mask channels for bounding box conditioning
        # second last channel is indicator of bounding box
        mask[:, 2, 0] = has_bbox[..., 0, None, None]
        mask[:, 2, -1] = has_bbox[..., -1, None, None]
        # Interpolate bbox_render to match latent dimensions
        bbox_render_resized = nn.functional.interpolate(
            bbox_render,
            size=(lat_h, lat_w),
            mode="bicubic",
            align_corners=False,
        )
        # last channel is renderred bbox
        mask[:, 3, 0] = bbox_render_resized[:, 0]
        mask[:, 3, -1] = bbox_render_resized[:, -1]

        if self.diffusion_forcing.enabled:
            image_embeds = torch.zeros(
                batch_size,
                4 + lat_c,
                lat_t,
                lat_h,
                lat_w,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            padded_images = torch.zeros(batch_size, 3, t - 1, h, w, device=self.device)
            padded_images = torch.cat(
                [rearrange(images, "b 1 c h w -> b c 1 h w"), padded_images], dim=2
            )
            image_embeds = self.encode_video(
                padded_images
            )  # b, lat_c, lat_t, lat_h, lat_w
            image_embeds = torch.cat([mask, image_embeds], 1)
            mask[:, :2, 0] = 1
        batch["image_embeds"] = image_embeds

        return batch

    