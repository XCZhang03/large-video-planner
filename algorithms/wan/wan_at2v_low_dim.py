import torch
from .wan_at2v_base import WanActionTextToVideoBase


class WanActionTextToVideoLowDim(WanActionTextToVideoBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def check_cfg(self):
        super().check_cfg()
        assert self.diffusion_forcing.enabled and self.diffusion_forcing.cond_mode in ["embed", "cross-attn"], \
            "WanActionTextToVideo with low-dim actions supports embedding and cross-attn."

    def configure_model(self):
        super().configure_model()
        action_dim = self.cfg.get("action_dim", 7)
        from .modules.action_encoder import ActionEncoder
        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            hidden_dim=self.model.dim if self.diffusion_forcing.cond_mode == "embed" else self.model.text_dim,
            AdaLN_proj=(self.diffusion_forcing.cond_mode == "embed")
        )
        if self.cfg.model.tuned_ckpt_path is not None:
            self.action_encoder.load_state_dict(
                self._load_tuned_state_dict(prefix="action_encoder."),
            )
        if not self.is_inference:
            self.action_encoder.to(self.dtype).train()


    @torch.no_grad()
    def prepare_embeds(self, batch, **kwargs):
        batch = super().prepare_embeds(batch, **kwargs)

        actions = batch['low_dim_conds']
        with torch.enable_grad():
             actions_hidden_states  = self.action_encoder(actions)
        
        if self.diffusion_forcing.cond_mode == "embed":
            batch['cond_lat'] = actions_hidden_states
        else:
            batch['cond_lat'] = None

        if self.diffusion_forcing.cond_mode == "cross-attn":
            batch['prompt_embeds'] = actions_hidden_states
            batch['negative_prompt_embeds'] = torch.zeros_like(batch['prompt_embeds'])
        return batch