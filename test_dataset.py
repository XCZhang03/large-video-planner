import hydra
from omegaconf import DictConfig, OmegaConf
from datasets.robosuite import RobosuiteDataset
from algorithms.wan.wan_at2v_low_dim import WanActionTextToVideoLowDim
from algorithms.wan.wan_at2v import WanActionTextToVideo
from algorithms.wan.wan_at2v_cam import WanActionTextToVideoCam
import torch

@hydra.main(version_base=None, config_path="./configurations", config_name="config.yaml")
def main(cfg: DictConfig):
    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg))
    cfg.dataset.width = 64
    cfg.dataset.height = 64
    cfg.algorithm.diffusion_forcing.cond_mode = "concat+global"

    dataset = RobosuiteDataset(cfg.dataset, split="all")
    print(f"RobosuiteDataset instantiated, length {len(dataset)}.")
    data = dataset[0]
    print(f"Sample data keys: {list(data.keys())}")
    # dl = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=16,
    #     shuffle=True,
    #     num_workers=1,
    # )
    # batch = next(iter(dl))

    # algorithm = WanActionTextToVideoCam(cfg.algorithm).to("cuda")
    # algorithm.configure_model()
    # algorithm = algorithm.to("cuda")
    # print(f"Algorithm instantiated.")

    # dl = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=2,
    #     shuffle=True,
    #     num_workers=1,
    # )
    # batch = next(iter(dl))
    # for k, v in batch.items():
    #     if isinstance(v, torch.Tensor):
    #         batch[k] = v.to(device="cuda", dtype=algorithm.dtype if not v.dtype == torch.long else torch.long)
    # batch = algorithm.on_after_batch_transfer(batch, 0)  
    # loss = algorithm.training_step(batch, 0)
    # loss.backward()
    # for name, param in algorithm.named_parameters():
    #     if param.requires_grad and param.grad is None:
    #         print(f"Parameter {name} has no gradient!")

if __name__ == "__main__":
    main()