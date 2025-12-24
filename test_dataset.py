import hydra
from omegaconf import DictConfig, OmegaConf
from datasets.robosuite import RobosuiteDataset
from algorithms.wan import WanActionTextToVideo
import torch

@hydra.main(version_base=None, config_path="./configurations", config_name="config.yaml")
def main(cfg: DictConfig):
    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg))

    dataset = RobosuiteDataset(cfg.dataset)
    # dataset.cache_prompt_embed(cfg.algorithm)
    print(f"RobosuiteDataset instantiated, length {len(dataset)}.")
    data = dataset[0]
    print(f"Sample data keys: {list(data.keys())}")
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=1,
    )
    batch = next(iter(dl))

    # algorithm = WanActionToVideo(cfg.algorithm).to("cuda")
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
    #         batch[k] = v.to(device="cuda", dtype=algorithm.dtype)
    # batch = algorithm.on_after_batch_transfer(batch, 0)  
    # loss_dict = algorithm.training_step(batch, 0)

if __name__ == "__main__":
    main()