import torch
import pytorch_lightning as pl
from src.data_module import DataModule
from src.model import ResNetModule, ResultWriter
from pytorch_lightning import Trainer
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="hifigan", version_base=None)
def main(cfg : DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed)
    data_module = DataModule(cfg.data)
    model = ResNetModule(**cfg.model)
    writer = ResultWriter(cfg.predict.output_dir, 'test', 'epoch', cfg.data)
    trainer = Trainer(**cfg.trainer, callbacks=[writer])
    trainer.predict(model, dataloaders=data_module.test_dataloader(), ckpt_path=cfg.predict.ckpt_path)

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()