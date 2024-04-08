import torch
import pytorch_lightning as pl
from src.data_module import DataModule
from src.model import ResNetModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="resnet", version_base=None)
def main(cfg : DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed)
    data_module = DataModule(cfg.data)
    model = ResNetModule(**cfg.model)
    logger = TensorBoardLogger(save_dir='tblog', name=cfg.log.log_dir)
    checkpoint_callback = ModelCheckpoint(**cfg.checkpoint)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor]
    )
    trainer.fit(model, data_module, ckpt_path=cfg.ckpt_path)

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()