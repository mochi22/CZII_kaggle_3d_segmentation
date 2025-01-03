import os
import sys
import time
import hydra
import numpy as np
import torch
import warnings
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from utilities.seed import set_seed
from utilities.logger import get_logger
from src.dataloader import create_dataloader
from src.model import model2d1d, model25d

warnings.filterwarnings('ignore')

def loading_npy(cfg, names):
    files = []
    for name in names:
        image = np.load(f"{cfg.dir.TRAIN_DATA_DIR}/train_image_{name}.npy")
        label = np.load(f"{cfg.dir.TRAIN_DATA_DIR}/train_label_{name}.npy")
        files.append({"image": image, "label": label})
    return files


def save_cfg(cfg, exp_name):
    output_path = Path(cfg.dir.exp_dir)
    print(f"ouput_path: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    print("#"*10, "current cfg","#"*10)
    print(OmegaConf.to_yaml(cfg))
    print("##"*20)

    config_path = output_path / f"{exp_name}.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    print("seve cfg done. save path:", config_path)


@hydra.main(version_base=None, config_path=".", config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    # settign logger
    log_dir = Path(cfg.dir.exp_dir) / "logs"
    os.makedirs(log_dir, exist_ok=True)
    main_logger = get_logger('main_logger', log_dir / 'main.log')
    main_logger.info("Starting the main process")

    runtime_choices = HydraConfig.get().runtime.choices
    print("runtime_choices:", runtime_choices)
    exp_name = f"{runtime_choices.exp}"
    print(f"exp_name: {exp_name}")

    # check and save cfg
    save_cfg(cfg, exp_name)

    # setting seed
    set_seed(cfg.exp.seed)

    # loading data
    train_names = ['TS_5_4', 'TS_69_2', 'TS_6_6', 'TS_73_6', 'TS_86_3', 'TS_99_9']
    valid_names = ['TS_6_4']
    train_files = loading_npy(cfg, train_names)
    valid_files = loading_npy(cfg, valid_names)
    
    train_loader = create_dataloader(cfg, train_files)
    valid_loader = create_dataloader(cfg, valid_files, shuffle=False)

    # data2=next(iter(train_loader))
    # print("++"*10)
    # print(data2["image"].shape, data2["label"].shape)

    # modeling
    # model = model2d1d(
    model = model25d(
        n_channels=cfg.exp.model.n_channels, 
        n_classes=cfg.exp.model.n_classes,
        lr=cfg.exp.model.lr,
        arch=cfg.exp.model.arch,
        is_train=True
    )

    torch.set_float32_matmul_precision('medium')

    # Check if CUDA is available and then count the GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}, current GPU:{torch.cuda.current_device()}")
        devices = list(range(num_gpus))
        print(devices)
    else:
        print("No GPU available. Running on CPU.", torch.cuda.is_available())
        main_logger.error("This is no GPU!!!")
    


    # setting MLflow logging
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logger.mlflow.experiment_name,
        tracking_uri=cfg.logger.mlflow.tracking_uri
    )
    mlf_logger.log_hyperparams(cfg)


    # setting callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(cfg.dir.exp_dir) / 'checkpoints',
            filename='{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=cfg.exp.trainer.patience,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # https://lightning.ai/docs/pytorch/stable/common/trainer.html
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=cfg.exp.trainer.max_epochs,
        logger=mlf_logger,
        # callbacks=callbacks, # this makes any errors. https://github.com/huggingface/transformers/issues/3887
        log_every_n_steps=cfg.exp.trainer.log_every_n_steps,
        # val_check_interval=cfg.exp.trainer.val_check_interval,
        devices=cfg.exp.trainer.gpus,
        strategy=cfg.exp.trainer.strategy,
        # accumulate_grad_batches=cfg.exp.trainer.accumulate_grad_batches,
        # precision=cfg.exp.trainer.precision,
        # gradient_clip_val=cfg.exp.trainer.gradient_clip_val,
        # deterministic=True, # if True, this may be override to `torch.use_deterministic_algorithms(True)`. However `torch.use_deterministic_algorithms(True, warn_only=True)` is expected
    )

    # Train
    main_logger.info("Starting model training")
    trainer.fit(model, train_loader, valid_loader)
    main_logger.info("done model training")



if __name__ == "__main__":
    main()