import os
import sys
import time
import hydra
import numpy as np
import torch
import warnings
import wandb
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
# from lightning.pytorch.callbacks import WandbCallback
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from utilities.seed import set_seed
from utilities.logger import get_logger
from src.dataloader import create_dataloader, TomogramDataModule
from src.model import model2d1d, model25d, ProteinSegmentor25D, SegmentorMid25d, UMC

warnings.filterwarnings('ignore')

def loading_npy(cfg, names):
    files = []
    for name in names:
        image = np.load(f"{cfg.dir.TRAIN_DATA_DIR}/train_image_{name}.npy")
        label = np.load(f"{cfg.dir.TRAIN_DATA_DIR}/train_label_{name}.npy")
        # files.append({"image": image, "label": label})
        files.append({"filename": name, "image": image, "label": label})
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

    ## 0xx系のexperimentsで使う
    # train_loader = create_dataloader(cfg, train_files, shuffle=True, is_train=True)
    # valid_loader = create_dataloader(cfg, valid_files, shuffle=True, is_train=True)

    ## 1xx系のexperimentsで使う
    # train_loader = create_dataloader(cfg, train_files)
    # valid_loader = create_dataloader(cfg, valid_files, shuffle=False)

    ## 2xx系
    data_module = TomogramDataModule(
        train_files=train_files,
        val_files=valid_files,
        batch_size=cfg.exp.BATCH_SIZE,
        patch_size=cfg.exp.patch_size,
        overlap=cfg.exp.overlap,
        num_workers=cfg.exp.NUM_WORKERS,
        target_size=cfg.exp.target_size
    )


    # data2=next(iter(train_loader))
    # print("++"*10)
    # print(data2["image"].shape, data2["label"].shape) # torch.Size([1, 32, 256, 256]) torch.Size([1, 7, 32, 256, 256])

    # modeling
    # model = model2d1d(

    ## 0xx系のexperimentsで使う
    # model = model25d(
    #     n_channels=cfg.exp.model.n_channels, 
    #     n_classes=cfg.exp.model.n_classes,
    #     lr=cfg.exp.model.lr,
    #     arch=cfg.exp.model.arch,
    #     is_train=True
    # )

    # model=SegmentorMid25d(
    #     n_channels=cfg.exp.model.n_channels, 
    #     n_classes=cfg.exp.model.n_classes,
    #     lr=cfg.exp.model.lr,
    #     arch=cfg.exp.model.arch,
    #     is_train=True,
    #     n_frames=cfg.exp.NUM_SLICE
    # )

    ## 1xx系のexperimentsで使う
    # model = ProteinSegmentor25D(in_ch=cfg.exp.model.n_channels, out_ch=cfg.exp.model.n_classes, frames_per_group=1, lr=cfg.exp.model.lr)

    ## 2xx
    model = UMC(
        n_channels=cfg.exp.model.n_channels, 
        n_classes=cfg.exp.model.n_classes,
        lr=cfg.exp.model.lr,
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
    # mlf_logger = MLFlowLogger(
    #     experiment_name=cfg.exp.logger.mlflow.experiment_name,
    #     tracking_uri=cfg.exp.logger.mlflow.tracking_uri,
    #     run_name=cfg.exp.logger.mlflow.run_name
    # )
    # mlf_logger.log_hyperparams(cfg)
    # MLFlowLoggerの代わりに以下を使用
    wandb_logger = WandbLogger(
        project=cfg.exp.logger.wandb.experiment_name,
        name=cfg.exp.logger.wandb.run_name,
        save_dir=Path(cfg.dir.exp_dir) / f'wandb_logs/{cfg.exp.logger.wandb.run_name}/',
        log_model=True
    )
    wandb_logger.log_hyperparams(cfg)

    # mlflow.start_run(run)
    # mlflow.pytorch.autolog()
    # mlflow.set_experiment(cfg.exp.logger.mlflow.experiment_name)
    # callbacks.append(WandbCallback())



    # setting callbacks
    if cfg.exp.USE_CALLBACK is not None:
        callbacks = [
            ModelCheckpoint(
                dirpath=Path(cfg.dir.exp_dir) / f'checkpoints/{cfg.exp.logger.wandb.run_name}/',
                filename='{epoch:02d}-{val_loss:.2f}',
                save_top_k=3,
                # monitor='val_loss',
                monitor='val_total_loss',
                mode='min'
            ),
            EarlyStopping(
                # monitor='val_loss',
                monitor='val_total_loss',
                patience=cfg.exp.trainer.patience,
                mode='min'
            ),
            LearningRateMonitor(logging_interval='step')
        ]
    else:
        callbacks=[]
    
    # https://lightning.ai/docs/pytorch/stable/common/trainer.html
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=cfg.exp.trainer.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks, # this makes any errors. https://github.com/huggingface/transformers/issues/3887
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
    # trainer.fit(model, train_loader, valid_loader)

    ## 2xx
    trainer.fit(model, data_module)
    main_logger.info("done model training")



if __name__ == "__main__":
    main()