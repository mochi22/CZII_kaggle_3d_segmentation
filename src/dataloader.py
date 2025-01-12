from monai.data import DataLoader
from monai.transforms import (
    Compose, 
    EnsureChannelFirstd, 
    Orientationd,  
    AsDiscrete,  
    RandFlipd, 
    RandRotate90d, 
    NormalizeIntensityd,
    RandCropByLabelClassesd,
    Resized,
    ToTensord
)
from .dataset import SlicedVolumeDataset

# def create_dataloader(cfg, file_ids, shuffle=True, is_train=True):
def create_dataloader(cfg, data_files, shuffle=True, is_train=True):
    if is_train:
        transform = Compose([
            Resized(
                keys=["image", "label"],
                spatial_size=(cfg.exp.IMAGE_SIZE, cfg.exp.IMAGE_SIZE),  #(cfg.exp.NUM_SLICE, cfg.exp.IMAGE_SIZE, cfg.exp.IMAGE_SIZE),
                mode=("trilinear", "nearest")  # 画像は三線形補間、ラベルは最近傍補間
            ),
            NormalizeIntensityd(keys="image"),
            # ToTensord(keys=["image"]),
        ])
    else:
        transform = Compose([
            Resized(
                keys=["image"],
                spatial_size=(cfg.exp.IMAGE_SIZE, cfg.exp.IMAGE_SIZE),  #(cfg.exp.NUM_SLICE, cfg.exp.IMAGE_SIZE, cfg.exp.IMAGE_SIZE),
                mode=("trilinear")  # 画像は三線形補間、ラベルは最近傍補間
            ),
            NormalizeIntensityd(keys="image"),
            # ToTensord(keys=["image"]),
        ])

    # if is_train:
    #     static_dir=f"{cfg.dir.TEST_DATA_DIR}/train/static/ExperimentRuns"
    #     overlay_dir=f"{cfg.dir.TEST_DATA_DIR}/train/overlay/ExperimentRuns"
    # else:
    #     static_dir=f"{cfg.dir.TEST_DATA_DIR}/test/static/ExperimentRuns"
    #     overlay_dir=f"{cfg.dir.TEST_DATA_DIR}/test/overlay/ExperimentRuns"

    # dataset
    dataset = SlicedVolumeDataset(
        data_files, 
        num_slices=cfg.exp.NUM_SLICE, 
        stride=cfg.exp.STRIDE, 
        transform=transform
    ) #num_slices=64, stride=32)

    # dataset = SlicedVolumeDataset(
    #     ids=file_ids,
    #     static_dir=static_dir, # cfg.dir.TRAIN_DATA_DIRはkaggleのコンペのデータセットを読み込む
    #     overlay_dir=overlay_dir,
    #     num_slices=cfg.exp.NUM_SLICE,
    #     stride=cfg.exp.STRIDE,
    #     transform=transform,
    #     is_train=is_train
    #     ##num_slices=64,stride=32)
    # )

    # DataLoader
    loader = DataLoader(dataset, batch_size=cfg.exp.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.exp.NUM_WORKERS)

    return loader