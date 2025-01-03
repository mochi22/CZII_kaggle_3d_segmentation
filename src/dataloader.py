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

def create_dataloader(cfg, data_files, shuffle=True):
    transform = Compose([
        Resized(
            keys=["image", "label"],
            spatial_size=(cfg.exp.IMAGE_SIZE, cfg.exp.IMAGE_SIZE),  #(cfg.exp.NUM_SLICE, cfg.exp.IMAGE_SIZE, cfg.exp.IMAGE_SIZE),
            mode=("trilinear", "nearest")  # 画像は三線形補間、ラベルは最近傍補間
        ),
        NormalizeIntensityd(keys="image"),
        # ToTensord(keys=["image"]),
    ])

    # dataset
    dataset = SlicedVolumeDataset(data_files, num_slices=cfg.exp.NUM_SLICE, stride=cfg.exp.STRIDE, transform=transform) #num_slices=64, stride=32)

    # DataLoader
    loader = DataLoader(dataset, batch_size=cfg.exp.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.exp.NUM_WORKERS)

    return loader