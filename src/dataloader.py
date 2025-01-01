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
    Resized
)
from .dataset import SlicedVolumeDataset

def create_dataloader(cfg, data_files, shuffle=True):
    transform = Compose([
        # AddChanneld(keys=["image", "label"]),
        # ScaleIntensityd(keys="image"),
        # Resized(keys=["image", "label"], spatial_size=(BATCH_SIZE, NUM_SLICE, 64, 64)),  # サイズを調整
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"), #画像とラベルの向きを "RAS" (Right-Anterior-Superior) 方向に統一
    ])

    # dataset
    dataset = SlicedVolumeDataset(data_files, num_slices=cfg.exp.NUM_SLICE, stride=cfg.exp.STRIDE, transform=transform) #num_slices=64, stride=32)

    # DataLoader
    loader = DataLoader(dataset, batch_size=cfg.exp.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.exp.NUM_WORKERS)

    return loader