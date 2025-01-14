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
import albumentations as A
from .dataset import SlicedVolumeDataset, ProteinSegmentationDataset, TomogramDataset



## 0XX 系の時に使う
# def create_dataloader(cfg, file_ids, shuffle=True, is_train=True):
def create_dataloader(cfg, data_files, shuffle=True, is_train=True):
    if is_train:
        transform = Compose([
            Resized(
                keys=["image", "label"],
                spatial_size=(cfg.exp.IMAGE_SIZE, cfg.exp.IMAGE_SIZE),  #(cfg.exp.NUM_SLICE, cfg.exp.IMAGE_SIZE, cfg.exp.IMAGE_SIZE),
                mode=("trilinear", "nearest")  # 画像は三線形補間、ラベルは最近傍補間
            ),
            # NormalizeIntensityd(keys="image"),
            # ToTensord(keys=["image"]),
        ])
    else:
        transform = Compose([
            Resized(
                keys=["image"],
                spatial_size=(cfg.exp.IMAGE_SIZE, cfg.exp.IMAGE_SIZE),  #(cfg.exp.NUM_SLICE, cfg.exp.IMAGE_SIZE, cfg.exp.IMAGE_SIZE),
                mode=("trilinear")  # 画像は三線形補間、ラベルは最近傍補間
            ),
            # NormalizeIntensityd(keys="image"),
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
    )

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




## 1XX 系のexperimentsで使う
# def create_dataloader(cfg, data_files, shuffle=True, is_train=True):
#     if is_train:
#         transform = A.Compose(
#             [
#                 A.Resize(184,height=cfg.exp.IMAGE_SIZE, width=cfg.exp.IMAGE_SIZE, p=1),
#                 # A.RandomResizedCrop(height=cfg.exp.IMAGE_SIZE, width=cfg.exp.IMAGE_SIZE, scale=(0.8, 1.0), p=1.0),
#                 # A.HorizontalFlip(p=p),
#                 # A.VerticalFlip(p=p),
#                 # A.RandomRotate90(p=p),
#                 # A.OneOf([
#                 #     A.GaussianBlur(blur_limit=(3, 7), p=p),
#                 #     A.MedianBlur(blur_limit=5, p=p),
#                 # ], p=p),
#                 # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=p),
#             ], 
#             p=1.0
#         )
#     else:
#         transform = A.Compose(
#             [
#                 A.Resize(184,height=cfg.exp.IMAGE_SIZE, width=cfg.exp.IMAGE_SIZE, p=1),
#             ],
#             p=1.0
#         )

#     # dataset
#     # dataset = SlicedVolumeDataset(
#     #     data_files, 
#     #     num_slices=cfg.exp.NUM_SLICE, 
#     #     stride=cfg.exp.STRIDE, 
#     #     transform=transform
#     # )

#     dataset = ProteinSegmentationDataset(data_files, transform=transform, inference_mode=False)
#     # DataLoader
#     loader = DataLoader(dataset, batch_size=cfg.exp.BATCH_SIZE, shuffle=shuffle, num_workers=cfg.exp.NUM_WORKERS)

#     return loader


## 2xx
# データモジュールの定義
import lightning.pytorch as pl
# class TomogramDataModule(pl.LightningDataModule):
#     def __init__(self, train_files, val_files, batch_size=8, patch_size=64, 
#                  overlap=0.75, num_workers=4):
#         super().__init__()
#         self.train_files = train_files
#         self.val_files = val_files
#         self.batch_size = batch_size
#         self.patch_size = patch_size
#         self.overlap = overlap
#         self.num_workers = num_workers

#     def setup(self, stage=None):
#         self.train_dataset = TomogramDataset(
#             files=self.train_files,
#             patch_size=self.patch_size,
#             overlap=self.overlap,
#             augment=True
#         )
        
#         self.val_dataset = TomogramDataset(
#             files=self.val_files,
#             patch_size=self.patch_size,
#             overlap=self.overlap,
#             augment=False
#         )
        
#         print(f"Number of training patches: {len(self.train_dataset)}")
#         print(f"Number of validation patches: {len(self.val_dataset)}")

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             pin_memory=True
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             pin_memory=True
#         )
class TomogramDataModule(pl.LightningDataModule):
    def __init__(self, train_files, val_files, batch_size=8, patch_size=64, 
                 overlap=0.75, target_size=(184, 128, 128), num_workers=4):
        super().__init__()
        self.train_files = train_files
        self.val_files = val_files
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.overlap = overlap
        self.target_size = target_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = TomogramDataset(
            files=self.train_files,
            patch_size=self.patch_size,
            overlap=self.overlap,
            target_size=self.target_size,
            augment=True
        )
        
        self.val_dataset = TomogramDataset(
            files=self.val_files,
            patch_size=self.patch_size,
            overlap=self.overlap,
            target_size=self.target_size,
            augment=False
        )
        
        print(f"Number of training patches: {len(self.train_dataset)}")
        print(f"Number of validation patches: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )