import numpy as np
from src.dataset import SlicedVolumeDataset, ProteinSegmentationDataset
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
    ToTensord,
    Transform
)
import albumentations as A
import torch
import os

from monai.losses import TverskyLoss
from src.model import SegmentorMid25d
import segmentation_models_pytorch as smp
import torch.nn as nn
from monai.losses import DiceLoss

# class ComboLoss(nn.Module):
#     def __init__(self, alpha=0.5, smooth=1e-5, class_weights=torch.tensor([0,1,0,2,1,2,1])):
#         super(ComboLoss, self).__init__()
#         self.alpha = alpha
#         print("class_weights.view(1, 7, 1, 1, 1):", class_weights.view(1, 7, 1, 1, 1))
#         self.bce = nn.BCEWithLogitsLoss(weight=class_weights.view(1, 7, 1, 1, 1))  # (1, 7, 1, 1, 1))
#         # self.ce = nn.CrossEntropyLoss()
#         # self.dice = customDiceLoss(smooth=smooth)
#         self.dice = DiceLoss(include_background=False, to_onehot_y=False, softmax=True)
        
#     def forward(self, pred, target):
#         target = target.float() # これやらないとタイプエラーになる
#         print("pred, target:", pred.shape, target.shape)
#         bce_loss = self.bce(pred, target)
#         dice_loss = self.dice(pred, target)
#         # dice_loss = self.dice(torch.sigmoid(pred), target)
#         return self.alpha * bce_loss + (1 - self.alpha) * dice_loss
#         # return self.alpha * ce_loss + (1 - self.alpha) * dice_loss
#         # return bec_loss
# loss = ComboLoss()
# a=torch.randn(([1, 7, 32, 256, 256]))
# b=torch.randn(([1, 7, 32, 256, 256]))

# print("loss:", loss(a, b))

# model = SegmentorMid25d(arch= 'resnet50')
is_train=True
# backbone = smp.Unet(
#     encoder_name="resnet101",
#     encoder_weights='imagenet' if is_train else None,
#     in_channels=1,
#     classes=7,
#     decoder_channels=[ch * 8 for ch in (256, 128, 64, 32, 16)],
# ).encoder
# # print("model:", model)
# inputs = torch.randint(0, 7, (1, 184, 256, 256)).float()
# inputs=inputs.reshape(1 * 184, 1, 256, 256)
# print(backbone)
# features= backbone(inputs)
# for i in range(len(features)):
#     print(i, features[i].shape)
# # print(backbone.out_channels[1:]) (64, 256, 512, 1024, 2048)
# class Conv3dBlock(torch.nn.Sequential):
#     def __init__(
#         self, in_channels: int, out_channels: int, kernel_size: tuple[int, int, int], padding: tuple[int, int, int]
#     ):
#         super().__init__(
#             torch.nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, padding_mode="replicate"),
#             torch.nn.BatchNorm3d(out_channels),
#             torch.nn.LeakyReLU(),
#         )
# k=3
# conv3ds = nn.ModuleList([
#             nn.Sequential(
#                 Conv3dBlock(ch, ch, (2, k, k), (0, k // 2, k // 2)),
#                 Conv3dBlock(ch, ch, (2, k, k), (0, k // 2, k // 2)),
#             )
#             for ch in backbone.out_channels[1:]
# ])
# # print([(conv3d, feature.shape) for conv3d, feature in zip(conv3ds, features[1:])])

# decoder = smp.Unet(
#     encoder_name="resnet101",
#     encoder_weights='imagenet' if is_train else None,
#     in_channels=1,
#     classes=7,
#     decoder_channels=[ch * 8 for ch in (256, 128, 64, 32, 8)],
# ).decoder
# xxx=0
# features = [
#     torch.randn([32, 1, 256, 256]),   # C1
#     torch.randn([1, 64, 128, 128]),   # C2
#     torch.randn([1, 256, 64, 64]),   # C3
#     torch.randn([1, 512, 32, 32]),  # C4
#     torch.randn([1, 1024, 16, 16]),    # C5
#     torch.randn([1, 2048, 8, 8]),
#     # torch.randn([1, 64, 30*xxx, 128, 128]),   # C2
#     # torch.randn([1, 256, 30*xxx, 64, 64]),   # C3
#     # torch.randn([1, 512, 30*xxx, 32, 32]),  # C4
#     # torch.randn([1, 1024, 30*xxx, 16, 16]),    # C5
#     # torch.randn([1, 2048, 30*xxx, 8, 8]),
# ]

# print("--"*10)
# for i in range(len(features)):
#     print(i, features[i].shape)
# # @@@: 0 torch.Size([32, 1, 256, 256])
# # @@@: 1 torch.Size([1, 64, 30, 128, 128])
# # @@@: 2 torch.Size([1, 256, 30, 64, 64])
# # @@@: 3 torch.Size([1, 512, 30, 32, 32])
# # @@@: 4 torch.Size([1, 1024, 30, 16, 16])
# # @@@: 5 torch.Size([1, 2048, 30, 8, 8])
# print(decoder(*features))
# print("==="*10)

# TverskyLossの初期化
# loss_function = TverskyLoss(
#     include_background=True,
#     to_onehot_y=False,  # ターゲットが既にone-hotエンコーディングの場合はFalse
#     softmax=True,
#     alpha=0.5,
#     beta=0.5
# )

# # モデルの出力（予測値）
# # 形状: (batch_size, num_classes, *spatial_dims)
# inputs = torch.randn(4, 7, 64, 64, 64)  # 3Dの例：バッチサイズ4、7クラス、64x64x64ボクセル

# # 正解ラベル（既にone-hotエンコーディング）
# # 形状: (batch_size, num_classes, *spatial_dims)
# targets = torch.randint(0, 2, (4, 7, 64, 64, 64)).float()  # 0または1のone-hotエンコーディング

# # 損失の計算
# loss = loss_function(inputs, targets)
# print("loss:", loss, inputs.shape, targets.shape)
# print("os.environ:",os.environ) #["WANDB_API_KEY"])

# encoder_dim = {
#     'resnet18': [64, 64, 128, 256, 512, ],
#     'resnet18d': [64, 64, 128, 256, 512, ],
#     'resnet34d': [64, 64, 128, 256, 512, ],
#     'resnet50d': [64, 256, 512, 1024, 2048, ],
#     'seresnext26d_32x4d': [64, 256, 512, 1024, 2048, ],
#     'convnext_small.fb_in22k': [96, 192, 384, 768],
#     'convnext_tiny.fb_in22k': [96, 192, 384, 768],
#     'convnext_base.fb_in22k': [128, 256, 512, 1024],
#     'tf_efficientnet_b4.ns_jft_in1k':[32, 56, 160, 448],
#     'tf_efficientnet_b5.ns_jft_in1k':[40, 64, 176, 512],
#     'tf_efficientnet_b6.ns_jft_in1k':[40, 72, 200, 576],
#     'tf_efficientnet_b7.ns_jft_in1k':[48, 80, 224, 640],
#     'pvt_v2_b1': [64, 128, 320, 512],
#     'pvt_v2_b2': [64, 128, 320, 512],
#     'pvt_v2_b4': [64, 128, 320, 512],
# }.get("resnet34d", [768])

# print("encoder_dim[-1]", encoder_dim[-1])
# print("encoder_dim[:-1][::-1]+[0]", encoder_dim[:-1][::-1]+[0])



TRAIN_DATA_DIR= "/kaggle/working/input/czii-only-kaggle-data-npy"
TEST_DATA_DIR= "/kaggle/working/input/czii-cryo-et-object-identification"


seed= 22
# SUB_DIMENTION= 16 # 32 # 64 # 96
STRIDE= 16
NUM_SLICE= 32
IMAGE_SIZE= 512 #640 # 元の画像サイズ630だと、unetのデコードとskipコネクション間でサイズ違いで面倒なので、実験の簡素化のためのリサイズ。。
NUM_WORKERS= 4
BATCH_SIZE= 1
PRETRAIN= True


# transform = Compose([
#     Resized(
#         keys=["image", "label"],
#         spatial_size=(IMAGE_SIZE, IMAGE_SIZE),  #(NUM_SLICE, IMAGE_SIZE, IMAGE_SIZE),
#         mode=("trilinear", "nearest")  # 画像は三線形補間、ラベルは最近傍補間
#     ),
#     NormalizeIntensityd(keys="image"),
#     # ToTensord(keys=["image"]),
# ])

class KeepExtraKeysd(Transform):
    def __call__(self, data):
        return data

transform = Compose([
    Resized(
        keys=["image", "label"],
        spatial_size=(IMAGE_SIZE, IMAGE_SIZE),
        mode=("trilinear", "nearest")
    ),
    # NormalizeIntensityd(keys="image"),
    # KeepExtraKeysd()
])

# transform = A.Compose(
#     [
#         A.Resize(height=630, width=630, p=1),
#         # A.RandomResizedCrop(height=cfg.exp.IMAGE_SIZE, width=cfg.exp.IMAGE_SIZE, scale=(0.8, 1.0), p=1.0),
#         # A.HorizontalFlip(p=p),
#         # A.VerticalFlip(p=p),
#         # A.RandomRotate90(p=p),
#         # A.OneOf([
#         #     A.GaussianBlur(blur_limit=(3, 7), p=p),
#         #     A.MedianBlur(blur_limit=5, p=p),
#         # ], p=p),
#         # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=p),
#     ], 
#     p=1.0
# )

def loading_npy( names):
    files = []
    for name in names:
        image = np.load(f"/kaggle/working/input/czii-only-kaggle-data-npy/train_image_{name}.npy")
        label = np.load(f"/kaggle/working/input/czii-only-kaggle-data-npy/train_label_{name}.npy")
        # files.append({"image": image, "label": label})
        files.append({"filename": name, "image": image, "label": label})
    return files



train_names = ['TS_5_4', 'TS_69_2', 'TS_6_6', 'TS_73_6', 'TS_86_3', 'TS_99_9']
valid_names = ['TS_6_4']
train_files = loading_npy( train_names)
valid_files = loading_npy( valid_names)

print(train_files)

# print("train_files:", np.max(train_files[0]["image"]), np.min(train_files[0]["image"]), np.max(train_files[1]["image"]), np.min(train_files[1]["image"]))

# dataset
dataset = SlicedVolumeDataset(train_files, num_slices=NUM_SLICE, stride=STRIDE, transform=transform) #num_slices=64, stride=32)
# dataset = ProteinSegmentationDataset(train_files, transform=transform)

# DataLoader
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

nnn = iter(loader)

for i in range(2):
    dd=next(nnn)
    print(dd.keys())
    # print(dd["image"].shape, dd["label"].shape, dd["start_slice"], dd["volume_idx"])
    print(dd["image"].shape, dd["label"].shape)
    print(torch.max(dd["image"]), torch.min(dd["image"]), torch.max(dd["label"]), torch.min(dd["label"]))



aa=64
bb=32
for z in range(0, 184 - aa + 1, bb):
    print(z, z+aa)


aa=[[1,10],2,3]
print(*aa)