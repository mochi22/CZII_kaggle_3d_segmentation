import numpy as np
from src.dataset import SlicedVolumeDataset
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





TRAIN_DATA_DIR= "/kaggle/working/input/czii-only-kaggle-data-npy"
TEST_DATA_DIR= "/kaggle/working/input/czii-cryo-et-object-identification"

def loading_npy(names):
    files = []
    for name in names:
        image = np.load(f"{TRAIN_DATA_DIR}/train_image_{name}.npy")
        label = np.load(f"{TRAIN_DATA_DIR}/train_label_{name}.npy")
        files.append({"image": image, "label": label})
    return files

# loading data
train_names = ['TS_5_4', 'TS_69_2', 'TS_6_6', 'TS_73_6', 'TS_86_3', 'TS_99_9']
valid_names = ['TS_6_4']
train_files = loading_npy(train_names)
valid_files = loading_npy(valid_names)

print(len(train_files))
print(train_files[0]["image"].shape, train_files[0]["label"].shape)


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
    NormalizeIntensityd(keys="image"),
    KeepExtraKeysd()
])

def loading_npy( names):
    files = []
    for name in names:
        image = np.load(f"/kaggle/working/input/czii-only-kaggle-data-npy/train_image_{name}.npy")
        label = np.load(f"/kaggle/working/input/czii-only-kaggle-data-npy/train_label_{name}.npy")
        files.append({"image": image, "label": label})
    return files

train_names = ['TS_5_4', 'TS_69_2', 'TS_6_6', 'TS_73_6', 'TS_86_3', 'TS_99_9']
valid_names = ['TS_6_4']
train_files = loading_npy( train_names)
valid_files = loading_npy( valid_names)

# dataset
dataset = SlicedVolumeDataset(train_files, num_slices=NUM_SLICE, stride=STRIDE, transform=None) #num_slices=64, stride=32)

# DataLoader
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

nnn = iter(loader)

for i in range(20):
    dd=next(nnn)
    print(dd.keys())
    print(dd["image"].shape, dd["label"].shape, dd["start_slice"], dd["volume_idx"])



aa=64
bb=32
for z in range(0, 184 - aa + 1, bb):
    print(z, z+aa)
