import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import warnings
warnings.simplefilter('ignore')

class SlicedVolumeDataset(Dataset):
    def __init__(self, data_files, num_slices=64, stride=32, transform=None):
        self.data_files = data_files
        self.num_slices = num_slices
        self.stride = stride
        self.transform = transform
        
        self.slices = self._prepare_slices()

    def _prepare_slices(self):
        slices = []
        for idx, data in enumerate(self.data_files):
            volume = data['image']
            D, H, W = volume.shape
            for z in range(0, D - self.num_slices + 1, self.stride):
                slices.append((idx, z))
        return slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        data_idx, start_slice = self.slices[idx]
        
        # ボリュームとラベルの取得
        volume = self.data_files[data_idx]['image']
        label = self.data_files[data_idx]['label']

        # 正規化
        max_val = volume.max()
        min_val = volume.min()
        volume = (volume - min_val) / (max_val - min_val)
        
        # スライスの抽出
        vol_slice = volume[start_slice:start_slice + self.num_slices]
        label_slice = label[start_slice:start_slice + self.num_slices]
        
        # 必要に応じて変換を適用
        if self.transform:
            data = {
                'image': vol_slice,
                'label': label_slice
            }
            data = self.transform(data)
            vol_slice = data["image"] # torch.Size([32, 512, 512]) #self.transform(vol_slice)
            label_slice = data["label"] # torch.Size([32, 512, 512]) #self.transform(label_slice)

        # ラベルをTensorに変換
        label_tensor = label_slice.long() #torch.from_numpy(label_slice).long()
        
        # one-hotエンコーディングに変換
        num_classes = 7  # 0から6までのクラス
        one_hot_label = F.one_hot(label_tensor, num_classes=num_classes)
        # (D, H, W, C) -> (C, D, H, W)
        one_hot_label = one_hot_label.permute(3, 0, 1, 2)

        return {
            'image': vol_slice.float(), #torch.from_numpy(vol_slice).float(),
            'label': one_hot_label,
            'start_slice': start_slice,
            'volume_idx': data_idx
        }


class ProteinSegmentationDataset(Dataset):
    def __init__(self, data_list, transform=None, inference_mode=False):
        self.data_list = data_list
        self.transform = transform
        self.inference_mode=inference_mode

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        # image = torch.from_numpy(data['image']).float()
        # label = torch.from_numpy(data['label']).long()
        image = data['image']
        label = data['label']

        # 画像の正規化
        max_val = image.max()
        min_val = image.min()
        image = (image - min_val) / (max_val - min_val)
        
        # if self.transform:
        #     image = self.transform(image)
        
        # if self.transform:
        #     transformed = self.transform(volume=image, mask3d=label)
        #     image = transformed["volume"]
        #     label = transformed["mask3d"]

        # print(image.shape, label.shape)

        if self.transform:
            transformed_slices = []
            transformed_labels = [] if not self.inference_mode else None
            
            for i in range(image.shape[0]):  # Iterate over depth
                if self.inference_mode:
                    transformed = self.transform(image=image[i])
                    transformed_slices.append(transformed['image'])
                else:
                    transformed = self.transform(image=image[i], mask=label[i])
                    transformed_slices.append(transformed['image'])
                    transformed_labels.append(transformed['mask'])
            
            image = np.stack(transformed_slices, axis=0)
            if not self.inference_mode:
                label = np.stack(transformed_labels, axis=0)


        # チャンネル次元を追加 (184, 630, 630) -> (1, 184, 630, 630)
        # image = image.unsqueeze(0)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        print(image.shape, label.shape)

        return {
            'filename': data['filename'],
            'image': torch.from_numpy(image).float(),
            'label': torch.from_numpy(label).long() if not self.inference_mode else None
        }


# import pandas as pd
# import numpy as np
# import json
# import zarr

# PARTICLE= [
#     {
#         "name": "apo-ferritin",
#         "difficulty": 'easy',
#         "pdb_id": "4V1W",
#         "label": 1,
#         "color": [0, 255, 0, 0],
#         "radius": 60,
#         "map_threshold": 0.0418
#     },
#     {
#         "name": "beta-amylase",
#         "difficulty": 'ignore',
#         "pdb_id": "1FA2",
#         "label": 2,
#         "color": [0, 0, 255, 255],
#         "radius": 65,
#         "map_threshold": 0.035
#     },
#     {
#         "name": "beta-galactosidase",
#         "difficulty": 'hard',
#         "pdb_id": "6X1Q",
#         "label": 3,
#         "color": [0, 255, 0, 255],
#         "radius": 90,
#         "map_threshold": 0.0578
#     },
#     {
#         "name": "ribosome",
#         "difficulty": 'easy',
#         "pdb_id": "6EK0",
#         "label": 4,
#         "color": [0, 0, 255, 0],
#         "radius": 150,
#         "map_threshold": 0.0374
#     },
#     {
#         "name": "thyroglobulin",
#         "difficulty": 'hard',
#         "pdb_id": "6SCJ",
#         "label": 5,
#         "color": [0, 255, 255, 0],
#         "radius": 130,
#         "map_threshold": 0.0278
#     },
#     {
#         "name": "virus-like-particle",
#         "difficulty": 'easy',
#         "pdb_id": "6N4V",
#         "label": 6,
#         "color": [0, 0, 0, 255],
#         "radius": 135,
#         "map_threshold": 0.201
#     }
# ]

# PARTICLE_COLOR=[[0,0,0]]+[
#     PARTICLE[i]['color'][1:] for i in range(6)
# ]
# PARTICLE_NAME=['none']+[
#     PARTICLE[i]['name'] for i in range(6)
# ]

# '''
# (184, 630, 630)  
# (92, 315, 315)  
# (46, 158, 158)  
# '''

# # def read_one_data(id, static_dir):
# #     zarr_dir = f'{static_dir}/{id}/VoxelSpacing10.000'
# #     zarr_file = f'{zarr_dir}/denoised.zarr'
# #     zarr_data = zarr.open(zarr_file, mode='r')
# #     volume = zarr_data[0][:]
# #     max = volume.max()
# #     min = volume.min()
# #     volume = (volume - min) / (max - min)
# #     volume = volume.astype(np.float16)
# #     return volume


# # def read_one_truth(id, overlay_dir):
# #     location={}

# #     json_dir = f'{overlay_dir}/{id}/Picks'
# #     for p in PARTICLE_NAME[1:]:
# #         json_file = f'{json_dir}/{p}.json'

# #         with open(json_file, 'r') as f:
# #             json_data = json.load(f)

# #         num_point = len(json_data['points'])
# #         loc = np.array([list(json_data['points'][i]['location'].values()) for i in range(num_point)])
# #         location[p] = loc

# #     return location

# class SlicedVolumeDataset(Dataset):
#     def __init__(self, ids, static_dir, overlay_dir, num_slices=64, stride=32, transform=None, is_train=True):
#         self.ids = ids
#         self.static_dir = static_dir
#         self.overlay_dir = overlay_dir
#         self.num_slices = num_slices
#         self.stride = stride
#         self.transform = transform
#         self.is_train=is_train

#         self.slices = self._prepare_slices()

#     def _prepare_slices(self):
#         slices = []
#         for idx, id in enumerate(self.ids):
#             volume = self.read_one_data(id)
#             D, H, W = volume.shape
#             for z in range(0, D - self.num_slices + 1, self.stride):
#                 slices.append((idx, z))
#         return slices

#     def read_one_data(self, id):
#         zarr_dir = f'{self.static_dir}/{id}/VoxelSpacing10.000'
#         zarr_file = f'{zarr_dir}/denoised.zarr'
#         zarr_data = zarr.open(zarr_file, mode='r')
#         volume = zarr_data[0][:]
#         max_val = volume.max()
#         min_val = volume.min()
#         volume = (volume - min_val) / (max_val - min_val)
#         volume = volume.astype(np.float16)
#         return volume

#     def read_one_truth(self, id):
#         location = {}
#         json_dir = f'{self.overlay_dir}/{id}/Picks'
#         for p in PARTICLE_NAME[1:]:
#             json_file = f'{json_dir}/{p}.json'
#             with open(json_file, 'r') as f:
#                 json_data = json.load(f)
#             num_point = len(json_data['points'])
#             loc = np.array([list(json_data['points'][i]['location'].values()) for i in range(num_point)])
#             location[p] = loc
#         return location

#     def __len__(self):
#         return len(self.slices)

#     def __getitem__(self, idx):
#         data_idx, start_slice = self.slices[idx]
#         id = self.ids[data_idx]
        
#         # ボリュームとラベルの取得
#         volume = self.read_one_data(id)
#         if self.is_train:
#             truth = self.read_one_truth(id)
        
#         print("volume", volume.shape)
#         # print("truth:", truth)
        
#         # スライスの抽出
#         vol_slice = volume[start_slice:start_slice + self.num_slices]
        
#         # ラベルの生成（この部分は実際のデータ形式に合わせて調整が必要です）
#         if self.is_train:
#             label_slice = np.zeros((self.num_slices, volume.shape[1], volume.shape[2]), dtype=np.int64)
#             for p_name, locations in truth.items():
#                 print("PARTICLE_NAME.index(p_name)", PARTICLE_NAME.index(p_name))
#                 for loc in locations:
#                     z, y, x = loc
                    
#                     if start_slice <= z < start_slice + self.num_slices:
#                         label_slice[int(z - start_slice), int(y), int(x)] = PARTICLE_NAME.index(p_name)
        
#         # パディング
#         H, W = volume.shape[1:]
#         pad_vol_slice = np.pad(vol_slice, [[0, 0], [0, 640 - H], [0, 640 - W]], mode='constant', constant_values=0)
#         if self.is_train:
#             pad_label_slice = np.pad(label_slice, [[0, 0], [0, 640 - H], [0, 640 - W]], mode='constant', constant_values=0)
        
#         # 必要に応じて変換を適用
#         if self.transform:
#             if self.is_train:
#                 data = {
#                     'image': pad_vol_slice,
#                     'label': pad_label_slice
#                 }
#             else:
#                 data = {
#                     'image': pad_vol_slice,
#                 }
#             data = self.transform(data)
#             return data
#         if self.is_train:
#             return {
#                 'image': torch.from_numpy(pad_vol_slice).float(),
#                 'label': torch.from_numpy(pad_label_slice).long(),
#                 'start_slice': start_slice,
#                 'volume_idx': data_idx,
#                 'id': id
#             }
#         else:
#             return {
#                 'image': torch.from_numpy(pad_vol_slice).float(),
#                 # 'label': torch.from_numpy(pad_label_slice).long(),
#                 'start_slice': start_slice,
#                 'volume_idx': data_idx,
#                 'id': id
#             }


## 2xx  3dCNN用のやつ
from scipy.ndimage import zoom
# class TomogramDataset(Dataset):
#     def __init__(self, files, patch_size=64, overlap=0.75, augment=True):
#         """
#         Args:
#             files: リストof辞書 [{"filename": name, "image": image, "label": label}, ...]
#             patch_size: パッチのサイズ
#             overlap: オーバーラップの割合
#             augment: データ拡張を行うかどうか
#         """
#         self.files = files
#         self.patch_size = patch_size
#         self.overlap = overlap
#         self.augment = augment
#         self.patches = self._create_patches()

#     def _create_patches(self):
#         patches = []
#         step_size = int(self.patch_size * (1 - self.overlap))
        
#         for file_data in self.files:
#             tomogram = file_data["image"]
#             label = file_data["label"]
#             filename = file_data["filename"]
            
#             depth, height, width = tomogram.shape
            
#             # 各次元でのパッチ数を計算
#             z_steps = max(1, int(np.ceil((depth - self.patch_size) / step_size)) + 1)
#             y_steps = max(1, int(np.ceil((height - self.patch_size) / step_size)) + 1)
#             x_steps = max(1, int(np.ceil((width - self.patch_size) / step_size)) + 1)
            
#             for z_idx in range(z_steps):
#                 z_start = min(z_idx * step_size, depth - self.patch_size)
                
#                 for y_idx in range(y_steps):
#                     y_start = min(y_idx * step_size, height - self.patch_size)
                    
#                     for x_idx in range(x_steps):
#                         x_start = min(x_idx * step_size, width - self.patch_size)
                        
#                         # パッチの切り出し
#                         patch = tomogram[
#                             z_start:z_start + self.patch_size,
#                             y_start:y_start + self.patch_size,
#                             x_start:x_start + self.patch_size
#                         ]
                        
#                         label_patch = label[
#                             z_start:z_start + self.patch_size,
#                             y_start:y_start + self.patch_size,
#                             x_start:x_start + self.patch_size
#                         ]
                        
#                         # パッチのサイズが正しいことを確認
#                         if patch.shape == (self.patch_size, self.patch_size, self.patch_size):
#                             patches.append({
#                                 'data': patch,
#                                 'label': label_patch,
#                                 'position': (z_start, y_start, x_start),
#                                 'filename': filename
#                             })
        
#         return patches

#     def __len__(self):
#         return len(self.patches)

#     def __getitem__(self, idx):
#         patch_info = self.patches[idx]
#         patch = patch_info['data']
#         label_patch = patch_info['label']
        
#         if self.augment and np.random.rand() > 0.5:
#             patch = np.flip(patch, axis=2)  # 水平フリップ
#             label_patch = np.flip(label_patch, axis=2)
        
#         # チャンネル次元を追加し、適切なデータ型に変換
#         patch_tensor = torch.from_numpy(patch.copy()).float().unsqueeze(0)
#         label_tensor = torch.from_numpy(label_patch.copy()).long()
        
#         return {
#             'data': patch_tensor,
#             'label': label_tensor,
#             'position': patch_info['position'],
#             'filename': patch_info['filename']
#         }
import ast
from typing import Tuple, Any
def parse_tuple_str(tuple_str: str) -> Tuple[Any, ...]:
    """文字列形式のtupleを実際のtupleに変換する"""
    try:
        return ast.literal_eval(tuple_str)
    except (ValueError, SyntaxError):
        raise ValueError(f"Invalid tuple string format: {tuple_str}")

class TomogramDataset(Dataset):
    def __init__(self, files, patch_size=64, overlap=0.75, target_size=(184, 128, 128), augment=True):
        """
        Args:
            files: リストof辞書 [{"filename": name, "image": image, "label": label}, ...]
            patch_size: パッチのサイズ
            overlap: オーバーラップの割合
            target_size: リサイズ後のサイズ (D, H, W)
            augment: データ拡張を行うかどうか
        """
        self.files = files
        self.patch_size = patch_size
        self.overlap = overlap
        self.target_size = parse_tuple_str(target_size) if isinstance(target_size, str) else target_size # Hydraから読み込まれる際は文字列になるのでparse
        self.augment = augment
        self.patches = self._create_patches()

    def _resize_volume(self, volume, is_label=False):
        """ボリュームをtarget_sizeにリサイズする"""
        current_size = volume.shape
        factors = (
            self.target_size[0] / current_size[0],
            self.target_size[1] / current_size[1],
            self.target_size[2] / current_size[2]
        )
        
        if is_label:
            # ラベルの場合は最近傍補間を使用
            return zoom(volume, factors, order=0)
        else:
            # 画像の場合は3次スプライン補間を使用
            return zoom(volume, factors, order=3)

    def _create_patches(self):
        patches = []
        step_size = int(self.patch_size * (1 - self.overlap))
        
        for file_data in self.files:
            # データのリサイズ
            tomogram = self._resize_volume(file_data["image"])
            label = self._resize_volume(file_data["label"], is_label=True)
            filename = file_data["filename"]
            
            depth, height, width = tomogram.shape
            
            # 各次元でのパッチ数を計算
            z_steps = max(1, int(np.ceil((depth - self.patch_size) / step_size)) + 1)
            y_steps = max(1, int(np.ceil((height - self.patch_size) / step_size)) + 1)
            x_steps = max(1, int(np.ceil((width - self.patch_size) / step_size)) + 1)
            
            for z_idx in range(z_steps):
                z_start = min(z_idx * step_size, depth - self.patch_size)
                
                for y_idx in range(y_steps):
                    y_start = min(y_idx * step_size, height - self.patch_size)
                    
                    for x_idx in range(x_steps):
                        x_start = min(x_idx * step_size, width - self.patch_size)
                        
                        # パッチの切り出し
                        patch = tomogram[
                            z_start:z_start + self.patch_size,
                            y_start:y_start + self.patch_size,
                            x_start:x_start + self.patch_size
                        ]
                        
                        label_patch = label[
                            z_start:z_start + self.patch_size,
                            y_start:y_start + self.patch_size,
                            x_start:x_start + self.patch_size
                        ]
                        
                        # パッチのサイズが正しいことを確認
                        if patch.shape == (self.patch_size, self.patch_size, self.patch_size):
                            # 原画像でのパッチ位置を計算
                            original_z = int(z_start * (630 / self.target_size[1]))
                            original_y = int(y_start * (630 / self.target_size[1]))
                            original_x = int(x_start * (630 / self.target_size[2]))
                            
                            patches.append({
                                'data': patch,
                                'label': label_patch,
                                'position': (z_start, y_start, x_start),
                                'original_position': (original_z, original_y, original_x),
                                'filename': filename
                            })
        
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        patch = patch_info['data']
        label_patch = patch_info['label']
        
        if self.augment:
            # データ拡張
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=2)  # 水平フリップ
                label_patch = np.flip(label_patch, axis=2)
            
            # 必要に応じて他のデータ拡張を追加
            
        # チャンネル次元を追加し、適切なデータ型に変換
        patch_tensor = torch.from_numpy(patch.copy()).float().unsqueeze(0)
        label_tensor = torch.from_numpy(label_patch.copy()).long()
        
        return {
            'data': patch_tensor,
            'label': label_tensor,
            'position': patch_info['position'],
            'original_position': patch_info['original_position'],
            'filename': patch_info['filename']
        }
