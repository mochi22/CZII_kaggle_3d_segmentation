import torch
from torch.utils.data import Dataset

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
            # vol_slice = self.transform(vol_slice)
            # label_slice = self.transform(label_slice)
            return data
        
        return {
            'image': torch.from_numpy(vol_slice).float(),
            'label': torch.from_numpy(label_slice).long(),
            'start_slice': start_slice,
            'volume_idx': data_idx
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