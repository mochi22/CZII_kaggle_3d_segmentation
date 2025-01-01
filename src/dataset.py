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

    