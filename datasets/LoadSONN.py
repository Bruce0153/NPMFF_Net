"""
ScanObjectNN download: http://103.24.77.34/scanobjectnn/h5_files.zip
"""

import collections
import h5py
import numpy as np
import os
from scipy.linalg import expm, norm
import torch
from torch.utils.data import Dataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    # 只平移前3个维度 (x, y, z)，不平移法向量 (nx, ny, nz)
    translated_xyz = np.add(np.multiply(pointcloud[:, :3], xyz1), xyz2).astype('float32')
    translated_pointcloud = np.hstack((translated_xyz, pointcloud[:, 3:])).astype('float32')
    return translated_pointcloud


class PointsToTensor(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            if not torch.is_tensor(data[key]):
                data[key] = torch.from_numpy(np.array(data[key]))
        return data


class PointCloudScaling(object):
    def __init__(self,
                 scale=[2. / 3, 3. / 2],
                 anisotropic=True,
                 scale_xyz=[True, True, True],
                 symmetries=[0, 0, 0],  # mirror scaling, x --> -x
                 **kwargs):
        self.scale_min, self.scale_max = np.array(scale).astype(np.float32)
        self.anisotropic = anisotropic
        self.scale_xyz = scale_xyz
        self.symmetries = torch.from_numpy(np.array(symmetries))

    def __call__(self, data):
        device = data['pos'].device if hasattr(data, 'keys') else data.device
        scale = torch.rand(3 if self.anisotropic else 1, dtype=torch.float32, device=device) * (
                self.scale_max - self.scale_min) + self.scale_min
        symmetries = torch.round(torch.rand(3, device=device)) * 2 - 1
        self.symmetries = self.symmetries.to(device)
        symmetries = symmetries * self.symmetries + (1 - self.symmetries)
        scale *= symmetries
        for i, s in enumerate(self.scale_xyz):
            if not s: scale[i] = 1

        # 缩放点云的坐标 (pos)
        if hasattr(data, 'keys'):
            data['pos'][:, :3] *= scale  # 只缩放前3个维度 (x, y, z)

            # 如果法向量存在, 需要对法向量做处理
            if 'pos' in data:
                data['pos'][:, 3:] *= symmetries  # 法向量的对称处理
                data['pos'][:, 3:] = torch.nn.functional.normalize(data['pos'][:, 3:], dim=-1)  # 保持法向量归一化
        else:
            data[:, :3] *= scale
            data[:, 3:] *= symmetries
            data[:, 3:] = torch.nn.functional.normalize(data[:, 3:], dim=-1)
        return data


class PointCloudCenterAndNormalize(object):
    def __init__(self, centering=True,
                 normalize=True,
                 gravity_dim=2,
                 append_xyz=False,
                 **kwargs):
        self.centering = centering
        self.normalize = normalize
        self.gravity_dim = gravity_dim
        self.append_xyz = append_xyz

    def __call__(self, data):
        if hasattr(data, 'keys'):
            if self.append_xyz:
                data['heights'] = data['pos'][:, :3] - torch.min(data['pos'][:, :3])
            else:
                height = data['pos'][:, self.gravity_dim:self.gravity_dim + 1]
                data['heights'] = height - torch.min(height)

            if self.centering:
                data['pos'][:, :3] = data['pos'][:, :3] - torch.mean(data['pos'][:, :3], axis=0, keepdims=True)

            if self.normalize:
                m = torch.max(torch.sqrt(torch.sum(data['pos'][:, :3] ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True)[0]
                data['pos'][:, :3] = data['pos'][:, :3] / m
        else:
            if self.centering:
                data[:, :3] = data[:, :3] - torch.mean(data[:, :3], axis=-1, keepdims=True)
            if self.normalize:
                m = torch.max(torch.sqrt(torch.sum(data[:, :3] ** 2, axis=-1, keepdims=True)), axis=0, keepdims=True)[0]
                data[:, :3] = data[:, :3] / m
        return data


class PointCloudRotation(object):
    def __init__(self, angle=[0, 0, 0], **kwargs):
        self.angle = np.array(angle) * np.pi

    @staticmethod
    def M(axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, data):
        if hasattr(data, 'keys'):
            device = data['pos'].device
        else:
            device = data.device

        if isinstance(self.angle, collections.Iterable):
            rot_mats = []
            for axis_ind, rot_bound in enumerate(self.angle):
                theta = 0
                axis = np.zeros(3)
                axis[axis_ind] = 1
                if rot_bound is not None:
                    theta = np.random.uniform(-rot_bound, rot_bound)
                rot_mats.append(self.M(axis, theta))
            # Use random order
            np.random.shuffle(rot_mats)
            rot_mat = torch.tensor(rot_mats[0] @ rot_mats[1] @ rot_mats[2], dtype=torch.float32, device=device)
        else:
            raise ValueError()
        if hasattr(data, 'keys'):
            data['pos'][:, :3] = data['pos'][:, :3] @ rot_mat.T  # 旋转坐标
            if 'pos' in data:
                data['pos'][:, 3:] = data['pos'][:, 3:] @ rot_mat.T  # 旋转法向量
        else:
            data[:, :3] = data[:, :3] @ rot_mat.T
            data[:, 3:] = data[:, 3:] @ rot_mat.T
        return data


def load_scanobjectnn_data(variant, split, partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    all_data = []
    all_label = []

    if variant == 'OBJ-BG':
        DATA_DIR = os.path.dirname(BASE_DIR) + '/data/OBJ_BG_split_1_h5/'
        h5_name = DATA_DIR + partition + '_objectdataset.h5'
    elif variant == 'PB_T25':
        DATA_DIR = os.path.dirname(BASE_DIR) + '/data/PB_T25_main_split_h5/'
        h5_name = DATA_DIR + partition + '_objectdataset_augmented25_norot.h5'
    elif variant == 'PB_T25_R':
        DATA_DIR = os.path.dirname(BASE_DIR) + '/data/PB_T25_R_main_split_h5/'
        h5_name = DATA_DIR + partition + '_objectdataset_augmented25rot.h5'
    elif variant == 'PB_T50_R':
        DATA_DIR = os.path.dirname(BASE_DIR) + '/data/PB_T50_R_main_split_h5/'
        h5_name = DATA_DIR + partition + '_objectdataset_augmentedrot.h5'
    elif variant == 'PB_T50_RS':
        DATA_DIR = os.path.dirname(BASE_DIR) + '/data/PB_T50_RS_split_1_h5/'
        h5_name = DATA_DIR + partition + '_objectdataset_augmentedrot_scale75.h5'
    else:
        h5_name = variant
        print(f"Warning: wrong variant: {variant}.")
    print('Load scan h5 file: ', h5_name)

    f = h5py.File(h5_name, mode="r")

    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')

    f.close()

    all_data.append(data)
    all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label

class ScanObjectNN(Dataset):
    def __init__(self, num_points=2048, variant='OBJ-BG', split=1, partition='training', transform=None):
        self.data, self.label = load_scanobjectnn_data(variant, split, partition)
        self.num_points = num_points
        self.partition = partition
        self.transform = transform

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]

        label = self.label[item]
        if self.partition == 'training':
            #pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    train_loader = DataLoader(ScanObjectNN(partition='training', num_points=1024), num_workers=8,
                              batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(ScanObjectNN(partition='test', num_points=1024), num_workers=8,
                              batch_size=32, shuffle=False, drop_last=True)

    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}  | data shape: {data.shape}| ;lable shape: {label.shape}")
