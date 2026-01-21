import os
import numpy as np
import h5py
import torch
from pointnet2_ops import pointnet2_utils

# 文件路径
data_dir = './PB_T25'
output_dir = './data/PB_T25_split_1_h5'
num_points = 2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检查是否有可用的 GPU

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points: indexed points data, [B, S, C]
    """
    B = points.shape[0]
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(B, 1).repeat(1, idx.shape[1])
    return points[batch_indices, idx, :]


# 读取split文件
def load_split_file(split_file_path):
    with open(split_file_path, 'r') as f:
        lines = f.readlines()
    split_data = {}
    for line in lines:
        parts = line.strip().split()
        file_name = parts[0].rsplit('.', 1)[0]  # 去掉.bin后缀，如 'scene0065_00_00008'
        class_id = int(parts[1])
        is_test = len(parts) > 2 and parts[2] == 't'
        split_data[file_name] = (class_id, is_test)
    return split_data


def load_bin_file(file_path):
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)

    num_points_in_file = int(data[0])
    points = data[1:].reshape((num_points_in_file, 11))

    points_tensor = torch.from_numpy(points[:, :6]).to(torch.float32).cuda().unsqueeze(0)  # [1, N, 6]
    points_search = points_tensor[:, :, :3].contiguous()  # 只用 xyz 采样
    indices = pointnet2_utils.furthest_point_sample(points_search, num_points).long()
    points_tensor = index_points(points_tensor, indices)  # [1, num_points, 6]

    points = points_tensor.squeeze(0).cpu().numpy()

    return points

def convert_dataset_to_h5(data_dir, output_dir, split_file_path):
    split_data = load_split_file(split_file_path)

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.bin'):

                # 检查 file_base 是否在 split_data 中
                file_base = file.rsplit('_', 1)[0]  # 去掉最后的后缀部分，如 'scene0065_00_00008'
                if file_base in split_data:
                    class_id, is_test = split_data[file_base]

                    file_path = os.path.join(root, file)
                    points = load_bin_file(file_path)

                    if is_test:
                        test_data.append(points)
                        test_labels.append(class_id)
                    else:
                        train_data.append(points)
                        train_labels.append(class_id)

                    print(f"Processed {file} to {'test' if is_test else 'train'} set.")
                else:
                    print(f"Warning: {file} not found in split file.")

    train_data = np.array(train_data, dtype=np.float32)  # 维度应为 (点云数, num_points, 6)
    train_labels = np.array(train_labels, dtype=np.int32)
    test_data = np.array(test_data, dtype=np.float32)  # 维度应为 (点云数, num_points, 6)
    test_labels = np.array(test_labels, dtype=np.int32)

    # 保存到 HDF5 文件
    save_to_h5(os.path.join(output_dir, 'training_objectdataset_augmented25_norot.h5'), train_data, train_labels)
    save_to_h5(os.path.join(output_dir, 'test_objectdataset_augmented25_norot.h5'), test_data, test_labels)

    print("Training and testing datasets have been combined and saved.")


def save_to_h5(h5_file_path, data, labels):
    with h5py.File(h5_file_path, 'w') as h5f:
        h5f.create_dataset('data', data=data, dtype='float32')  # 保存3D数据
        h5f.create_dataset('label', data=labels, dtype='int32')


# 开始转换
split_file_path = './split3/PB_T50_RS/split1.txt'
convert_dataset_to_h5(data_dir, output_dir, split_file_path)
