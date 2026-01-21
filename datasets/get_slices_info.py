import h5py
import numpy as np
import open3d as o3d

def get_cloud_slice_info(h5_file_path, cloud_i, slice_j):
    """
    获取指定点云切片的标签、逆变换参数，以及点云数据。
    :param h5_file_path: H5 文件路径
    :param cloud_i: 点云索引
    :param slice_j: 切片索引
    :return: 标签、逆旋转矩阵 R_inverse、逆平移向量 t_inverse，以及点云切片数据
    """
    with h5py.File(h5_file_path, 'r') as f:
        cloud_key = f"cloud_{cloud_i}"
        if cloud_key not in f:
            raise KeyError(f"Cloud index {cloud_i} does not exist in the H5 file.")

        # 获取切片和标签
        group = f[cloud_key]

        # 检查切片是否存在
        slice_key = f"slice_{slice_j}"
        if slice_key not in group:
            raise KeyError(f"Slice {slice_j} does not exist in cloud {cloud_i}.")

        # 获取标签
        label = group.attrs['label']

        # 获取逆变换参数
        R_inverse = group.attrs[f"{slice_key}_inverse_rotation"].reshape(3, 3)
        t_inverse = group.attrs[f"{slice_key}_inverse_translation"]

        # 获取点云切片数据
        point_cloud_data = group[slice_key][...]

        return label, R_inverse, t_inverse, point_cloud_data


def save_slice_as_pcd(point_cloud, output_path):
    """
    将点云数据保存为 PCD 文件。
    :param point_cloud: 点云数据 (Nx3 array)
    :param output_path: 保存的文件路径
    """
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # 保存为 PCD 文件
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved slice as PCD file: {output_path}")


if __name__ == "__main__":
    # 指定 H5 文件路径
    h5_file_path = "/home/kjds512/zhl/Point-NN/data/cut_datasets/modelnet40_voxel_cut.h5"

    while True:
        user_input = input("Enter cloud index (cloud_i) and slice index (slice_j) as 'cloud_i slice_j', or 'q' to quit: ")

        if user_input.lower() == 'q':
            print("Exiting the program.")
            break

        try:
            # 拆分用户输入
            cloud_i, slice_j = map(int, user_input.split())

            # 获取切片信息
            label, R_inverse, t_inverse, point_cloud = get_cloud_slice_info(h5_file_path, cloud_i, slice_j)

            # 输出结果
            print(f"\nCloud {cloud_i}, Slice {slice_j}:")
            print(f"Label: {label}")
            print(f"R_inverse (Rotation Matrix):\n{R_inverse}")
            print(f"t_inverse (Translation Vector):\n{t_inverse}\n")

            # 保存为 PCD 文件
            output_path = f"/home/kjds512/zhl/Point-NN/data/cut_datasets/cloud_slice/cloud_{cloud_i}_slice_{slice_j}.pcd"
            save_slice_as_pcd(point_cloud, output_path)

        except KeyError as e:
            print(f"Error: {e}")
        except ValueError:
            print("Invalid input. Please enter in the format 'cloud_i slice_j' or 'q' to quit.")
