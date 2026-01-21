import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_cluster import knn_graph


def adjust_normals_direction(normals, reference_vector):
    dot_product = torch.sum(normals * reference_vector, dim=2, keepdim=True)
    mask = dot_product < 0
    mask = mask.expand_as(normals)
    normals[mask] = -normals[mask]
    return normals


def compute_line_pca(point_cloud, k=30):
    normals = compute_norm_pca(point_cloud, k).permute(0, 2, 1)
    cross_product = torch.cross(point_cloud, normals, dim=1)
    dot_product = torch.sum(point_cloud * normals, dim=1, keepdim=True)
    embedding_element = torch.cat((point_cloud, cross_product, dot_product), dim=1)

    return embedding_element

def compute_norm_rpi(point_cloud, k=50):
    device = point_cloud.device
    batch_size, _, N_points = point_cloud.shape
    x = point_cloud.permute(0, 2, 1).reshape(-1, 3)  # Reshape to (batch_size * N_points, 3)
    batch = torch.arange(batch_size, device=device).repeat_interleave(N_points)

    edge_index = knn_graph(x, k, batch, loop=False)
    start, end = edge_index
    centered = x[end] - x[start].mean(dim=0)
    cov_matrices = torch.einsum('ij,ik->ijk', centered, centered)
    cov_matrices = cov_matrices.view(batch_size, N_points, k, 3, 3).mean(dim=2)

    _, normals_batch = randomized_power_iteration(cov_matrices, num_iters=90)
    normals_batch = F.normalize(normals_batch, p=2, dim=1)
    normals = normals_batch.reshape(batch_size, 3, N_points)

    return normals

def compute_line_rpi(point_cloud, k=50):
    normals = compute_norm_rpi(point_cloud, k)
    cross_product = torch.cross(point_cloud, normals, dim=1)
    dot_product = torch.sum(point_cloud * normals, dim=1, keepdim=True)
    embedding_element = torch.cat((point_cloud, cross_product, dot_product), dim=1)
    return embedding_element

def compute_line_normals(point_cloud, normals):
    cross_product = torch.cross(point_cloud, normals, dim=1)
    dot_product = torch.sum(point_cloud * normals, dim=1, keepdim=True)
    embedding_element = torch.cat((point_cloud, cross_product, dot_product), dim=1)
    return embedding_element

def compute_line_normals_nc(point_cloud, normals):
    dot_product = torch.sum(point_cloud * normals, dim=1, keepdim=True)
    embedding_element = torch.cat((point_cloud, normals, dot_product), dim=1)
    return embedding_element

def compute_surface_normal(point_cloud, normals):
    cross_product = torch.cross(point_cloud, normals, dim=1)
    embedding_element = torch.cat((normals, cross_product), dim=1)
    return embedding_element

def compute_oriented_tangent_plane(point_cloud, normals):
    dot_product = torch.sum(point_cloud * normals, dim=1, keepdim=True)
    embedding_element = torch.cat((normals, dot_product), dim=1)
    return embedding_element

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

