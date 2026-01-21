# Non-Parametric Networks for 3D Point Cloud Part Segmentation
import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils

from .model_utils import *

def normalize(x, ref):
    """
    对输入 x 进行标准化，基于 ref 的均值和标准差
    """
    mean = ref.unsqueeze(dim=-2)
    std = torch.std(x - mean)
    return (x - mean) / (std + 1e-5)

# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long() 
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)

        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)

        return lc_xyz, lc_x, knn_xyz, knn_x

class Line_block(nn.Module):
    def __init__(self, p, use_anp=True):
        super().__init__()
        self.use_anp = use_anp

        if use_anp:
            self.k_anp = K_ANP(initial_p=p, dim=-1)

        self.Fourier_embedding_knn = None  # 延迟初始化，以根据输入维度适应

    def forward(self, lc_x, knn_x):

        # Compute local cross product and dot product
        B, G, K, N = knn_x.shape

        lc_normal = lc_x[:, :, 3:6].unsqueeze(2).repeat(1, 1, K, 1)

        lc_line = lc_x
        lc_line_norm = lc_line.norm(dim=2, keepdim=True)
        lc_line = (lc_line / lc_line_norm).transpose(1, 2)

        # Compute KNN cross product and dot product
        knn_product = knn_x[:, :, :, 3:6]
        knn_cross_product = torch.cross(knn_product, lc_normal, dim=3)
        knn_dot_product = torch.sum(knn_product * lc_normal, dim=3, keepdim=True)

        knn_line = torch.cat((knn_product, knn_cross_product, knn_dot_product), dim=3).permute(0, 3, 1, 2)

        self.Fourier_embedding_knn = FourierFeatureMapping(knn_line.shape[1], lc_line.shape[1], 10)
        knn_line = self.Fourier_embedding_knn(knn_line)

        if self.use_anp:
            k_anp_x = self.k_anp(knn_line)
            knn_line = k_anp_x.squeeze() + knn_line.mean(-1)
        else:
            knn_line = knn_line.max(-1)[0] + knn_line.mean(-1)

        knn_line_norm = knn_line.norm(dim=-1, keepdim=True)
        knn_line = knn_line / knn_line_norm

        line_element = knn_line - lc_line
        line_element = torch.cat([line_element, lc_line], dim=1)

        return line_element

class NonparametricAttentionPooling(nn.Module):
    def __init__(self, bandwidth=0.5):
        super().__init__()
        self.bandwidth = bandwidth
        self.out_transform = None

    def forward(self, out_f, x):

        if self.out_transform is None or self.out_transform[0].num_features != out_f:
            self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_f),
                nn.GELU()
            ).to(x.device)

        norms = torch.cdist(x.transpose(1, 2), x.transpose(1, 2))  # shape: [B, N, N]

        weights_gaussian_1 = torch.exp(-0.5 * (norms / self.bandwidth) ** 2)
        weights_gaussian_2 = torch.exp(-0.5 * (norms / (self.bandwidth * 2)) ** 2)
        weights_gaussian_3 = torch.exp(-0.5 * (norms / (self.bandwidth / 2)) ** 2)

        gaussian_weights = torch.tensor([0.5, 0.3, 0.2])

        weights = (gaussian_weights[0] * weights_gaussian_1 +
                   gaussian_weights[1] * weights_gaussian_2 +
                   gaussian_weights[2] * weights_gaussian_3)


        weights /= weights.sum(dim=2, keepdim=True) + 1e-8  # shape: [B, N, N]

        new_features = torch.bmm(weights, x.transpose(1, 2))  # shape: [B, N, F]
        new_features = new_features.transpose(1, 2)  # shape: [B, F, N]
        new_features = self.out_transform(new_features)

        return new_features

class NonparametricCrossAttentionPooling(nn.Module):
    def __init__(self, bandwidth):
        super().__init__()
        self.bandwidth = bandwidth
        self.out_transform = None

    def forward(self, query, key_value):
        device = query.device
        batch_size, query_length, query_features = query.size()
        _, key_value_length, key_value_features = key_value.size()

        if self.out_transform is None or self.out_transform[0].num_features != query_features:
            self.out_transform = nn.Sequential(
                nn.BatchNorm1d(query_features),
                nn.GELU()
            ).to(device)

        norms = torch.cdist(query, key_value)  # shape: [B, Nq, Nk]

        # Apply Gaussian kernels
        weights_gaussian_1 = torch.exp(-0.5 * (norms / self.bandwidth) ** 2)
        weights_gaussian_2 = torch.exp(-0.5 * (norms / (self.bandwidth * 2)) ** 2)
        weights_gaussian_3 = torch.exp(-0.5 * (norms / (self.bandwidth / 2)) ** 2)

        gaussian_weights = torch.tensor([0.5, 0.3, 0.2], device=device)

        weights = (gaussian_weights[0] * weights_gaussian_1 +
                   gaussian_weights[1] * weights_gaussian_2 +
                   gaussian_weights[2] * weights_gaussian_3)

        weights /= weights.sum(dim=2, keepdim=True) + 1e-8  # shape: [B, Nq, Nk]

        new_features = torch.bmm(weights, key_value)  # shape: [B, Nq, Ck]
        new_features = self.out_transform(new_features.transpose(1, 2)).transpose(1, 2)  # shape: [B, Nq, Cq]
        return new_features

class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta):
        super().__init__()
        self.out_dim = out_dim
        self.alpha = alpha
        self.beta = beta
    def Tr_embed(self, my_dim, Tensor):
        B, in_dim, G, K = Tensor.shape

        feat_dim = my_dim // (2 * in_dim)
        feat_range = torch.arange(feat_dim).float().cuda()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * Tensor.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)  # [B, _, G, K, feat]
        cos_embed = torch.cos(div_embed)
        Tr_embed = torch.stack([sin_embed, cos_embed], dim=5).flatten(4)

        Tr_embed = Tr_embed.permute(0, 1, 4, 2, 3).reshape(B, my_dim, G, K)

        return Tr_embed

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):
        B, G, K, C = knn_x.shape

        knn_x = normalize(knn_x, lc_x)
        knn_xyz = normalize(knn_xyz, lc_xyz)

        lc_xyz = lc_xyz.reshape(B, G, 1, -1).repeat(1, 1, K, 1).permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)

        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        cross_product = torch.cross(knn_xyz, lc_xyz, dim=1)
        dot_product = torch.sum(knn_xyz * lc_xyz, dim=1, keepdim=True)

        self.Fourier_embedding_f = FourierFeatureMapping(7, knn_x.shape[1], 50)
        embedding_element = torch.cat((knn_xyz, cross_product, dot_product), dim=1)
        embedding_element = self.Fourier_embedding_f(embedding_element)

        knn_x_w = torch.cat([knn_x, embedding_element], dim=1)
        position_embed = self.Tr_embed(knn_x_w.shape[1], knn_xyz) + self.Tr_embed(knn_x_w.shape[1], lc_xyz)
        final_x = knn_x_w + position_embed
        final_x *= position_embed

        return final_x

# Pooling
class K_ANP(nn.Module):
    def __init__(self, initial_p=2.0, dim=-1):
        super(K_ANP, self).__init__()
        self.initial_p = initial_p
        self.dim = dim

    def forward(self, knn_x_w):
        std_dev = torch.std(knn_x_w, dim=self.dim, keepdim=True)
        # 使用标量而不是张量调整 p
        p = self.initial_p + torch.log1p(std_dev.mean()).item()
        #p = self.initial_p
        norm = torch.norm(knn_x_w, p=p, dim=self.dim, keepdim=True)
        lc_x = norm / (torch.sum(norm, dim=self.dim, keepdim=True) + 1e-8)
        #print(lc_x.shape)

        e_x = torch.exp(lc_x) # B 2C G K
        up = (knn_x_w * e_x).mean(self.dim) # # B 2C G
        down = e_x.mean(self.dim)
        lc_x = torch.div(up, down)

        return lc_x

class Pooling(nn.Module):
    def __init__(self, out_dim, use_anp=True, p=2):
        super().__init__()
        self.use_anp = use_anp
        self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_dim),
                nn.GELU())
        #self.out_transform = nn.LeakyReLU(0.1)
        if use_anp:
            self.anp = K_ANP(initial_p=p, dim=-1)

    def forward(self, knn_x_w):
        if self.use_anp:
            k_anp_x = self.anp(knn_x_w)
            lc_x = k_anp_x + knn_x_w.mean(-1)
        else:
            lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        lc_x = self.out_transform(lc_x)
        return lc_x

class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim, mapping_size, scale=50.0):
        """
        input_dim: 输入维度（例如，点云的维度为3）
        mapping_size: 映射后的高维空间大小
        scale: 控制 Fourier Features 的频率范围
        """
        super(FourierFeatureMapping, self).__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        # scale_min = 1
        # scale_max = 100
        # scales = torch.linspace(scale_min, scale_max, steps=mapping_size // 2).unsqueeze(0)  # [1, mapping_size // 2]
        # self.B = torch.randn(input_dim, mapping_size // 2) * scales  # [input_dim, mapping_size // 2]

        self.B = torch.randn(input_dim, mapping_size // 2) * scale


    def forward(self, x):
        """
        x: 输入点云数据，形状为 [batch_size, 3, num_points]
        返回：高维空间的映射，形状为 [batch_size, mapping_size, num_points]
        """
        if len(x.shape) == 3:
            x = x.permute(0, 2, 1)  # 转置为 [batch_size, num_points, 3]
        if len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1)  # Shape: [batch_size, num_points, K_neighbors, input_dim]

        # 计算 Fourier 特征：Bx
        x_proj = torch.matmul(x, self.B.to(x.device))  # [batch_size, num_points, mapping_size // 2]

        # 使用 sin 和 cos 映射
        fourier_features = torch.cat([torch.sin(2 * torch.pi * x_proj),
                                      torch.cos(2 * torch.pi * x_proj)], dim=-1)
        fourier_features = torch.sign(fourier_features) * torch.abs(fourier_features) ** 5
        if len(x.shape) == 3:
            return fourier_features.permute(0, 2, 1)
        if len(x.shape) == 4:
            return fourier_features.permute(0, 3, 1, 2)

class Tr_Pos(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz):
        B, _, N = xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim).float().cuda()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)

        return position_embed

# Non-Parametric Encoder
class EncNP(nn.Module):  
    def __init__(self, input_points, depth, embed_dim, k_neighbors, alpha, beta, use_anp, p):
        super().__init__()
        self.input_points = input_points
        self.depth = depth
        self.embed_dim = embed_dim
        self.k_neighbors = k_neighbors
        self.alpha, self.beta = alpha, beta

        self.FPS_kNN_list = nn.ModuleList() # FPS, kNN
        self.LGA_list = nn.ModuleList() # Local Geometry Aggregation
        self.Line_block_list = nn.ModuleList()
        self.Pooling_list = nn.ModuleList() # Pooling

        self.attention_list_1 = nn.ModuleList()
        self.attention_list_2 = nn.ModuleList()
        self.attention_list_3 = nn.ModuleList()
        self.fourier_embed = FourierFeatureMapping(7, embed_dim, scale=20.0)

        out_dim = self.embed_dim
        group_num = self.input_points
        self.Tr_embed = Tr_Pos(3, out_dim=embed_dim, alpha=alpha, beta=beta)

        # Multi-stage Hierarchy
        for i in range(self.depth):
            out_dim = out_dim * 2
            group_num = group_num // 2
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.Line_block_list.append(Line_block(p=p))
            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta))
            self.Pooling_list.append(Pooling(out_dim, use_anp=use_anp, p=p))
            self.attention_list_1.append(NonparametricAttentionPooling(bandwidth=0.3))
            self.attention_list_2.append(NonparametricAttentionPooling(bandwidth=0.6))
            self.attention_list_3.append(NonparametricCrossAttentionPooling(bandwidth=0.5))

    def forward(self, xyz, x, normals):
        #x = self.Tr_embed(x)
        x_ln = compute_line_normals(x, normals)  #  7
        #x_nc = compute_line_normals_nc(x, normals).repeat(1, self.embed_dim//7, 1)  #  7
        #x_fn = compute_surface_normal(x, normals)  #  6
        #x = compute_surface_normal(x, normals).repeat(1, self.embed_dim//6, 1)  #  6
        #x = compute_oriented_tangent_plane(x, normals).repeat(1, self.embed_dim//4, 1)  #   4
        x = self.fourier_embed(x_ln)

        fixed_weights = torch.tensor([0.1, 0.2, 0.7, 1.0, 1.0])

        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]  # [B, C, N]

        # Multi-stage Hierarchy
        for i in range(self.depth):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))

            line_elem = self.Line_block_list[i](lc_x, knn_x)

            # Local Geometry Aggregation
            knn_x_w = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)

            # Pooling
            pooling_x = self.Pooling_list[i](knn_x_w)
            x = line_elem + pooling_x

            x_cat = torch.cat((lc_x.permute(0, 2, 1), x, line_elem), dim=1)
            # attention
            att_x_1 = self.attention_list_1[i](x_cat.shape[1], x_cat)
            att_x_2 = self.attention_list_2[i](att_x_1.shape[1], att_x_1)

            new_x = torch.cat((x_cat, att_x_1, att_x_2), dim=1)
            new_x = fixed_weights[i] * new_x

            xyz_list.append(xyz)
            x_list.append(new_x)

        return xyz_list, x_list


# Non-Parametric Decoder
class DecNP(nn.Module):
    def __init__(self, depth, de_neighbors):
        super().__init__()
        self.depth = depth
        self.de_neighbors = de_neighbors


    def propagate(self, xyz1, xyz2, points1, points2):
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :self.de_neighbors], idx[:, :, :self.de_neighbors]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            weight = weight.view(B, N, self.de_neighbors, 1)

            points2_indexed = index_points(points2, idx)

            index_points(xyz1, idx)
            interpolated_points = torch.sum(points2_indexed * weight, dim=2)

            mean_interpolated_points = interpolated_points.mean()
            std_interpolated_points = torch.std(interpolated_points - mean_interpolated_points)
            interpolated_points = (interpolated_points - mean_interpolated_points) / (std_interpolated_points + 1e-5)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)

        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        return new_points


    def forward(self, xyz_list, x_list):
        xyz_list.reverse()
        x_list.reverse()

        x = x_list[0]
        for i in range(self.depth):
            x = self.propagate(xyz_list[i+1], xyz_list[i], x_list[i+1], x)
        return x

# Non-Parametric Network
class NPMFF_Seg_normals(nn.Module):
    def __init__(self, input_points=2048, depth=4, embed_dim=84, k_neighbors=128, use_anp=True, p=2, de_neighbors=6, beta=1500, alpha=100):
        super().__init__()
        # Non-Parametric Encoder and Decoder

        self.EncNP = EncNP(input_points, depth, embed_dim, k_neighbors, alpha, beta, use_anp, p)
        self.DecNP = DecNP(depth, de_neighbors)


    def forward(self, x):
        # xyz: point coordinates
        # x: point features
        xyz = x[:, :3, :].permute(0, 2, 1).contiguous()
        normals = x[:, 3:, :]
        xyz_norm = x
        x = x[:, :3, :].contiguous()

        xyz_list, x_list = self.EncNP(xyz, x, normals)
        x = self.DecNP(xyz_list, x_list)

        return x