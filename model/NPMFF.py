# Non-Parametric Networks for 3D Point Cloud Classification
import torch
from pointnet2_ops import pointnet2_utils
from .model_utils import *
import matplotlib.pyplot as plt
#from data.show3d_balls import showpoints

def normalize(x, ref):
    """
    对输入 x 进行标准化，基于 ref 的均值和标准差
    """
    mean = ref.unsqueeze(dim=-2)
    std = torch.std(x - mean)
    #std = torch.std(x - mean, dim=-1, keepdim=True)
    return (x - mean) / (std + 1e-5)

class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x):
        B, N, _ = xyz.shape
        #FPS
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long()
        lc_xyz = index_points(xyz, fps_idx)
        lc_x = index_points(x, fps_idx)

        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)

        #knn_normals = index_points(normals, knn_idx)

        return lc_xyz, lc_x, knn_xyz, knn_x

class Line_block(nn.Module):
    def __init__(self, p, use_anp=True):
        super().__init__()
        self.use_anp = use_anp
        if use_anp:
            self.k_anp = K_ANP(initial_p=p, dim=-1)
        self.Fourier_embedding_knn = None  # 延迟初始化，以根据输入维度适应

    def forward(self, lc_x, knn_x):
        # 标准化
        knn_x = normalize(knn_x, lc_x)

        # Compute local cross product and dot product
        B, G, K, N = knn_x.shape

        # lc_normal: [B, G, K, 3], repeat 将 normal 扩展到所有 K
        lc_exp = lc_x.unsqueeze(2).repeat(1, 1, K, 1)
        lc_normal, _ = lc_x[:, :, :3].topk(3, dim=2)  # 获取前三个最大的值
        lc_normal = lc_normal.unsqueeze(2).repeat(1, 1, K, 1)
        # 标准化 lc_line，并进行转置 [B, N, G]
        lc_line = lc_x
        lc_line_norm = lc_line.norm(dim=2, keepdim=True)
        lc_line = (lc_line / lc_line_norm).transpose(1, 2)

        # knn_product = knn_x[:, :, :, 3:6]
        #test = lc_exp * knn_x

        knn_product, _ = knn_x[:, :, :, :3].topk(3, dim=3)  # 获取前三个最大的值
        knn_cross_product = torch.cross(knn_product, lc_normal, dim=3)
        knn_dot_product = torch.sum(knn_x * lc_exp, dim=3, keepdim=True)

        knn_line = torch.cat((knn_x, knn_cross_product, knn_dot_product), dim=3).permute(0, 3, 1, 2)
        # 延迟初始化 Fourier embedding
        if self.Fourier_embedding_knn is None:
            self.Fourier_embedding_knn = FourierFeatureMapping(knn_line.shape[1], lc_line.shape[1], 10)
        knn_line = self.Fourier_embedding_knn(knn_line)

        if self.use_anp:
            k_anp_x = self.k_anp(knn_line)
            knn_line = k_anp_x + knn_line.mean(-1)
        else:
            knn_line = knn_line.max(-1)[0] + knn_line.mean(-1)

        knn_line_norm = knn_line.norm(dim=-1, keepdim=True)
        knn_line = F.gelu(knn_line / knn_line_norm)

        #knn_line = self.out_transform(knn_line)

        line_element = knn_line - lc_line
        #print(knn_line.shape, lc_line.shape)
        line_element = torch.cat([line_element, lc_line], dim=1)

        return line_element


class NonparametricAttentionPooling(nn.Module):
    def __init__(self, bandwidth):
        super().__init__()
        self.bandwidth = bandwidth

    def forward(self ,x):
        device = x.device
        B, C, N = x.shape
        self.out_transform = nn.Sequential(
                nn.BatchNorm1d(C),
                nn.GELU() ).to(device)

        norms = torch.cdist(x.transpose(1, 2), x.transpose(1, 2))  # shape: [B, N, N]

        # 使用高斯核
        weights_gaussian_1 = torch.exp(-0.5 * (norms / self.bandwidth) ** 2)
        weights_gaussian_2 = torch.exp(-0.5 * (norms / (self.bandwidth * 2)) ** 2)
        weights_gaussian_3 = torch.exp(-0.5 * (norms / (self.bandwidth / 2)) ** 2)

        # 固定的加权平均的权重
        gaussian_weights = torch.tensor([0.5, 0.3, 0.2])

        # 加权组合
        weights = (gaussian_weights[0] * weights_gaussian_1 +
                   gaussian_weights[1] * weights_gaussian_2 +
                   gaussian_weights[2] * weights_gaussian_3)

        # 归一化权重
        weights /= weights.sum(dim=2, keepdim=True) + 1e-8  # shape: [B, N, N]

        # 加权平
        new_features = torch.bmm(weights, x.transpose(1, 2))  # shape: [B, N, F]
        new_features = new_features.transpose(1, 2)  # shape: [B, F, N]
        new_features = self.out_transform(new_features)

        return new_features

# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta):
        super().__init__()
        self.out_dim = out_dim
        self.alpha = alpha
        self.beta = beta

    def Tr_embed(self, my_dim, Tensor):
        B, in_dim, G, K = Tensor.shape

        Tensor = Tensor.permute(0, 2, 3, 1)[..., None]
        feat_dim = my_dim // (in_dim * 2)

        feat_range = torch.arange(feat_dim).float().cuda()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * Tensor.unsqueeze(-1), dim_embed)
        Tr_embed = torch.stack([torch.sin(div_embed), torch.cos(div_embed)], dim=4).flatten(3)
        Tr_embed = Tr_embed.permute(0, 3, 1, 2)

        return Tr_embed

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):
        B, G, K, C = knn_x.shape

        # Normalize x (features) and xyz (coordinates)
        knn_x = normalize(knn_x, lc_x)
        knn_xyz = normalize(knn_xyz, lc_xyz)

        lc_xyz = lc_xyz.reshape(B, G, 1, -1).repeat(1, 1, K, 1).permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)

        cross_product = torch.cross(knn_xyz, lc_xyz, dim=1)
        dot_product = torch.sum(knn_xyz * lc_xyz, dim=1, keepdim=True)
        ##_______________Fourier_embed_________________##

        # self.Fourier_embedding_f = FourierFeatureMapping(7, knn_x.shape[1], 10)
        # embedding_element = torch.cat((knn_xyz, cross_product, dot_product), dim=1)
        # embedding_element = self.Fourier_embedding_f(embedding_element)

        ##_______________simply repeat_________________##
        embedding_element = torch.cat((knn_xyz, cross_product, dot_product), dim=1).repeat(1, knn_x.shape[1]//7, 1, 1)

        knn_x_w = torch.cat([knn_x, embedding_element], dim=1)

        position_embed = self.Tr_embed(self.out_dim, knn_xyz) + self.Tr_embed(self.out_dim, lc_xyz)
        final_x = knn_x_w + position_embed
        final_x *= position_embed
        #print(final_x.shape)

        return final_x


# Pooling
class K_ANP(nn.Module):
    def __init__(self, initial_p=2.0, dim=-1):
        super(K_ANP, self).__init__()
        self.initial_p = initial_p
        self.dim = dim

    def forward(self, knn_x_w):
        std_dev = torch.std(knn_x_w, dim=self.dim, keepdim=True)

        # *********   Method 1: adaptive p with std_dev   *********
        p = self.initial_p + torch.log1p(std_dev.mean()).item()

        norm = torch.norm(knn_x_w, p=p, dim=self.dim, keepdim=True)
        lc_x = norm / (torch.sum(norm, dim=self.dim, keepdim=True) + 1e-8)

        e_x = torch.exp(lc_x)
        up = (knn_x_w * e_x).mean(self.dim) # # B 2C G
        down = e_x.mean(self.dim)
        lc_x = torch.div(up, down)

        return lc_x

class Pooling(nn.Module):
    def __init__(self, out_dim, use_anp, p=2):
        super().__init__()
        self.use_anp = use_anp
        self.out_transform = nn.Sequential(
            nn.BatchNorm1d(out_dim),
            nn.GELU()
        )
        if use_anp:
            self.k_anp = K_ANP(initial_p=p, dim=-1)

    def forward(self, knn_x_w):
        if self.use_anp:
            k_anp_x = self.k_anp(knn_x_w)
            lc_x = k_anp_x + knn_x_w.mean(-1)
            #lc_x = k_anp_x + knn_x_w.max(-1)[0]
            #lc_x = k_anp_x
        else:
            lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        lc_x = self.out_transform(lc_x)
        return lc_x


class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim, mapping_size, scale=10.0):
        """
        input_dim: 输入维度（例如，点云的维度为3）
        mapping_size: 映射后的高维空间大小
        scale: 控制 Fourier Features 的频率范围
        """
        super(FourierFeatureMapping, self).__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale

        self.B = torch.randn(input_dim, mapping_size//2) * scale

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
        fourier_features = torch.cat([torch.sin(2 * torch.pi * x_proj), torch.cos(2 * torch.pi * x_proj)], dim=-1)

        # 使用 sin 和 cos 映射
        #fourier_features = torch.cat([torch.sin(2 * torch.pi * x_proj), torch.cos(2 * torch.pi * x_proj)], dim=-1)

        #fourier_features = torch.cos(2 * torch.pi * x_proj)

        #fourier_features = torch.sign(fourier_features) * torch.abs(fourier_features) ** 5

        if len(x.shape) == 3:
            return fourier_features.permute(0, 2, 1)
        if len(x.shape) == 4:
            return fourier_features.permute(0, 3, 1, 2)

class NonparametricCrossAttentionPooling(nn.Module):
    def __init__(self, bandwidth):
        super().__init__()
        self.bandwidth = bandwidth
        self.out_transform = None

    def forward(self, query, key_value):
        """
        Args:
            query: Tensor of shape [B, Fq, Nq] (batch size, query features, query sequence length)
            key_value: Tensor of shape [B, Fk, Nk] (batch size, key-value features, key-value sequence length)

        Returns:
            new_features: Tensor of shape [B, Fq, Nq] (output features aligned to query)
        """
        device = query.device
        batch_size, query_length, query_features = query.size()
        _, key_value_length, key_value_features = key_value.size()

        if self.out_transform is None or self.out_transform[0].num_features != query_features:
            self.out_transform = nn.Sequential(
                nn.BatchNorm1d(query_features),
                nn.GELU()
            ).to(device)

        # Compute pairwise distances between query and key-value
        #query_t = query.transpose(1, 2)  # shape: [B, Nq, Fq]
        norms = torch.cdist(query, key_value)  # shape: [B, Nq, Nk]

        # Apply Gaussian kernels
        weights_gaussian_1 = torch.exp(-0.5 * (norms / self.bandwidth) ** 2)
        weights_gaussian_2 = torch.exp(-0.5 * (norms / (self.bandwidth * 2)) ** 2)
        weights_gaussian_3 = torch.exp(-0.5 * (norms / (self.bandwidth / 2)) ** 2)

        # Fixed weighted averages
        gaussian_weights = torch.tensor([0.5, 0.3, 0.2], device=device)

        # Weighted combination
        weights = (gaussian_weights[0] * weights_gaussian_1 +
                   gaussian_weights[1] * weights_gaussian_2 +
                   gaussian_weights[2] * weights_gaussian_3)

        # Normalize weights along the key-value dimension
        weights /= weights.sum(dim=2, keepdim=True) + 1e-8  # shape: [B, Nq, Nk]

        # Key-value transpose for weighted sum: [B, Nk, Ck]
        new_features = torch.bmm(weights, key_value)  # shape: [B, Nq, Ck]
        new_features = self.out_transform(new_features.transpose(1, 2)).transpose(1, 2)  # shape: [B, Nq, Cq]
        return new_features

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
    def __init__(self, input_points, depth, embed_dim, k_neighbors, alpha, beta, use_anp, p=2):
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
        self.fourier_embed = FourierFeatureMapping(3, embed_dim, scale=30.0)
        self.Tr_embed = Tr_Pos(3, embed_dim, alpha, beta)


        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.depth):
            out_dim = out_dim * 2
            group_num = group_num // 2
            self.FPS_kNN_list.append(FPS_kNN(group_num, k_neighbors))
            self.Line_block_list.append(Line_block(p=p, use_anp=use_anp))
            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta))
            self.Pooling_list.append(Pooling(out_dim, use_anp=True, p=p))
            self.attention_list_1.append(NonparametricAttentionPooling(bandwidth=0.3))
            self.attention_list_2.append(NonparametricAttentionPooling(bandwidth=0.8))
            self.attention_list_3.append(NonparametricCrossAttentionPooling(bandwidth=0.5))

    def forward(self, xyz, x, normals):
        #x = compute_line_rpi(x, 50)
        #x = compute_line_normals(x, normals).repeat(1, self.embed_dim//7, 1)
        #x_ln = compute_line_normals(x, normals)
        #x_nc = compute_line_normals_nc(x, normals)  #  7
        #x_fn = compute_surface_normal(x, normals)  #  6
        #x = compute_surface_normal(x, normals).repeat(1, self.embed_dim//6, 1)  #  6
        #x_ot = compute_oriented_tangent_plane(x, normals) #   4
        #x = torch.concat([0.3 * x_fn, 0.8 * x_ot], dim=1)
        #x = compute_oriented_tangent_plane(x, normals).repeat(1, self.embed_dim//4, 1) #   4
        #x = xyz.permute(0, 2, 1).repeat(1, self.embed_dim//3, 1)

        #x = torch.cat([x_fn, x_ot], dim=1) #86.06
        #print(x.shape)
        #x = self.Tr_embed(x)
        x = self.fourier_embed(x)

        new_x_list = []  # 用于存储所有 new_x
        fixed_weights = torch.tensor([0.1, 0.1, 0.8, 1.2])

        for i in range(self.depth):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))

            # Line Elements Extract
            line_elem = self.Line_block_list[i](lc_x, knn_x)
            knn_x_w = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)

            pooling_x = self.Pooling_list[i](knn_x_w)
            x = line_elem + pooling_x

            x_cat = torch.cat((lc_x.permute(0, 2, 1), x, line_elem), dim=1)
            att_x_1 = self.attention_list_1[i](x_cat)
            att_x_2 = self.attention_list_2[i](att_x_1)
            att_x_3 = self.attention_list_3[i](att_x_1, lc_x.permute(0, 2, 1))
            #att_x_3 = self.attention_list_3[i](lc_x, att_x_1.permute(0, 2, 1))
            x_new = torch.cat((x_cat, att_x_1, att_x_2, att_x_3), dim=1)

            new_x = x_new.max(-1)[0] + x_new.mean(-1)
            new_x = fixed_weights[i] * new_x
            new_x_list.append(new_x)

        fused_feature  = torch.cat(new_x_list, dim=-1)
        fused_feature = fused_feature/fused_feature.sum(dim=-1, keepdim=True) + 1e-8

        return fused_feature
        #return new_x

# Non-Parametric Network
class NPMFF(nn.Module):
    def __init__(self, input_points=1024, depth=4, embed_dim=72, k_neighbors=90, beta=100, alpha=1500, use_anp=True, p=2):
        super().__init__()
        # Non-Parametric Encoder
        self.EncNP = EncNP(input_points, depth, embed_dim, k_neighbors, alpha, beta, use_anp, p)

    def forward(self, x):
        # xyz: point coordinates
        # x: point features
        xyz = x[:, :3, :].permute(0, 2, 1).contiguous()
        normals = x[:, 3:, :].contiguous()

        x = x[:, :3, :].contiguous()
        x = self.EncNP(xyz, x, normals)

        return x
