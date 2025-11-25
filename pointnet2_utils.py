import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

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

#
# def index_points(points, idx):
#     """
#
#     Input:
#         points: input points data, [B, N, C]
#         idx: sample index data, [B, S]
#     Return:
#         new_points:, indexed points data, [B, S, C]
#     """
#     device = points.device
#     B = points.shape[0]
#     view_shape = list(idx.shape)
#     view_shape[1:] = [1] * (len(view_shape) - 1)
#     repeat_shape = list(idx.shape)
#     repeat_shape[0] = 1
#     batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
#     new_points = points[batch_indices, idx, :]
#
#     device = points.device
#     B, N, C = points.shape
#     if torch.any(idx >= N):
#         print(f"Error: idx contains values >= {N}")
#         print(f"Max idx value: {idx.max()}")
#         print(f"idx shape: {idx.shape}")
#         print(f"points shape: {points.shape}")
#         raise ValueError("Index out of bounds in index_points")
#
#     return new_points
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] 或 [B, S, K]
    Return:
        new_points: indexed points data, [B, S, C] 或 [B, S, K, C]
    """
    device = points.device
    B, N, C = points.shape

    # 检查 idx 的维度，以处理不同的情况
    if idx.dim() == 2:
        S = idx.shape[1]
        # 检查 idx 是否在有效范围内
        if torch.any(idx >= N) or torch.any(idx < 0):
            print(f"Error in index_points: idx contains invalid indices.")
            print(f"idx max value: {idx.max()}, N: {N}")
            print(f"idx min value: {idx.min()}, N: {N}")
            print(f"idx shape: {idx.shape}")
            print(f"points shape: {points.shape}")
            raise ValueError("Index out of bounds in index_points.")

        # 构建 [B, S] 的 batch_indices
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(B, 1).repeat(1, S)  # [B, S]
    elif idx.dim() == 3:
        S, K = idx.shape[1], idx.shape[2]
        # 检查 idx 是否在有效范围内
        if torch.any(idx >= N) or torch.any(idx < 0):
            print(f"Error in index_points: idx contains invalid indices.")
            print(f"idx max value: {idx.max()}, N: {N}")
            print(f"idx min value: {idx.min()}, N: {N}")
            print(f"idx shape: {idx.shape}")
            print(f"points shape: {points.shape}")
            raise ValueError("Index out of bounds in index_points.")

        # 构建 [B, S, K] 的 batch_indices
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(B, 1, 1).repeat(1, S, K)  # [B, S, K]
    else:
        raise ValueError("idx 的维度必须是 2 或 3")

    # 执行索引操作
    new_points = points[batch_indices, idx, :]  # [B, S, C] 或 [B, S, K, C]

    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

# def query_ball_point(radius, nsample, xyz, new_xyz):
#     """
#     initional
#     Input:
#         radius: local region radius
#         nsample: max sample number in local region
#         xyz: all points, [B, N, 3]
#         new_xyz: query points, [B, S, 3]
#     Return:
#         group_idx: grouped points index, [B, S, nsample]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     _, S, _ = new_xyz.shape
#     group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
#     sqrdists = square_distance(new_xyz, xyz)
#     group_idx[sqrdists > radius ** 2] = N
#     group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
#     group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
#     mask = group_idx == N
#     group_idx[mask] = group_first[mask]
#     return group_idx


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    输入：
        radius: 球形区域的半径
        nsample: 每个区域内的最大点数
        xyz: 所有点的坐标，形状 [B, N, 3]
        new_xyz: 查询点的坐标，形状 [B, S, 3]
    返回：
        group_idx: 分组点的索引，形状 [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    # 将超过半径的距离设置为一个大值
    mask = sqrdists > radius ** 2
    sqrdists[mask] = 1e10  # 设定一个很大的距离值
    # 获取距离最近的 nsample 个点的索引
    group_idx = torch.argsort(sqrdists, dim=-1)[:, :, :nsample]  # [B, S, nsample]
    return group_idx

# def query_ball_point(radius, nsample, xyz, new_xyz):
#     """
#     Input:
#         radius: local region radius
#         nsample: max sample number in local region
#         xyz: all points, [B, N, 3]
#         new_xyz: query points, [B, S, 3]
#     Return:
#         group_idx: grouped points index, [B, S, nsample]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     _, S, _ = new_xyz.shape
#     group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
#     sqrdists = square_distance(new_xyz, xyz)
#     group_idx[sqrdists > radius ** 2] = N  # 将超出半径的点索引设为 N
#     group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
#     group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
#     mask = group_idx == N
#     group_idx[mask] = group_first[mask]
#
#     # 检查索引是否在有效范围内
#     if torch.any(group_idx >= N) or torch.any(group_idx < 0):
#         print(f"Error in query_ball_point: group_idx contains invalid indices.")
#         print(f"group_idx max value: {group_idx.max()}, N: {N}")
#         print(f"group_idx min value: {group_idx.min()}, N: {N}")
#         print(f"group_idx shape: {group_idx.shape}")
#         print(f"xyz shape: {xyz.shape}")
#         raise ValueError("Index out of bounds in query_ball_point.")
#
#     return group_idx

# def query_ball_point(radius, nsample, xyz, new_xyz, fps_idx):
#     """
#     Input:
#         radius: local region radius
#         nsample: max sample number in local region
#         xyz: all points, [B, N, 3]
#         new_xyz: query points, [B, S, 3]
#         fps_idx: farthest point indices, [B, S]
#     Return:
#         group_idx: grouped points index, [B, S, nsample]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     _, S, _ = new_xyz.shape
#     group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])  # [B, S, N]
#     sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
#     group_idx[sqrdists > radius ** 2] = N  # 设置超出半径的点索引为 N
#     group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # 取前 nsample 个点
#     group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])  # [B, S, nsample]
#     mask = group_idx == N
#     group_idx[mask] = group_first[mask]  # 将无效索引替换为第一个索引
#
#     # **关键修改**：将每个组的第一个索引设置为对应的 fps_idx
#     group_idx[:, :, 0] = fps_idx # 确保每个组至少包含其自身的索引
#
#     # 检查索引是否有效
#     if torch.any(group_idx >= N) or torch.any(group_idx < 0):
#         print(f"Error in query_ball_point: group_idx contains invalid indices.")
#         print(f"group_idx max value: {group_idx.max()}, N: {N}")
#         print(f"group_idx min value: {group_idx.min()}, N: {N}")
#         print(f"group_idx shape: {group_idx.shape}")
#         print(f"xyz shape: {xyz.shape}")
#         raise ValueError("Index out of bounds in query_ball_point.")
#
#     return group_idx


# def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
#     """
#     Input:
#         npoint:
#         radius:
#         nsample:
#         xyz: input points position data, [B, N, 3]
#         points: input points data, [B, N, D]
#     Return:
#         new_xyz: sampled points position data, [B, npoint, 3]
#         new_points: sampled points data, [B, npoint, nsample, 3+D]
#     """
#     B, N, C = xyz.shape
#     S = npoint
#     fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
#     new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
#     idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, npoint, nsample]
#     grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, 3]
#     grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
#
#     if points is not None:
#         grouped_points = index_points(points, idx)  # [B, npoint, nsample, D]
#         new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, 3+D]
#     else:
#         new_points = grouped_xyz_norm  # [B, npoint, nsample, 3]
#
#     if returnfps:
#         return new_xyz, new_points, grouped_xyz, fps_idx
#     else:
#         return new_xyz, new_points
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
    sqrdists = square_distance(new_xyz, xyz)  # [B, npoint, N]
    idx = torch.argsort(sqrdists, dim=-1)[:, :, :nsample]  # [B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
    if points is not None:
        grouped_points = index_points(points, idx)  # [B, npoint, nsample, D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, 3+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

