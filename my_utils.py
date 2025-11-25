import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = x1x2 + y1y2 + z1z2;
    sum(src^2, dim=-1) = x1^2 + y1^2 + z1^2;
    sum(dst^2, dim=-1) = x2^2 + y2^2 + z2^2;
    dist = (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: [B, N, M]
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
        new_points: sampled points data, [B, S, C]
    """
    device = points.device
    B=points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples,8192
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

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(k,xyz,new_xyz):
    """
    Input:
        k: number of points in the neighborhood
        xyz:all points, [B, N, 3]
        new_xyz:query points, [B, S, 3]
    Return:
        group_idx:grouped points index, [B, S, k]
        dist:distances, [B, S, k]
    """
    sqrdists = square_distance(new_xyz,xyz)
    #topk find the k nearest points
    _,grup_idx = torch.topk(sqrdists,k,dim=-1,largest=False,sorted=False)
    return grup_idx