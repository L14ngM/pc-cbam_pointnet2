import torch
import torch.nn as nn
import torch.nn.functional as F
from my_utils import farthest_point_sample, index_points, query_ball_point, knn_point

class SElayer(nn.Module):
    """
    Squeeze-and-Excitation Layer for Point Cloud Feature [B, C, N', K]
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = F.adaptive_max_pool2d(x, 1)  # [B, C, 1, 1]
        y = self.fc(y)
        return x * y.expand_as(x)

class pc_CBAM_module(nn.Module):
    """
    Point Cloud CBAM Module with Spatial-First Strategy
    """
    def __init__(self, channels, reduction_ration=16):
        super(pc_CBAM_module, self).__init__()

        self.mlp_edge = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.mlp_spatial = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1, 1),
        )

        self.mlp_channel = nn.Sequential(
            nn.Linear(channels, channels // reduction_ration, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ration, channels, bias=False),
        )

    def forward(self, F_local, pos_local_norm):
        # F_local: [B, C, N', K]
        # pos_local_norm: [B, 3, N', K] (normalized relative to center points)

        center_feature = F_local[:, :, :, 0:1].expand_as(F_local)
        edge_feature = torch.cat([center_feature, F_local - center_feature], dim=1)

        edge_feature_transformed = self.mlp_edge(edge_feature)

        Ms_raw = self.mlp_spatial(edge_feature_transformed)
        Ms = torch.sigmoid(Ms_raw)  # [B, 1, N', K]
        F_spatial = F_local * Ms

        avg_pool = torch.mean(F_spatial, dim=3)        # [B, C, N']
        max_pool = torch.max(F_spatial, dim=3)[0]      # [B, C, N']

        avg_pool_out = self.mlp_channel(avg_pool.transpose(1, 2))  # [B, N', C]
        max_pool_out = self.mlp_channel(max_pool.transpose(1, 2))  # [B, N', C]

        Mc_raw = avg_pool_out + max_pool_out
        Mc = torch.sigmoid(Mc_raw).transpose(1, 2).unsqueeze(-1)   # [B, C, N', 1]
        F_cbam = F_spatial * Mc

        F_out = F_cbam + F_local
        return F_out

class ConfigurableSALayer(nn.Module):
    """
    SA layer with ablation controls for enabling/disabling SE and PC-CBAM.
    """
    def __init__(self, npoint, k, in_channel, mlp_channels, use_se=True, use_pc_cbam=True):
        super(ConfigurableSALayer, self).__init__()
        self.npoint = npoint
        self.k = k
        self.use_se = use_se
        self.use_pc_cbam = use_pc_cbam

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3

        for i, out_channel in enumerate(mlp_channels):
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        if self.use_se:
            self.se = SElayer(mlp_channels[0])

        if self.use_pc_cbam:
            self.pc_cbam = pc_CBAM_module(mlp_channels[-1])

    def forward(self, xyz, points):
        # xyz: [B, 3, N]
        # points: [B, C_in, N]

        xyz_trans = xyz.permute(0, 2, 1)
        if points is not None:
            points_trans = points.permute(0, 2, 1)
        else:
            points_trans = None

        center_idx = farthest_point_sample(xyz_trans, self.npoint)
        new_xyz = index_points(xyz_trans, center_idx)

        group_idx = knn_point(self.k, xyz_trans, new_xyz)
        grouped_xyz = index_points(xyz_trans, group_idx)

        grouped_xyz_norm = grouped_xyz - new_xyz.view(xyz.shape[0], self.npoint, 1, 3)

        if points_trans is not None:
            grouped_points = index_points(points_trans, group_idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm

        new_points = new_points.permute(0, 3, 1, 2)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
            if i == 0 and self.use_se and len(self.mlp_convs) > 1:
                new_points = self.se(new_points)

        F_local = new_points

        if self.use_pc_cbam:
            grouped_xyz_norm_permuted = grouped_xyz_norm.permute(0, 3, 1, 2)
            F_enhanced = self.pc_cbam(F_local, grouped_xyz_norm_permuted)
        else:
            F_enhanced = F_local

        new_points_aggregated = torch.max(F_enhanced, 3)[0]

        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points_aggregated

class EnhancedSALayer(nn.Module):
    def __init__(self, npoint, k, in_channel, mlp_channels):
        super(EnhancedSALayer, self).__init__()
        self.npoint = npoint
        self.k = k

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3

        for i, out_channel in enumerate(mlp_channels):
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.se = SElayer(mlp_channels[0])
        self.pc_cbam = pc_CBAM_module(mlp_channels[-1])

    def forward(self, xyz, points):
        # xyz: [B, 3, N]
        # points: [B, C_in, N]

        xyz_trans = xyz.permute(0, 2, 1)
        if points is not None:
            points_trans = points.permute(0, 2, 1)
        else:
            points_trans = None

        center_idx = farthest_point_sample(xyz_trans, self.npoint)
        new_xyz = index_points(xyz_trans, center_idx)

        group_idx = knn_point(self.k, xyz_trans, new_xyz)
        grouped_xyz = index_points(xyz_trans, group_idx)

        grouped_xyz_norm = grouped_xyz - new_xyz.view(xyz.shape[0], self.npoint, 1, 3)

        if points_trans is not None:
            grouped_points = index_points(points_trans, group_idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm

        new_points = new_points.permute(0, 3, 1, 2)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
            if i == 0 and len(self.mlp_convs) > 1:
                new_points = self.se(new_points)

        F_local = new_points

        grouped_xyz_norm_permuted = grouped_xyz_norm.permute(0, 3, 1, 2)
        F_cbam_out = self.pc_cbam(F_local, grouped_xyz_norm_permuted)

        new_points_aggregated = torch.max(F_cbam_out, 3)[0]

        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points_aggregated

class CBAMOnlySALayer(nn.Module):
    """SA layer using only PC-CBAM (no SE)."""
    def __init__(self, npoint, k, in_channel, mlp_channels):
        super(CBAMOnlySALayer, self).__init__()
        self.npoint = npoint
        self.k = k

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3

        for i, out_channel in enumerate(mlp_channels):
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.pc_cbam = pc_CBAM_module(mlp_channels[-1])

    def forward(self, xyz, points):
        # xyz: [B, 3, N]
        # points: [B, C_in, N]

        xyz_trans = xyz.permute(0, 2, 1)
        if points is not None:
            points_trans = points.permute(0, 2, 1)
        else:
            points_trans = None

        center_idx = farthest_point_sample(xyz_trans, self.npoint)
        new_xyz = index_points(xyz_trans, center_idx)

        group_idx = knn_point(self.k, xyz_trans, new_xyz)
        grouped_xyz = index_points(xyz_trans, group_idx)

        grouped_xyz_norm = grouped_xyz - new_xyz.view(xyz.shape[0], self.npoint, 1, 3)

        if points_trans is not None:
            grouped_points = index_points(points_trans, group_idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm

        new_points = new_points.permute(0, 3, 1, 2)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        F_local = new_points

        grouped_xyz_norm_permuted = grouped_xyz_norm.permute(0, 3, 1, 2)
        F_cbam_out = self.pc_cbam(F_local, grouped_xyz_norm_permuted)

        new_points_aggregated = torch.max(F_cbam_out, 3)[0]

        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points_aggregated

class BaselineSALayer(nn.Module):
    """Baseline SA layer without any attention mechanism."""
    def __init__(self, npoint, k, in_channel, mlp_channels):
        super(BaselineSALayer, self).__init__()
        self.npoint = npoint
        self.k = k

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3

        for i, out_channel in enumerate(mlp_channels):
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        # xyz: [B, 3, N]
        # points: [B, C_in, N]

        xyz_trans = xyz.permute(0, 2, 1)
        if points is not None:
            points_trans = points.permute(0, 2, 1)
        else:
            points_trans = None

        center_idx = farthest_point_sample(xyz_trans, self.npoint)
        new_xyz = index_points(xyz_trans, center_idx)

        group_idx = knn_point(self.k, xyz_trans, new_xyz)
        grouped_xyz = index_points(xyz_trans, group_idx)

        grouped_xyz_norm = grouped_xyz - new_xyz.view(xyz.shape[0], self.npoint, 1, 3)

        if points_trans is not None:
            grouped_points = index_points(points_trans, group_idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz_norm

        new_points = new_points.permute(0, 3, 1, 2)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points_aggregated = torch.max(new_points, 3)[0]

        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points_aggregated
