import torch
import torch.nn as nn
import torch.nn.functional as F
from my_utils import farthest_point_sample,index_points,query_ball_point,knn_point

class SElayer(nn.Module):
    
    """
    Squeeze-and-Excitation Layer for Point Cloud Feature [B,C,N',K]
    """
    def __init__(self,in_channels,reduction=16):
        super().__init__()
        #使用Conv2d处理[B,C,N',K]input
        self.fc=nn.Sequential(
            nn.Conv2d(in_channels,in_channels//reduction,1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction,in_channels,1,bias=False),
            nn.Sigmoid()
        )        
    def forward(self,x):
        y=F.adaptive_max_pool2d(x,1)#[B,C,1,1]to[B,C,1,1]
        y=self.fc(y)#rescale the feature map
        return x*y.expand_as(x)

class pc_CBAM_module(nn.Module):
    """
    Point Cloud CBAM Module with Spatial First Strategy
    """
    def __init__(self,channels,reduction_ration=16):
        super(pc_CBAM_module,self).__init__()

        """
        PC-SAM Component
        Edge Convolution's MLP :Process the feature map of edge convolution
        """
        self.mlp_edge=nn.Sequential(
            nn.Conv2d(channels*2,channels,1,bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        #MLP to generate the spatial weight from EdgeConv's output
        self.mlp_spatial=nn.Sequential(
            nn.Conv2d(channels,channels//4,1,bias=False),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4,1,1,1),
        )
        """
        PC-CAM Component
        Shared MLP for channel attention
        """
        self.mlp_channel=nn.Sequential(
            nn.Linear(channels,channels//reduction_ration,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction_ration,channels,bias=False),
        )


    def forward(self, F_local, pos_local_norm):
        # F_local: [B, C, N', K]
        # pos_local_norm: [B, 3, N', K] (already normalized relative to center points)
        
        # ==================== PC-SAM (Spatial Attention) ====================
        center_feature = F_local[:, :, :, 0:1].expand_as(F_local)
        # 构造边特征，但不包含相对坐标，因为EdgeConv的经典实现不一定包含它
        # 如果需要，可以拼接 pos_local_norm
        edge_feature = torch.cat([center_feature, F_local - center_feature], dim=1)
        
        edge_feature_transformed = self.mlp_edge(edge_feature)
        
        Ms_raw = self.mlp_spatial(edge_feature_transformed)
        Ms = torch.sigmoid(Ms_raw)  # Spatial weights [B, 1, N', K]

        F_spatial = F_local * Ms  # Apply spatial attention

        # ==================== PC-CAM (Channel Attention) ====================
        avg_pool = torch.mean(F_spatial, dim=3) # -> [B, C, N']
        max_pool = torch.max(F_spatial, dim=3)[0] # -> [B, C, N']
        
        avg_pool_out = self.mlp_channel(avg_pool.transpose(1, 2)) # -> [B, N', C]
        max_pool_out = self.mlp_channel(max_pool.transpose(1, 2)) # -> [B, N', C]
        
        Mc_raw = avg_pool_out + max_pool_out
        Mc = torch.sigmoid(Mc_raw).transpose(1, 2).unsqueeze(-1) # Channel weights [B, C, N', 1]

        F_cbam = F_spatial * Mc # Apply channel attention

        # ==================== Residual Connection ====================
        F_out = F_cbam + F_local

        return F_out

# === 新增：可配置的消融实验SA层 ===
class ConfigurableSALayer(nn.Module):
    """
    支持消融实验的SA层，可以独立控制SE和PC-CBAM的启用状态
    """
    def __init__(self, npoint, k, in_channel, mlp_channels, use_se=True, use_pc_cbam=True):
        super(ConfigurableSALayer, self).__init__()
        self.npoint = npoint
        self.k = k
        self.use_se = use_se
        self.use_pc_cbam = use_pc_cbam

        # PointNet unit
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3  # concatenate relative position xyz

        # Define MLP layers
        for i, out_channel in enumerate(mlp_channels):
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        # SE layer (可选)
        if self.use_se:
            self.se = SElayer(mlp_channels[0])

        # PC-CBAM (可选)
        if self.use_pc_cbam:
            self.pc_cbam = pc_CBAM_module(mlp_channels[-1])

    def forward(self, xyz, points):
        # xyz: [B, 3, N]
        # points: [B, C_in, N]
        
        # Transpose to [B, N, C] format for helper functions
        xyz_trans = xyz.permute(0, 2, 1)
        if points is not None:
            points_trans = points.permute(0, 2, 1)
        else:
            points_trans = None

        # 1. Sampling
        center_idx = farthest_point_sample(xyz_trans, self.npoint)
        new_xyz = index_points(xyz_trans, center_idx)  # [B, npoint, 3]
        
        # 2. Grouping
        group_idx = knn_point(self.k, xyz_trans, new_xyz)
        grouped_xyz = index_points(xyz_trans, group_idx)  # [B, npoint, K, 3]
        
        # Normalize coordinates relative to center points
        grouped_xyz_norm = grouped_xyz - new_xyz.view(xyz.shape[0], self.npoint, 1, 3)
        
        if points_trans is not None:
            grouped_points = index_points(points_trans, group_idx)
            # Concatenate features with normalized coordinates
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, K, 3+C_in]
        else:
            new_points = grouped_xyz_norm
            
        # Transpose to [B, C, N', K] for Conv2d
        new_points = new_points.permute(0, 3, 1, 2)
        
        # 3. PointNet Unit
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
            # 可选的SE层：只在第一个卷积后应用
            if i == 0 and self.use_se and len(self.mlp_convs) > 1:
                new_points = self.se(new_points)

        F_local = new_points  # [B, C_out, npoint, K]
        
        # 4. 可选的PC-CBAM Enhancement
        if self.use_pc_cbam:
            grouped_xyz_norm_permuted = grouped_xyz_norm.permute(0, 3, 1, 2)
            F_enhanced = self.pc_cbam(F_local, grouped_xyz_norm_permuted)
        else:
            F_enhanced = F_local

        # 5. Max Pooling
        new_points_aggregated = torch.max(F_enhanced, 3)[0]  # [B, C_out, npoint]

        # Transpose xyz back to [B, 3, npoint]
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points_aggregated

class EnhancedSALayer(nn.Module):
    def __init__(self, npoint, k, in_channel, mlp_channels):
        super(EnhancedSALayer, self).__init__()
        self.npoint = npoint
        self.k=k

        #pointnet unit SE\
        self.mlp_convs=nn.ModuleList()
        self.mlp_bns=nn.ModuleList()
        last_channel=in_channel+3#concatenate relative position xyz

        #definet mlp layers
        for i,out_channel in enumerate(mlp_channels):
            self.mlp_convs.append(nn.Conv2d(last_channel,out_channel,1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel=out_channel

        #SE layer
        self.se=SElayer(mlp_channels[0])

        self.pc_cbam=pc_CBAM_module(mlp_channels[-1])

    def forward(self, xyz, points):
        # xyz: [B, 3, N]
        # points: [B, C_in, N]
        
        # Transpose to [B, N, C] format for helper functions
        xyz_trans = xyz.permute(0, 2, 1)
        if points is not None:
            points_trans = points.permute(0, 2, 1)
        else:
            # Handle case where there are no input features
            points_trans = None

        # 1. Sampling
        center_idx = farthest_point_sample(xyz_trans, self.npoint)
        new_xyz = index_points(xyz_trans, center_idx) # [B, npoint, 3]
        
        # 2. Grouping
        group_idx = knn_point(self.k, xyz_trans, new_xyz)
        grouped_xyz = index_points(xyz_trans, group_idx) # [B, npoint, K, 3]
        
        # Normalize coordinates relative to center points
        grouped_xyz_norm = grouped_xyz - new_xyz.view(xyz.shape[0], self.npoint, 1, 3)
        
        if points_trans is not None:
            grouped_points = index_points(points_trans, group_idx)
            # Concatenate features with normalized coordinates
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, K, 3+C_in]
        else:
            new_points = grouped_xyz_norm
            
        # Transpose to [B, C, N', K] for Conv2d
        new_points = new_points.permute(0, 3, 1, 2)
        
        # 3. PointNet Unit (with SE)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
            # Apply SE after the first convolution
            if i == 0 and len(self.mlp_convs) > 1: # Only apply if there's more than one laye
                new_points = self.se(new_points)
        
        F_local = new_points # [B, C_out, npoint, K]
        
        # 4. PC-CBAM Enhancement
        # PC-CBAM needs coordinates, we pass the normalized ones
        grouped_xyz_norm_permuted = grouped_xyz_norm.permute(0, 3, 1, 2)
        F_cbam_out = self.pc_cbam(F_local, grouped_xyz_norm_permuted)

        # 5. Max Pooling
        new_points_aggregated = torch.max(F_cbam_out, 3)[0] # [B, C_out, npoint]

        # Transpose xyz back to [B, 3, npoint]
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points_aggregated

class CBAMOnlySALayer(nn.Module):
    """只使用PC-CBAM注意力机制的SA层，不使用SE"""
    def __init__(self, npoint, k, in_channel, mlp_channels):
        super(CBAMOnlySALayer, self).__init__()
        self.npoint = npoint
        self.k = k

        # PointNet unit (不使用SE)
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3  # concatenate relative position xyz

        # Define MLP layers
        for i, out_channel in enumerate(mlp_channels):
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        # 只使用PC-CBAM，不使用SE
        self.pc_cbam = pc_CBAM_module(mlp_channels[-1])

    def forward(self, xyz, points):
        # xyz: [B, 3, N]
        # points: [B, C_in, N]
        
        # Transpose to [B, N, C] format for helper functions
        xyz_trans = xyz.permute(0, 2, 1)
        if points is not None:
            points_trans = points.permute(0, 2, 1)
        else:
            points_trans = None

        # 1. Sampling
        center_idx = farthest_point_sample(xyz_trans, self.npoint)
        new_xyz = index_points(xyz_trans, center_idx)  # [B, npoint, 3]
        
        # 2. Grouping
        group_idx = knn_point(self.k, xyz_trans, new_xyz)
        grouped_xyz = index_points(xyz_trans, group_idx)  # [B, npoint, K, 3]
        
        # Normalize coordinates relative to center points
        grouped_xyz_norm = grouped_xyz - new_xyz.view(xyz.shape[0], self.npoint, 1, 3)
        
        if points_trans is not None:
            grouped_points = index_points(points_trans, group_idx)
            # Concatenate features with normalized coordinates
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, K, 3+C_in]
        else:
            new_points = grouped_xyz_norm
            
        # Transpose to [B, C, N', K] for Conv2d
        new_points = new_points.permute(0, 3, 1, 2)
        
        # 3. PointNet Unit (不使用SE)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        F_local = new_points  # [B, C_out, npoint, K]
        
        # 4. PC-CBAM Enhancement (只使用PC-CBAM)
        grouped_xyz_norm_permuted = grouped_xyz_norm.permute(0, 3, 1, 2)
        F_cbam_out = self.pc_cbam(F_local, grouped_xyz_norm_permuted)

        # 5. Max Pooling
        new_points_aggregated = torch.max(F_cbam_out, 3)[0]  # [B, C_out, npoint]

        # Transpose xyz back to [B, 3, npoint]
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points_aggregated

class BaselineSALayer(nn.Module):
    """不使用任何注意力机制的原始SA层"""
    def __init__(self, npoint, k, in_channel, mlp_channels):
        super(BaselineSALayer, self).__init__()
        self.npoint = npoint
        self.k = k

        # 原始PointNet unit（不使用任何注意力机制）
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3  # concatenate relative position xyz

        # Define MLP layers
        for i, out_channel in enumerate(mlp_channels):
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        # 不使用任何注意力机制

    def forward(self, xyz, points):
        # xyz: [B, 3, N]
        # points: [B, C_in, N]
        
        # Transpose to [B, N, C] format for helper functions
        xyz_trans = xyz.permute(0, 2, 1)
        if points is not None:
            points_trans = points.permute(0, 2, 1)
        else:
            points_trans = None

        # 1. Sampling
        center_idx = farthest_point_sample(xyz_trans, self.npoint)
        new_xyz = index_points(xyz_trans, center_idx)  # [B, npoint, 3]
        
        # 2. Grouping
        group_idx = knn_point(self.k, xyz_trans, new_xyz)
        grouped_xyz = index_points(xyz_trans, group_idx)  # [B, npoint, K, 3]
        
        # Normalize coordinates relative to center points
        grouped_xyz_norm = grouped_xyz - new_xyz.view(xyz.shape[0], self.npoint, 1, 3)
        
        if points_trans is not None:
            grouped_points = index_points(points_trans, group_idx)
            # Concatenate features with normalized coordinates
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, K, 3+C_in]
        else:
            new_points = grouped_xyz_norm
            
        # Transpose to [B, C, N', K] for Conv2d
        new_points = new_points.permute(0, 3, 1, 2)
        
        # 3. 原始PointNet Unit（不使用任何注意力机制）
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # 4. Max Pooling（不使用注意力机制）
        new_points_aggregated = torch.max(new_points, 3)[0]  # [B, C_out, npoint]

        # Transpose xyz back to [B, 3, npoint]
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points_aggregated