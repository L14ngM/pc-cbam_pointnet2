import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation
from pc_cbam import CBAMOnlySALayer, ConfigurableSALayer
import torch


class get_model(nn.Module):
    def __init__(self, num_classes, input_features=12, use_se=True, use_pc_cbam=True):
        """
        PointNet2-CBAM模型，支持消融实验
        
        参数:
            num_classes: 分类类别数
            input_features: 输入特征维度 (默认12维)
            use_se: 是否使用SE注意力层
            use_pc_cbam: 是否使用PC-CBAM注意力层
        """
        super(get_model, self).__init__()

        # 记录配置用于调试
        self.use_se = use_se
        self.use_pc_cbam = use_pc_cbam

        # --- 使用可配置的SA层进行消融实验 ---
        # 注意：in_channel现在是附加特征的通道数，3维坐标会在层内自动拼接
        # 对于4维输入(Z+RGB)：input_features=4，RGB特征维度=3
        # 对于12维输入：input_features=12，附加特征维度=9
        if input_features == 4:
            # 4维输入：Z + RGB，RGB特征维度为3
            additional_feature_dim = 3
        else:
            # 标准情况：XYZ + 其他特征
            additional_feature_dim = input_features - 3
            
        # 修改SA层点数配置：2048 -> 512 -> 128 -> 32
        # 使用更保险的k值配置，避免过大的感受野导致训练不稳定
        self.sa1 = ConfigurableSALayer(npoint=4096, k=32, in_channel=additional_feature_dim, 
                                     mlp_channels=[64, 128], use_se=use_se, use_pc_cbam=use_pc_cbam)
        self.sa2 = ConfigurableSALayer(npoint=1024, k=48, in_channel=128, 
                                     mlp_channels=[128, 256], use_se=use_se, use_pc_cbam=use_pc_cbam)
        self.sa3 = ConfigurableSALayer(npoint=256, k=48, in_channel=256, 
                                     mlp_channels=[256, 512], use_se=use_se, use_pc_cbam=use_pc_cbam)
        self.sa4 = ConfigurableSALayer(npoint=16, k=24, in_channel=512, 
                                     mlp_channels=[512, 1024], use_se=use_se, use_pc_cbam=use_pc_cbam)

        # --- FP层的输入通道数需要根据SA层的输出重新计算 ---
        # fp4 输入: sa4_out(1024) + sa3_out(512) = 1536
        self.fp4 = PointNetFeaturePropagation(in_channel=1536, mlp=[512, 512])
        # fp3 输入: fp4_out(512) + sa2_out(256) = 768
        self.fp3 = PointNetFeaturePropagation(in_channel=768, mlp=[256, 256])
        # fp2 输入: fp3_out(256) + sa1_out(128) = 384
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        # fp1 输入: fp2_out(128) + l0_points(additional_feature_dim)
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + additional_feature_dim, mlp=[128, 128, 128])

        # --- 分割头保持不变 ---
        self.conv1 = nn.Conv1d(128, 128, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(p=0.4)  # 增加Dropout率从0.5到0.7
        self.conv2 = nn.Conv1d(128, num_classes, kernel_size=1)

    def forward(self, xyz_and_features):
        # --- 修改输入处理，清晰分离坐标和特征 ---
        # 检查输入维度
        input_dim = xyz_and_features.shape[1]
        
        if input_dim == 4:
            # 4维输入：Z + RGB
            # 构造3D坐标：XY设为0，Z使用实际值
            batch_size, _, num_points = xyz_and_features.shape
            z_coords = xyz_and_features[:, 0:1, :]  # Z坐标
            x_coords = torch.zeros_like(z_coords)   # X坐标设为0
            y_coords = torch.zeros_like(z_coords)   # Y坐标设为0
            l0_xyz = torch.cat([x_coords, y_coords, z_coords], dim=1)  # [B, 3, N]
            l0_points = xyz_and_features[:, 1:, :]  # RGB特征 [B, 3, N]
        else:
            # 标准情况：XYZ + 其他特征
            l0_xyz = xyz_and_features[:, :3, :]
            l0_points = xyz_and_features[:, 3:, :]  # 附加特征维度

        # --- 调用新的SA层 ---
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # --- FP层的调用需要对应修改 ---
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # fp1的跳层连接是l0_points
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        # --- 分割头保持不变 ---
        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x, l4_points

    def get_config_description(self):
        """获取当前模型配置的描述，用于日志记录"""
        config_parts = []
        if self.use_se and self.use_pc_cbam:
            config_parts.append("SE+PC-CBAM")
        elif self.use_se:
            config_parts.append("SE-only")
        elif self.use_pc_cbam:
            config_parts.append("PC-CBAM-only")
        else:
            config_parts.append("Baseline")
        
        return "_".join(config_parts)

# === 新增：消融实验的便捷工厂函数 ===
def get_model_full(num_classes, input_features=12):
    """完整模型：SE + PC-CBAM"""
    return get_model(num_classes, input_features, use_se=True, use_pc_cbam=True)

def get_model_se_only(num_classes, input_features=12):
    """仅SE模型：SE + 无PC-CBAM"""
    return get_model(num_classes, input_features, use_se=True, use_pc_cbam=False)

def get_model_cbam_only(num_classes, input_features=12):
    """仅PC-CBAM模型：无SE + PC-CBAM"""
    return get_model(num_classes, input_features, use_se=False, use_pc_cbam=True)

def get_model_baseline(num_classes, input_features=12):
    """基线模型：无SE + 无PC-CBAM"""
    return get_model(num_classes, input_features, use_se=False, use_pc_cbam=False)

class Loss(nn.Module):
    def __init__(self, loss_type='cross_entropy', weight=None, gamma=2.0):
        super(Loss, self).__init__()
        self.loss_type = loss_type
        self.weight = weight
        self.gamma = gamma

    def forward(self, pred, target):
        if self.loss_type in ['nll_loss', 'cross_entropy']:
            pred = pred.view(-1, pred.shape[-1])  # [batch_size * num_points, num_classes]
            target = target.view(-1)
            loss = F.cross_entropy(pred, target, weight=self.weight)
        elif self.loss_type == 'focal_loss':
            loss = self.focal_loss(pred, target, weight=self.weight, gamma=self.gamma)
        elif self.loss_type == 'dice_loss':
            loss = self.dice_loss(pred, target)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        return loss

    def focal_loss(self, inputs, targets, weight=None, gamma=2.0, reduction='mean'):
        inputs = inputs.view(-1, inputs.shape[-1])
        targets = targets.view(-1)
        ce_loss = F.cross_entropy(inputs, targets, weight=weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** gamma) * ce_loss
        if reduction == 'mean':
            return focal_loss.mean()
        elif reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    def dice_loss(self, inputs, targets, smooth=1e-6):
        inputs = F.softmax(inputs, dim=2)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[2]).float()
        intersection = torch.sum(inputs * targets_one_hot, dim=(1, 2))
        union = torch.sum(inputs, dim=(1, 2)) + torch.sum(targets_one_hot, dim=(1, 2))
        dice_coeff = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice_coeff
        return dice_loss.mean()

def get_loss(loss_type='cross_entropy', weight=None, gamma=2.0):
    return Loss(loss_type=loss_type, weight=weight, gamma=gamma)


if __name__ == '__main__':
    import torch
    print("--- 正在测试 PointNet2-CBAM 模型 ---")
    
    test_configs = {
        "SE+PC-CBAM (Full)": {"use_se": True, "use_pc_cbam": True},
        "SE-Only": {"use_se": True, "use_pc_cbam": False},
        "PC-CBAM-Only": {"use_se": False, "use_pc_cbam": True},
        "Baseline (No Attention)": {"use_se": False, "use_pc_cbam": False},
    }
    
    for name, config in test_configs.items():
        print(f"\n--- 测试配置: {name} ---")
        try:
            model = get_model(num_classes=2, input_features=12, **config)
            
            # [Batch, Features, Num_Points]
            test_input = torch.rand(2, 12, 2048)
            
            output, features = model(test_input)
            
            print(f"  - 输入形状: {test_input.shape}")
            print(f"  - 输出形状 (Logits): {output.shape}")
            print(f"  - 特征形状 (Deepest): {features.shape}")
            
            # 验证输出形状
            assert output.shape == (2, 2048, 2)
            assert features.shape[0] == 2
            assert features.shape[1] > 0
            
            print(f"  ✅ 模型测试通过!")
            
        except Exception as e:
            print(f"  ❌ 模型测试失败: {e}")