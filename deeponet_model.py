import torch
import torch.nn as nn


class DeepONet(nn.Module):
    def __init__(self, n_sensors, hidden_dim, p, coord_dim=2, out_channels=1):
        """
        Args:
            n_sensors: 输入函数的采样点数 (对于1D是nx，对于2D是nx*ny)
            hidden_dim: 隐藏层维度
            p: 特征编码维度
            coord_dim: 坐标维度 (1 for 1D problems, 2 for 2D problems)
            out_channels: 输出变量个数 (1 for scalar, >1 for multi-variate)
        """
        super(DeepONet, self).__init__()
        self.out_channels = out_channels
        self.p = p

        # Branch 网络：编码输入函数 f
        # 输入: [batch, n_sensors]
        self.branch = nn.Sequential(
            nn.Linear(n_sensors, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, p * out_channels)  # 修改：输出维度扩展，支持多变量
        )

        # Trunk 网络：编码查询坐标
        # 输入: [batch, n_points, coord_dim]
        self.trunk = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),  # 修改：输入维度由 coord_dim 决定
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, p * out_channels)  # 修改：输出维度扩展
        )

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, f_obs, grid_coords):
        """
        Args:
            f_obs: 输入函数观测值 [batch, n_sensors]
            grid_coords: 查询点坐标
                         1D: [batch, nx, 1]
                         2D: [batch, nx*ny, 2]
        Returns:
            output:
                1D: [batch, nx, out_channels]
                2D: [batch, nx*ny, out_channels]
        """
        # Branch: [batch, p * out_channels]
        B = self.branch(f_obs)

        # Trunk: [batch, n_points, p * out_channels]
        T = self.trunk(grid_coords)

        # 维度重塑以便计算内积
        # B reshape为: [batch, out_channels, p]
        B = B.view(-1, self.out_channels, self.p)

        # T reshape为: [batch, n_points, out_channels, p]
        T = T.view(-1, T.shape[1], self.out_channels, self.p)

        # 计算内积: 在 p 维度上求和
        # result: [batch, n_points, out_channels]
        # 公式: sum(B * T)
        output = torch.einsum("bop,bnop->bno", B, T) + self.bias

        return output
