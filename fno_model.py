import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================== 一维 FNO 模块 (新增) ======================

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # 保留的傅里叶模态数量

        self.scale = (1 / (in_channels * out_channels))
        # 复数权重矩阵
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        # 复数乘法: (batch, in_channel, mode) x (in_channel, out_channel, mode) -> (batch, out_channel, mode)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # 1D FFT
        x_ft = torch.fft.rfft(x)

        # 输出缓冲区
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)

        # 只对低频模态进行卷积
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # 逆 FFT 回到物理空间
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width, in_channels=1, out_channels=1):
        super(FNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 输入层：投影到更高维通道 (输入: x + grid_coord, 所以是 in_channels + 1)
        self.fc0 = nn.Linear(in_channels + 1, self.width)

        # 谱卷积层
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)

        # 普通卷积分支
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)

        # 输出层
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x shape: (batch, nx, in_channels)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)  # 拼接坐标

        # 投影并调整维度: (batch, nx, width) -> (batch, width, nx)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        # 卷积块 1
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        # 卷积块 2
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        # 调整回: (batch, nx, width)
        x = x.permute(0, 2, 1)

        # 输出投影
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        # 生成 0-1 归一化坐标
        gridx = torch.linspace(0, 1, size_x, device=device)
        # 扩展维度以匹配 batch
        gridx = gridx.view(1, size_x, 1).repeat(batchsize, 1, 1)
        return gridx


# ====================== 二维 FNO 模块 (原有，稍作优化) ======================

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def complex_matmul(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, x.shape[1], x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.complex_matmul(x_ft[:, :, :self.modes1, :self.modes2],
                                                                       self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.complex_matmul(x_ft[:, :, -self.modes1:, :self.modes2],
                                                                        self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    # 修改：增加 in_channels 和 out_channels 参数以支持多变量
    def __init__(self, modes1, modes2, width, in_channels=1, out_channels=1):
        super(FNO2d, self).__init__()
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 输入包含物理量 + x坐标 + y坐标
        self.fc0 = nn.Linear(in_channels + 2, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)

    def get_grid(self, shape, device):
        b, nx, ny = shape[0], shape[1], shape[2]
        gx = torch.linspace(0, 1, nx, device=device).view(1, nx, 1, 1).expand(b, -1, ny, -1)
        gy = torch.linspace(0, 1, ny, device=device).view(1, 1, ny, 1).expand(b, nx, -1, -1)
        return torch.cat((gx, gy), dim=-1)
