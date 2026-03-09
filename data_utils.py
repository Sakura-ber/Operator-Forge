import numpy as np
import random


# ====================== 原有二维辅助函数 ======================

def compute_derivatives(u, dx, dy):
    """计算二维导数"""
    ux, uy = np.gradient(u, dx, dy)
    uxx, uxy = np.gradient(ux, dx, dy)
    _, uyy = np.gradient(uy, dx, dy)
    return ux, uy, uxx, uyy, uxy


def calculate_sobolev_norm(u, dx, dy, k, p):
    """计算二维 Sobolev 范数"""
    total_sum = np.sum(np.abs(u) ** p)
    if k >= 1:
        ux, uy = np.gradient(u, dx, dy)
        total_sum += np.sum(np.abs(ux) ** p + np.abs(uy) ** p)
    if k >= 2:
        ux, uy = np.gradient(u, dx, dy)
        uxx, uxy = np.gradient(ux, dx, dy)
        _, uyy = np.gradient(uy, dx, dy)
        total_sum += np.sum(np.abs(uxx) ** p + 2 * np.abs(uxy) ** p + np.abs(uyy) ** p)
    norm_val = (total_sum * dx * dy) ** (1 / p)
    return norm_val


def apply_robin_boundary(u, X, Y, alpha, beta, geometry='rect'):
    """应用 Robin 边界条件"""
    if geometry == 'rect':
        dist = np.minimum(np.minimum(X, 1 - X), np.minimum(Y, 1 - Y))
    else:
        r = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2)
        dist = np.maximum(0, 0.5 - r)
    weight = np.tanh(dist * 10)
    return u * weight


def generate_base(X, Y):
    """生成二维基础函数"""
    type_ = random.choice(['power', 'sin', 'exp', 'log', 'constant', 'polynomial'])
    var = random.choice(['X', 'Y'])
    v = X if var == 'X' else Y
    if type_ == 'power':
        return v ** random.uniform(0.5, 2)
    elif type_ == 'sin':
        return np.sin(random.randint(1, 2) * np.pi * v)
    elif type_ == 'exp':
        return np.exp(random.uniform(-0.5, 0.5) * v)
    elif type_ == 'log':
        return np.log(1.1 + random.uniform(0.1, 1) * v)
    elif type_ == 'constant':
        return np.full_like(X, random.uniform(-1, 1))
    else:
        return random.uniform(-0.5, 0.5) * v ** 2 + random.uniform(-0.5, 0.5) * v


def build_G(X, Y):
    """构建二维随机场"""
    ops = [lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b]
    funcs = [generate_base(X, Y) for _ in range(3)]
    res = funcs[0]
    for f in funcs[1:]:
        op = random.choice(ops)
        res = op(res, f)
    return res


# ====================== 新增：一维 Burgers 方程相关函数 ======================

def compute_derivatives_1d(u, dx):
    """计算一维导数"""
    ux = np.gradient(u, dx)
    uxx = np.gradient(ux, dx)
    return ux, uxx


def generate_ic_1d(nx, L=2 * np.pi, max_modes=5):
    """
    生成一维随机初始条件
    Args:
        nx: 网格点数
        L: 计算域长度，默认 2*pi
        max_modes: 傅里叶模态数
    Returns:
        u0: 初始场
        x: 空间坐标
    """
    x = np.linspace(0, L, nx, endpoint=False)
    u0 = np.zeros(nx)

    # 叠加随机傅里叶模态
    for k in range(1, max_modes + 1):
        coef_a = np.random.normal(0, 1) / (k ** 0.5)  # 随机振幅
        coef_b = np.random.normal(0, 1) / (k ** 0.5)  # 随机相位相关系数
        u0 += coef_a * np.sin(2 * np.pi * k * x / L) + \
              coef_b * np.cos(2 * np.pi * k * x / L)

    # 归一化处理，控制初始场的幅度
    if np.max(np.abs(u0)) > 1e-6:
        u0 = u0 / np.max(np.abs(u0)) * np.random.uniform(0.5, 2.0)
    return u0, x


def burg_rhs_spectral(u, k, nu, dealiasing_mask):
    """
    谱方法计算右端项: du/dt = -u * du/dx + nu * d^2u/dx^2
    Args:
        u: 当前物理空间速度场
        k: 波数数组
        nu: 粘性系数
        dealiasing_mask: 抗混叠掩膜
    Returns:
        du_dt: 时间导数
    """
    # 1. 计算导数 (谱空间 -> 物理空间)
    u_hat = np.fft.fft(u)

    # 计算一阶导数 u_x
    ux_hat = 1j * k * u_hat
    ux = np.real(np.fft.ifft(ux_hat))

    # 计算二阶导数 u_xx (用于粘性项)
    uxx_hat = -k ** 2 * u_hat

    # 2. 计算非线性项 -u * u_x (物理空间)
    # 应用抗混叠
    u_filtered = np.real(np.fft.ifft(u_hat * dealiasing_mask))
    ux_filtered = np.real(np.fft.ifft(ux_hat * dealiasing_mask))

    convection = -u_filtered * ux_filtered

    # 3. 组合
    # 粘性项在谱空间计算: nu * d^2u/dx^2 -> -nu * k^2 * u_hat
    # 转换回物理空间
    diffusion = np.real(np.fft.ifft(nu * uxx_hat))

    return convection + diffusion


def solve_burgers_1d(u0, x, nu, T_final, dt=None):
    """
    一维 Burgers 方程求解器 (伪谱法 + RK4)
    Args:
        u0: 初始条件
        x: 空间坐标
        nu: 粘性系数
        T_final: 终止时间
        dt: 时间步长 (可选，默认自动计算)
    Returns:
        u_T: 终止时刻的解
    """
    nx = len(x)
    L = x[-1] - x[0] + (x[1] - x[0])
    dx = L / nx

    # 波数设置 (用于 FFT)
    k = np.fft.fftfreq(nx, d=dx) * 2 * np.pi

    # 抗混叠设置 (2/3法则)
    kmax = nx // 3
    dealiasing_mask = np.ones(nx)
    dealiasing_mask[kmax:nx - kmax] = 0

    # 自动计算时间步长 (基于 CFL 条件)
    if dt is None:
        cfl_limit = 0.5
        u_max = np.max(np.abs(u0)) + 1e-6
        dt_adv = cfl_limit * dx / u_max  # 对流限制
        dt_dif = cfl_limit * dx ** 2 / (nu + 1e-8)  # 扩散限制
        dt = min(dt_adv, dt_dif, 0.01)  # 设定上限防止过长

    # 无粘 Burgers 特殊处理：
    # 如果 nu=0，数值上必须有极微小的耗散来维持稳定
    current_nu = nu
    if current_nu == 0:
        current_nu = 1e-6  # 添加极小的人工粘性以保证数值稳定性

    # RK4 时间步进
    u = u0.copy()
    num_steps = int(T_final / dt)

    for _ in range(num_steps):
        # RK4 Steps
        k1 = burg_rhs_spectral(u, k, current_nu, dealiasing_mask)
        k2 = burg_rhs_spectral(u + 0.5 * dt * k1, k, current_nu, dealiasing_mask)
        k3 = burg_rhs_spectral(u + 0.5 * dt * k2, k, current_nu, dealiasing_mask)
        k4 = burg_rhs_spectral(u + dt * k3, k, current_nu, dealiasing_mask)

        u = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return u


# ====================== 新增：多变量系统辅助函数 ======================

def generate_multivar_poisson_data(X, Y, dx, c_a, c_c, num_vars):
    """
    生成多变量泊松方程数据
    Args:
        X, Y: 网格坐标
        dx: 空间步长
        c_a, c_c: 方程系数
        num_vars: 变量个数 (如 2 或 4)
    Returns:
        f_list: 源项列表，shape (num_vars, NX, NY)
        u_list: 解列表，shape (num_vars, NX, NY)
    """
    f_list, u_list = [], []

    for _ in range(num_vars):
        # 复用已有的单变量生成逻辑
        u_raw = build_G(X, Y)
        u_bc = apply_robin_boundary(u_raw, X, Y, 1, 0, 'rect')

        # 简单筛选，如果 Sobolev 范数过大则重新生成
        # 这里为了演示简化处理，实际应用可加严格筛选
        if calculate_sobolev_norm(u_bc, dx, dx, 1, 2.0) > 100.0:
            u_raw = build_G(X, Y)
            u_bc = apply_robin_boundary(u_raw, X, Y, 1, 0, 'rect')

        ux, uy, uxx, uyy, _ = compute_derivatives(u_bc, dx, dx)
        f = -(c_a * uxx + c_c * uyy)

        u_list.append(u_bc)
        f_list.append(f)

    return np.stack(f_list, axis=0), np.stack(u_list, axis=0)
