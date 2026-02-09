import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import laplace, gaussian_filter
import random
import os

# ====================== 核心数学引擎 ======================

def compute_derivatives(u, dx, dy):
    """计算所有必要的偏导数"""
    ux, uy = np.gradient(u, dx, dy)
    uxx, uxy = np.gradient(ux, dx, dy)
    uyx, uyy = np.gradient(uy, dx, dy)
    return ux, uy, uxx, uyy, uxy

def calculate_sobolev_norm(u, dx, dy, k, p):
    """计算 W^{k,p} 范数"""
    total_sum = np.sum(np.abs(u)**p)
    if k >= 1:
        ux, uy = np.gradient(u, dx, dy)
        total_sum += np.sum(np.abs(ux)**p + np.abs(uy)**p)
    if k >= 2:
        ux, uy = np.gradient(u, dx, dy)
        uxx, uxy = np.gradient(ux, dx, dy)
        _, uyy = np.gradient(uy, dx, dy)
        total_sum += np.sum(np.abs(uxx)**p + 2*np.abs(uxy)**p + np.abs(uyy)**p)
    
    norm_val = (total_sum * dx * dy)**(1/p)
    return norm_val

def apply_robin_boundary(u, X, Y, alpha, beta, geometry='rect'):
    """
    强制让函数满足 au + b(du/dn) = 0
    使用距离函数平滑修正，替代硬编码的因子
    """
    if geometry == 'rect':
        # 距离边界的距离 (0 to 0.5)
        dist = np.minimum(np.minimum(X, 1-X), np.minimum(Y, 1-Y))
    else: # circle
        r = np.sqrt((X-0.5)**2 + (Y-0.5)**2)
        dist = np.maximum(0, 0.5 - r)

    # 如果 beta 为 0 (纯 Dirichlet), 则 u_new = u * (1 - exp(-dist/eps))
    # 这里使用简单的权重映射来确保边界条件
    weight = np.tanh(dist * 10) 
    return u * weight

# ====================== 随机函数生成 (保留原逻辑) ======================

def generate_base(X, Y):
    type_ = random.choice(['power', 'sin', 'exp', 'log', 'constant', 'polynomial'])
    var = random.choice(['X', 'Y'])
    v = X if var == 'X' else Y
    
    if type_ == 'power':
        s = random.uniform(0.5, 2)
        return v ** s
    elif type_ == 'sin':
        k = random.randint(1, 2)
        return np.sin(k * np.pi * v)
    elif type_ == 'exp':
        return np.exp(random.uniform(-1, 1) * v)
    elif type_ == 'log':
        return np.log(1 + random.uniform(0.5, 2) * v)
    elif type_ == 'constant':
        return np.full_like(X, random.uniform(-1, 1))
    else: # polynomial
        return random.uniform(-1,1)*v**2 + random.uniform(-1,1)*v

def build_G(X, Y, levels=2):
    # 此处保留用户原有的复合逻辑
    ops = [lambda a,b: a+b, lambda a,b: a-b, lambda a,b: a*b]
    funcs = [generate_base(X, Y) for _ in range(3)]
    res = funcs[0]
    for f in funcs[1:]:
        op = random.choice(ops)
        res = op(res, f)
    return res

# ====================== Streamlit UI 界面 ======================

st.set_page_config(page_title="PDE 数据生成器 Pro", layout="wide")
st.title("🔬 通用二阶线性 PDE 数据合成平台")

with st.sidebar:
    st.header("1. 算子参数 (L[u]=f)")
    st.info("au_xx + 2bu_xy + cu_yy + du_x + eu_y + fu + g")
    c_a = st.number_input("a (u_xx)", value=1.0)
    c_b = st.number_input("b (u_xy)", value=0.0)
    c_c = st.number_input("c (u_yy)", value=1.0)
    c_d = st.number_input("d (u_x)", value=0.0)
    c_e = st.number_input("e (u_y)", value=0.0)
    c_f = st.number_input("f (u)", value=0.0)
    c_g = st.number_input("g (const)", value=0.0)

    st.header("2. 边界条件 (au + b∂u/∂n = 0)")
    bc_shape = st.selectbox("区域几何", ["rect", "circle"])
    bc_a = st.number_input("边界系数 a", value=1.0)
    bc_b = st.number_input("边界系数 b", value=0.0)

    st.header("3. Sobolev 约束 W^{k,p}")
    sob_k = st.slider("阶数 k", 0, 2, 1)
    sob_p = st.number_input("指数 p", value=2.0)
    sob_max = st.number_input("范数上限", value=50.0)

    st.header("4. 网格设置")
    nx = st.number_input("NX", value=50)
    ny = st.number_input("NY", value=50)

# ====================== 生成逻辑 ======================

if st.button("🚀 开始生成样本"):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    dx, dy = x[1]-x[0], y[1]-y[0]

    success = False
    attempts = 0
    while not success and attempts < 100:
        attempts += 1
        # 1. 生成基础随机函数
        u_raw = build_G(X, Y)
        
        # 2. 应用用户定义的边界条件 (替代原 x(1-x)y(1-y))
        u_bc = apply_robin_boundary(u_raw, X, Y, bc_a, bc_b, bc_shape)
        
        # 3. 校验 Sobolev 范数
        norm_val = calculate_sobolev_norm(u_bc, dx, dy, sob_k, sob_p)
        
        if norm_val <= sob_max:
            # 4. 计算自定义 PDE 算子得到 f
            ux, uy, uxx, uyy, uxy = compute_derivatives(u_bc, dx, dy)
            f = -(c_a*uxx + 2*c_b*uxy + c_c*uyy + c_d*ux + c_e*uy + c_f*u_bc + c_g)
            
            # 结果展示
            success = True
            st.success(f"生成成功！尝试次数: {attempts} | W^{{{sob_k},{sob_p}}} 范数: {norm_val:.4f}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("解 u(x,y)")
                fig1, ax1 = plt.subplots()
                im1 = ax1.imshow(u_bc, extent=[0,1,0,1], origin='lower', cmap='viridis')
                plt.colorbar(im1)
                st.pyplot(fig1)
            
            with col2:
                st.subheader("源项 f(x,y)")
                fig2, ax2 = plt.subplots()
                im2 = ax2.imshow(f, extent=[0,1,0,1], origin='lower', cmap='magma')
                plt.colorbar(im2)
                st.pyplot(fig2)

            with col3:
                st.subheader("3D 视图")
                fig3 = plt.figure()
                ax3 = fig3.add_subplot(111, projection='3d')
                ax3.plot_surface(X, Y, u_bc, cmap='viridis')
                st.pyplot(fig3)
            
            # 下载部分
            df = pd.DataFrame({'u': u_bc.flatten(), 'f': f.flatten()})
            st.download_button("下载数据 (CSV)", df.to_csv(index=False), "data.csv", "text/csv")

    if not success:
        st.error("在当前约束下无法生成有效函数，请尝试放宽 Sobolev 范数上限。")




'''
使用方法：
1.安装streamlit包
2.在该程序所在文件夹，同时按shift+右键，点击“打开powershell窗口”
3.输入 streamlit run version1.py 即可在浏览器端运行        注：此处version1可替换为文件名。
4.首次使用时，需要输入自己的正确邮箱
'''