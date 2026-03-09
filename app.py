import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import io
import data_utils as du
# 引入 FNO1d 和 FNO2d
from fno_model import FNO2d, FNO1d
from deeponet_model import DeepONet

# ====================== 设备辅助函数 ======================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

st.set_page_config(page_title="PDE AI Platform", layout="wide")

# ====================== Session State 初始化 ======================
# 初始化所有必须的状态，防止 Key Error
if 'model' not in st.session_state: st.session_state.model = None
if 'pde_type' not in st.session_state: st.session_state.pde_type = None
if 'arch' not in st.session_state: st.session_state.arch = None
if 'nx' not in st.session_state: st.session_state.nx = None
# 兼容旧版本 state，如果不存在则设为默认值
if 'is_1d' not in st.session_state: st.session_state.is_1d = False
if 'num_vars' not in st.session_state: st.session_state.num_vars = 1

st.sidebar.title("🛠️ 导航与配置")
page = st.sidebar.radio("选择功能", ["1. 数据生成", "2. 模型训练", "3. 预测推理"])

# ====================== 1. 数据生成 ======================
if page == "1. 数据生成":
    st.title("🧬 数据合成")
    
    # 方程类型选择
    pde_options = ["二维泊松方程", "一维Burgers方程", "多变量系统 (2变量)", "多变量系统 (4变量)"]
    pde_type = st.sidebar.selectbox("选择方程类型", pde_options)
    st.session_state.pde_type = pde_type
    
    # 解析变量数
    num_vars = 1
    if "2变量" in pde_type: num_vars = 2
    elif "4变量" in pde_type: num_vars = 4
    st.session_state.num_vars = num_vars
    
    with st.sidebar:
        st.header("通用参数")
        nx = st.number_input("网格点数 NX", value=50)
        num_samples = st.number_input("样本数", value=100)
        
        if pde_type == "二维泊松方程":
            st.header("算子参数 (Lu=f)")
            c_a = st.number_input("a (u_xx)", value=1.0)
            c_c = st.number_input("c (u_yy)", value=1.0)
            sob_max = st.number_input("Sobolev 范数上限", value=50.0)
            
        elif pde_type == "一维Burgers方程":
            st.header("Burgers 参数")
            nu = st.number_input("粘性系数", value=0.1, format="%.4f")
            T_final = st.number_input("演化时间 T", value=1.0, format="%.2f")
            
        elif "多变量系统" in pde_type:
            st.header("系统参数")
            st.info(f"当前变量数: {num_vars}")
            c_a = st.number_input("系数 a", value=1.0)
            c_c = st.number_input("系数 c", value=1.0)

    if st.button("🚀 生成数据"):
        progress = st.progress(0)
        status = st.empty()
        
        # 逻辑 1: 一维 Burgers
        if pde_type == "一维Burgers方程":
            u_list, f_list = [], []
            x_pts = np.linspace(0, 2*np.pi, nx)
            for i in range(num_samples):
                u0, _ = du.generate_ic_1d(nx)
                uT = du.solve_burgers_1d(u0, x_pts, nu, T_final)
                f_list.append(u0); u_list.append(uT)
                progress.progress((i+1)/num_samples)
            buf = io.BytesIO()
            np.savez_compressed(buf, u=np.array(u_list), f=np.array(f_list), nx=nx)
            st.success("一维 Burgers 数据生成完成！")
            st.download_button("📥 下载 .npz", buf.getvalue(), "burgers_data.npz")

        # 逻辑 2 & 3: 二维泊松及多变量系统
        else:
            x_pts = np.linspace(0, 1, nx)
            X, Y = np.meshgrid(x_pts, x_pts)
            dx = x_pts[1]-x_pts[0]
            
            all_u, all_f = [], []
            found_count = 0
            
            # 循环生成
            while found_count < num_samples:
                if "多变量" in pde_type:
                    f_batch, u_batch = du.generate_multivar_poisson_data(X, Y, dx, c_a, c_c, num_vars)
                    all_f.append(f_batch.transpose(1, 2, 0))
                    all_u.append(u_batch.transpose(1, 2, 0))
                    found_count += 1
                    progress.progress(found_count / num_samples)
                else:
                    # 单变量
                    u_raw = du.build_G(X, Y)
                    u_bc = du.apply_robin_boundary(u_raw, X, Y, 1, 0, 'rect')
                    if du.calculate_sobolev_norm(u_bc, dx, dx, 1, 2.0) <= sob_max:
                        ux, uy, uxx, uyy, _ = du.compute_derivatives(u_bc, dx, dx)
                        f = -(c_a*uxx + c_c*uyy)
                        all_u.append(u_bc); all_f.append(f)
                        found_count += 1
                        progress.progress(found_count / num_samples)
            
            buf = io.BytesIO()
            np.savez_compressed(buf, u=np.array(all_u), f=np.array(all_f), nx=nx)
            st.success(f"{pde_type} 数据生成完成！")
            fname = f"multi_{num_vars}var.npz" if num_vars > 1 else "poisson_data.npz"
            st.download_button("📥 下载 .npz", buf.getvalue(), fname)

# ====================== 2. 模型训练 ======================
elif page == "2. 模型训练":
    st.title("🚂 训练配置")
    with st.sidebar:
        st.header("🍀 学习超参数")
        lr = st.number_input("学习率", value=1e-3, format="%.4f")
        bs = st.number_input("Batch Size", value=16)
        epochs = st.number_input("Epochs", value=50)
        st.divider()
        arch = st.selectbox("架构", ["FNO", "DeepONet"])
        
        if arch == "FNO":
            m1 = st.slider("Modes", 4, 32, 12)
            w = st.slider("Width", 16, 128, 32)
        else:
            hd = st.slider("Hidden Dim", 64, 512, 128)
            p_dim = st.slider("p Dim", 32, 256, 100)

    file = st.file_uploader("上传数据", type="npz")
     
    if file and st.button("开始训练"):
        data = np.load(file)
        f_data = data['f']
        u_data = data['u']
        nx = int(data['nx'])
        
        # 自动检测维度与通道数
        is_1d = False
        in_channels = 1
        out_channels = 1
        
        if f_data.ndim == 2: # 1D Data: (N, nx)
            is_1d = True
            f_t = torch.FloatTensor(f_data).unsqueeze(-1) # (N, nx, 1)
            u_t = torch.FloatTensor(u_data).unsqueeze(-1)
        elif f_data.ndim == 3: # 2D Single Var: (N, nx, ny)
            f_t = torch.FloatTensor(f_data).unsqueeze(-1) # (N, nx, ny, 1)
            u_t = torch.FloatTensor(u_data).unsqueeze(-1)
        elif f_data.ndim == 4: # 2D Multi Var: (N, nx, ny, C)
            in_channels = f_data.shape[-1]
            out_channels = u_data.shape[-1]
            st.info(f"检测到多变量数据: 输入通道={in_channels}, 输出通道={out_channels}")
            f_t = torch.FloatTensor(f_data)
            u_t = torch.FloatTensor(u_data)
        else:
            st.error("数据维度不支持！")
            st.stop()

        loader = DataLoader(TensorDataset(f_t, u_t), batch_size=int(bs), shuffle=True)
       
        # 设备设置
        device = get_device()
        st.info(f"当前计算设备: {device}")
       
        # 模型初始化
        if arch == "FNO":
            if is_1d:
                model = FNO1d(modes=m1, width=w, in_channels=in_channels, out_channels=out_channels)
            else:
                model = FNO2d(m1, m1, w, in_channels=in_channels, out_channels=out_channels)
        else: # DeepONet
            sensor_size = nx if is_1d else nx*nx
            coord_dim = 1 if is_1d else 2
            model = DeepONet(n_sensors=sensor_size*in_channels, hidden_dim=hd, p=p_dim, 
                             coord_dim=coord_dim, out_channels=out_channels)

        model.to(device)
       
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
       
        epoch_losses = []
        p_bar, status = st.progress(0), st.empty()
        
        # 训练循环变量，用于后续可视化
        last_bu = None
        last_pred = None
       
        for epoch in range(int(epochs)):
            loss_list = []
            for bf, bu in loader:
                bf, bu = bf.to(device), bu.to(device)
               
                optimizer.zero_grad()
                
                if arch == "FNO":
                    pred = model(bf)
                else: # DeepONet
                    batch_size = bf.size(0)
                    if is_1d:
                        coords = torch.linspace(0, 1, nx, device=device).view(1, nx, 1).repeat(batch_size, 1, 1)
                    else:
                        coords = torch.stack(torch.meshgrid(torch.linspace(0,1,nx), torch.linspace(0,1,nx), indexing='ij'), -1).reshape(-1,2).unsqueeze(0).repeat(batch_size,1,1)
                        coords = coords.to(device)
                    
                    pred = model(bf.view(batch_size, -1), coords)
                    
                    if is_1d:
                        pred = pred.view(-1, nx, out_channels) 
                    else:
                        pred = pred.view(-1, nx, nx, out_channels)
               
                loss = criterion(pred, bu)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
                
                # 保存最后一个batch用于可视化
                last_bu = bu
                last_pred = pred
           
            avg_loss = np.mean(loss_list)
            epoch_losses.append(avg_loss)
            p_bar.progress((epoch+1)/int(epochs))
            status.text(f"Epoch {epoch+1}/{int(epochs)} | Loss: {avg_loss:.6f}")
       
        # 保存模型 (移回CPU以便存储)
        st.session_state.model = model.cpu()
        st.session_state.arch = arch
        st.session_state.nx = nx
        st.session_state.is_1d = is_1d
        st.session_state.num_vars = out_channels
        st.success("训练成功！")
        
        # ====================== 结果可视化 ======================
        st.subheader("📊 学习结果分析")
        c1, c2 = st.columns(2)

        with c1:
            st.metric("最终损失", f"{epoch_losses[-1]:.6f}")
            fig_loss, ax = plt.subplots()
            ax.plot(epoch_losses, color='royalblue')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.set_title('Training Loss Curve')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig_loss)

        with c2:
            st.metric("相对误差", f"{np.sqrt(epoch_losses[-1]):.4f} (RMSE)")
            st.info("展示最后一个 Batch 的拟合效果对比")
            
            # 安全处理可视化数据
            if last_bu is not None and last_pred is not None:
                with torch.no_grad():
                    # 1. 确保数据移回 CPU
                    true_u = last_bu[0].cpu().squeeze().numpy()
                    pred_u = last_pred[0].cpu().squeeze().numpy()
                    
                    # 2. 根据维度和变量数绘图
                    if is_1d:
                        fig_cmp, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(true_u, label='Ground Truth', linewidth=2)
                        ax.plot(pred_u, '--', label='Prediction', linewidth=2)
                        ax.set_title("1D Fitting Result")
                        ax.legend(); ax.grid(True, alpha=0.3)
                        st.pyplot(fig_cmp)
                        
                    elif out_channels > 1:
                        # 多变量：显示第一个变量
                        st.warning(f"多变量模式：下方仅展示第 1 个变量的对比 (共 {out_channels} 个变量)")
                        fig_cmp, axes = plt.subplots(1, 3, figsize=(12, 4))
                        
                        # 防止 squeeze 过度降维
                        true_2d = true_u[..., 0] if true_u.ndim > 2 else true_u
                        pred_2d = pred_u[..., 0] if pred_u.ndim > 2 else pred_u

                        im0 = axes[0].imshow(true_2d, origin='lower')
                        axes[0].set_title("Ground Truth (Var 1)")
                        plt.colorbar(im0, ax=axes[0], fraction=0.046)

                        im1 = axes[1].imshow(pred_2d, origin='lower')
                        axes[1].set_title("Prediction (Var 1)")
                        plt.colorbar(im1, ax=axes[1], fraction=0.046)

                        im2 = axes[2].imshow(np.abs(true_2d - pred_2d), cmap='hot', origin='lower')
                        axes[2].set_title("Absolute Error (Var 1)")
                        plt.colorbar(im2, ax=axes[2], fraction=0.046)

                        plt.tight_layout()
                        st.pyplot(fig_cmp)
                        
                    else:
                        # 单变量 2D
                        fig_cmp, axes = plt.subplots(1, 3, figsize=(12, 4))
                        
                        im0 = axes[0].imshow(true_u, origin='lower')
                        axes[0].set_title("Ground Truth")
                        plt.colorbar(im0, ax=axes[0], fraction=0.046)

                        im1 = axes[1].imshow(pred_u, origin='lower')
                        axes[1].set_title("Prediction")
                        plt.colorbar(im1, ax=axes[1], fraction=0.046)

                        im2 = axes[2].imshow(np.abs(true_u - pred_u), cmap='hot', origin='lower')
                        axes[2].set_title("Absolute Error")
                        plt.colorbar(im2, ax=axes[2], fraction=0.046)

                        plt.tight_layout()
                        st.pyplot(fig_cmp)

# ====================== 3. 预测推理 ======================
elif page == "3. 预测推理":
    st.title("🔮 解析式预测")
    
    if st.session_state.model:
        model = st.session_state.model
        arch = st.session_state.arch
        nx = st.session_state.nx
        # 安全获取状态，兼容旧版本
        is_1d = st.session_state.get('is_1d', False)
        num_vars = st.session_state.get('num_vars', 1)
        
        device = get_device()
        model.to(device) # 推理时移回设备

        help_text = "多变量模式：输入第一个变量的表达式。" if num_vars > 1 else ""
        if is_1d:
            expr = st.text_input("输入 u0(x) 解析式 (变量: x)", "np.sin(x)", help=help_text)
        else:
            expr = st.text_input("输入 f(X, Y) 解析式", "np.sin(np.pi*X)*np.sin(np.pi*Y)", help=help_text)

        if st.button("开始预测"):
            try:
                with torch.no_grad():
                    if is_1d:
                        # 1D 推理
                        x = np.linspace(0, 2*np.pi, nx)
                        val = eval(expr)
                        # 构造多通道输入 (如果是多变量)
                        f_val = np.stack([val]*num_vars, axis=-1)
                        f_in = torch.FloatTensor(f_val).unsqueeze(0).to(device) # (1, nx, C)
                        
                        if arch == "FNO":
                            u_pred = model(f_in).squeeze().cpu().numpy()
                        else:
                            coords = torch.linspace(0, 1, nx, device=device).view(1, nx, 1)
                            u_pred = model(f_in.view(1,-1), coords).squeeze().cpu().numpy()
                        
                        # 绘图
                        st.subheader("预测结果")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(x, f_val[..., 0], label='Input u0', alpha=0.5, linestyle='--')
                        ax.plot(x, u_pred, label='Prediction', linewidth=2)
                        ax.set_xlabel('x'); ax.set_ylabel('u')
                        ax.legend(); ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                    else:
                        # 2D 推理
                        x_range = np.linspace(0, 1, nx)
                        X, Y = np.meshgrid(x_range, x_range)
                        val = eval(expr)
                        f_val = np.stack([val]*num_vars, axis=-1) # (nx, nx, C)
                        f_in = torch.FloatTensor(f_val).unsqueeze(0).to(device)
                        
                        if arch == "FNO":
                            u_pred = model(f_in).squeeze().cpu().numpy()
                        else:
                            coords = torch.stack(torch.meshgrid(torch.linspace(0,1,nx), torch.linspace(0,1,nx), indexing='ij'), -1).reshape(-1,2).unsqueeze(0).to(device)
                            u_pred = model(f_in.view(1,-1), coords).reshape(nx, nx, num_vars).cpu().numpy()
                        
                        st.subheader("预测结果")
                        if num_vars > 1:
                            cols = st.columns(num_vars)
                            for i in range(num_vars):
                                with cols[i]:
                                    st.markdown(f"**变量 {i+1}**")
                                    fig, ax = plt.subplots()
                                    im = ax.imshow(u_pred[..., i], extent=[0,1,0,1], origin='lower', cmap='plasma')
                                    fig.colorbar(im, ax=ax, label='Value')
                                    ax.set_xlabel('x'); ax.set_ylabel('y')
                                    st.pyplot(fig)
                        else:
                            c1, c2 = st.columns(2)
                            with c1:
                                st.subheader("输入源项 f(x, y)")
                                fig1, ax1 = plt.subplots()
                                im1 = ax1.imshow(f_val[..., 0], extent=[0,1,0,1], origin='lower', cmap='viridis')
                                fig1.colorbar(im1, ax=ax1, label='Value')
                                ax1.set_xlabel('x'); ax1.set_ylabel('y')
                                st.pyplot(fig1)
                            with c2:
                                st.subheader("预测响应解 u(x, y)")
                                fig2, ax2 = plt.subplots()
                                im2 = ax2.imshow(u_pred, extent=[0,1,0,1], origin='lower', cmap='plasma')
                                fig2.colorbar(im2, ax=ax2, label='Value')
                                ax2.set_xlabel('x'); ax2.set_ylabel('y')
                                st.pyplot(fig2)

            except Exception as e:
                st.error(f"解析错误或计算失败: {e}")
                if is_1d:
                    st.info("提示：一维模式请使用小写变量 x，例如 np.sin(x)。")
                else:
                    st.info("提示：二维模式请使用大写 X, Y。")
    else:
        st.warning("请先在‘模型训练’页完成模型训练。")
