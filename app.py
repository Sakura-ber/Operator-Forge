import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import io
import data_utils as du
from fno_model import FNO2d
from deeponet_model import DeepONet

# ====================== 设备辅助函数 ======================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

st.set_page_config(page_title="PDE AI Platform", layout="wide")

if 'model' not in st.session_state: st.session_state.model = None

st.sidebar.title("🛠️ 导航与配置")
page = st.sidebar.radio("选择功能", ["1. 数据生成", "2. 模型训练", "3. 预测推理"])

# ====================== 1. 数据生成 ======================
if page == "1. 数据生成":
    st.title("🧬 数据合成")
    # 算子参数 (Lu=f)
    with st.sidebar:
        st.header("算子参数")
        c_a = st.number_input("a (u_xx)", value=1.0)
        c_c = st.number_input("c (u_yy)", value=1.0)
        nx = st.number_input("NX/NY", value=50)
        num_samples = st.number_input("样本数", value=100)
        sob_max = st.number_input("Sobolev 范数上限", value=50.0)

    if st.button("🚀 生成数据"):
        x_pts = np.linspace(0, 1, nx)
        X, Y = np.meshgrid(x_pts, x_pts)
        dx = x_pts[1]-x_pts[0]
        u_list, f_list = [], []
        found, progress = 0, st.progress(0)
        while found < num_samples:
            u_raw = du.build_G(X, Y)
            u_bc = du.apply_robin_boundary(u_raw, X, Y, 1, 0, 'rect')
            if du.calculate_sobolev_norm(u_bc, dx, dx, 1, 2.0) <= sob_max:
                ux, uy, uxx, uyy, _ = du.compute_derivatives(u_bc, dx, dx)
                f = -(c_a*uxx + c_c*uyy)
                u_list.append(u_bc); f_list.append(f)
                found += 1
                progress.progress(found / num_samples)
        buf = io.BytesIO()
        np.savez_compressed(buf, u=np.array(u_list), f=np.array(f_list), nx=nx)
        st.success("生成完成！")
        st.download_button("📥 下载 .npz", buf.getvalue(), "data.npz")

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
            m1 = st.slider("Modes", 4, 32, 12); w = st.slider("Width", 16, 128, 32)
        else:
            hd = st.slider("Hidden Dim", 64, 512, 128); p_dim = st.slider("p Dim", 32, 256, 100)

    file = st.file_uploader("上传数据", type="npz")
    
    if file and st.button("开始训练"):
        # 数据加载
        data = np.load(file)
        f_t = torch.FloatTensor(data['f']).unsqueeze(-1)
        u_t = torch.FloatTensor(data['u']).unsqueeze(-1)
        nx = int(data['nx'])
        loader = DataLoader(TensorDataset(f_t, u_t), batch_size=int(bs), shuffle=True)
        
        # 设备设置
        device = get_device()
        st.info(f"当前计算设备: {device}")
        
        # 模型初始化
        model = FNO2d(m1, m1, w) if arch == "FNO" else DeepONet(nx*nx, hd, p_dim)
        model.to(device) # GPU 移动
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # 训练过程记录
        epoch_losses = []
        p_bar, status = st.progress(0), st.empty()
        
        for epoch in range(int(epochs)):
            loss_list = []
            for bf, bu in loader:
                # 数据移至GPU
                bf, bu = bf.to(device), bu.to(device)
                
                optimizer.zero_grad()
                if arch == "FNO": 
                    pred = model(bf)
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0,1,nx), torch.linspace(0,1,nx), indexing='ij'), -1).reshape(-1,2).unsqueeze(0).repeat(bf.size(0),1,1)
                    coords = coords.to(device) # 坐标移至GPU
                    pred = model(bf.view(bf.size(0),-1), coords).view(-1, nx, nx, 1)
                
                loss = criterion(pred, bu)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            
            avg_loss = np.mean(loss_list)
            epoch_losses.append(avg_loss)
            p_bar.progress((epoch+1)/int(epochs))
            status.text(f"Epoch {epoch+1}/{int(epochs)} | Loss: {avg_loss:.6f}")
        
        # 保存模型 (移回CPU以便存储)
        st.session_state.model = model.cpu()
        st.session_state.arch = arch
        st.session_state.nx = nx
        st.success("训练成功！")
 # ====================== 板块 2: 结果可视化 ======================
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
            st.info("右侧展示最后一个 Batch 的拟合效果对比")
            # 绘制最后一个 batch 的对比图
            with torch.no_grad():
                # 数据已在CPU
                true_u = bu[0].squeeze().numpy() if device.type == 'cpu' else bu[0].cpu().squeeze().numpy()
                pred_u = pred[0].squeeze().cpu().numpy() # pred在GPU上，需取回CPU
                
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
        expr = st.text_input("输入 f(X, Y) 解析式", "np.sin(np.pi*X)*np.sin(np.pi*Y)")
        nx = st.session_state.nx
        # 注意：这里需要确保 X, Y 对应 numpy 的 meshgrid
        x_range = np.linspace(0, 1, nx)
        X, Y = np.meshgrid(x_range, x_range)
        
        try:
            # 计算用户输入的函数值
            f_val = eval(expr)
            f_in = torch.FloatTensor(f_val).unsqueeze(0).unsqueeze(-1)
            
            with torch.no_grad():
                if st.session_state.arch == "FNO": 
                    u_pred = st.session_state.model(f_in).squeeze().numpy()
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0,1,nx), torch.linspace(0,1,nx), indexing='ij'), -1).reshape(-1,2).unsqueeze(0)
                    u_pred = st.session_state.model(f_in.view(1,-1), coords).reshape(nx, nx).numpy()
            
            # --- 绘图增强部分 ---
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("输入源项 f(x, y)")
                fig1, ax1 = plt.subplots()
                im1 = ax1.imshow(f_val, extent=[0,1,0,1], origin='lower', cmap='viridis')
                fig1.colorbar(im1, ax=ax1, label='Value') # 添加数值标尺
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                st.pyplot(fig1)
                
            with c2:
                st.subheader("预测响应解 u(x, y)")
                fig2, ax2 = plt.subplots()
                im2 = ax2.imshow(u_pred, extent=[0,1,0,1], origin='lower', cmap='plasma')
                fig2.colorbar(im2, ax=ax2, label='Value') # 添加数值标尺
                ax2.set_xlabel('x')
                ax2.set_ylabel('y')
                st.pyplot(fig2)
                
        except Exception as e: 
            st.error(f"解析错误或计算失败: {e}")
            st.info("提示：请确保表达式中使用的是大写 X, Y，并符合 numpy 语法。")
    else: 

        st.warning("请先切换到‘模型训练’标签页完成模型训练。")

