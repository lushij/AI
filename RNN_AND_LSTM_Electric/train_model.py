import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import wandb
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
# --- 1. 数据加载与预处理 ---
def load_data(file_path, window_size=96):
    data = pd.read_csv(file_path, header=None, skiprows=1)
    power_data = data[1].values.reshape(-1, 1)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(power_data)

    n = len(scaled_data)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    def create_sequences(dataset):
        X, Y = [], []
        for i in range(window_size, len(dataset)):
            X.append(dataset[i - window_size:i, :])
            Y.append(dataset[i])
        return np.array(X), np.array(Y)

    x_train, y_train = create_sequences(scaled_data[:train_end])
    x_val, y_val = create_sequences(scaled_data[train_end:val_end])
    x_test, y_test = create_sequences(scaled_data[val_end:])

    return x_train, y_train, x_val, y_val, x_test, y_test, scaler

# --- 2. MAPE 计算函数 ---
def calculate_mape(y_true, y_pred):
    # 避免除以 0
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# --- 3. 模型定义 ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 4. 评估逻辑 (集成 MAPE) ---
def evaluate(model, data_loader, criterion, device, scaler):
    model.eval()
    total_loss = 0
    all_preds = []
    all_reals = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            # 反标准化以计算真实的 MAPE
            preds_orig = scaler.inverse_transform(outputs.cpu().numpy())
            reals_orig = scaler.inverse_transform(batch_y.cpu().numpy())
            all_preds.append(preds_orig)
            all_reals.append(reals_orig)

    avg_loss = total_loss / len(data_loader)
    mape = calculate_mape(np.vstack(all_reals), np.vstack(all_preds))
    return avg_loss, mape

# --- 5. 训练主循环 ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="power-load-prediction",
        config={
            "lr": 0.001, "epochs": 100, "batch_size": 64,
            "hidden": 128, "layers": 2, "window": 96, "patience": 15
        }
    )
    c = wandb.config

    # 数据准备
    xt, yt, xv, yv, xtest, ytest, scaler = load_data('load.csv', c.window)
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(xt), torch.FloatTensor(yt)), batch_size=c.batch_size,
                              shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(xv), torch.FloatTensor(yv)), batch_size=c.batch_size)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(xtest), torch.FloatTensor(ytest)), batch_size=c.batch_size)

    model = LSTMModel(1, c.hidden, c.layers, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr)
    criterion = nn.MSELoss()

    best_mape = float('inf')
    early_stop_count = 0

    pbar = tqdm(range(c.epochs), desc="Training Progress")
    for epoch in pbar:
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        v_loss, v_mape = evaluate(model, val_loader, criterion, device, scaler)
        if v_mape < best_mape:
            best_mape = v_mape
            early_stop_count = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f" [★] Epoch {epoch + 1}: New Best MAPE {v_mape:.2f}%")
        else:
            early_stop_count += 1

        wandb.log({
            "epoch": epoch + 1,
            "val_mape": v_mape,
            "accuracy": 100 - v_mape,
            "train_loss": train_loss / len(train_loader)
        })

        pbar.set_postfix({"Best MAPE": f"{best_mape:.2f}%", "ES": f"{early_stop_count}/{c.patience}"})

        if early_stop_count >= c.patience:
            print("Early stopping triggered.")
            break

    # --- 4. 最终测试与可视化 ---
    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load('best_model.pth',weights_only=False))
    t_loss, t_mape = evaluate(model, test_loader, criterion, device, scaler)
    print(f"Final Test MAPE: {t_mape:.2f}%")

    # 抽取测试集前 200 个点画图对比
    model.eval()
    with torch.no_grad():
        sample_x = torch.FloatTensor(xtest[:200]).to(device)
        preds = scaler.inverse_transform(model(sample_x).cpu().numpy())
        reals = scaler.inverse_transform(ytest[:200])

        # 发送到 WandB 图表
        data = [[i, p[0], r[0]] for i, (p, r) in enumerate(zip(preds, reals))]
        table = wandb.Table(data=data, columns=["index", "Pred", "Real"])
        wandb.log({"Test_Prediction_Plot": wandb.plot.line_series(
            table=table, y_fields=["Pred", "Real"], x_name="index", title="Final Prediction vs Real")})

    wandb.finish()


def test_and_visualize(model_path, x_test, y_test, scaler, device):
    # 1. 结构初始化 (确保参数与训练时 config 一致)
    # 这里建议把参数写死或者传入，比如 hidden_size=128, num_layers=2
    model = LSTMModel(input_size=1, hidden_size=128, num_layers=2, out_size=1).to(device)

    # 2.
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 3. 【核心改进】使用 DataLoader 分批次预测，避免 OOM
    test_dataset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # 这里的 batch_size 可以设小点

    all_preds = []
    with torch.no_grad():
        for batch_x, _ in tqdm(test_loader, desc="Testing"):
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            all_preds.append(outputs.cpu().numpy())  # 跑完立刻传回 CPU

    # 拼接所有批次的结果
    predictions = np.vstack(all_preds)

    # 4. 反归一化
    real_predictions = scaler.inverse_transform(predictions)
    real_y_test = scaler.inverse_transform(y_test)

    # 5. 计算指标
    mae = np.mean(np.abs(real_predictions - real_y_test))
    mape = np.mean(np.abs((real_predictions - real_y_test) / (real_y_test + 1e-5))) * 100

    print(f"\n" + "=" * 30)
    print(f"测试集评估完成！")
    print(f"MAE:  {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"准确率: {100 - mape:.2f}%")
    print("=" * 30)

    # 6. 绘图 (这里可以直接用 matplotlib)
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 防止中文乱码
    plt.figure(figsize=(12, 5))
    plt.plot(real_y_test[:300], label='实际负荷', color='blue', linewidth=1.5)
    plt.plot(real_predictions[:300], label='预测负荷', color='red', linestyle='--', linewidth=1.5)
    plt.title('电力负荷预测 - 测试集对比 (前300点)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('prediction_results.svg', dpi=300)
    print("预测对比图已保存至项目根目录：prediction_results.svg")


def predict_future_24h(model_path, last_history_data, scaler, device, future_steps=96):
    """
    last_history_data: 训练集或测试集最后的 96 个原始数据点 (numpy array)
    """
    # 1. 加载模型
    model = LSTMModel(input_size=1, hidden_size=128, num_layers=2, out_size=1).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()

    # 2. 数据标准化
    # 注意：一定要用训练时的 scaler 进行变换
    current_sequence = scaler.transform(last_history_data.reshape(-1, 1))
    current_sequence = torch.FloatTensor(current_sequence).view(1, -1, 1).to(device)

    future_predictions = []

    # 3. 滚动预测
    with torch.no_grad():
        for _ in range(future_steps):
            # 预测下一时刻
            pred = model(current_sequence)
            future_predictions.append(pred.item())

            # 更新输入序列：移除第一个点，在末尾加入预测的点
            # pred 形状是 [1, 1], 需要增加维度对齐
            new_val = pred.view(1, 1, 1)
            current_sequence = torch.cat((current_sequence[:, 1:, :], new_val), dim=1)

    # 4. 反归一化
    final_preds = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # 5. 绘图
    save_path = 'future_24h_forecast.svg'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 释放内存

    print(f"\n" + "=" * 30)
    print(f"未来预测完成！")
    print(f"预测图片已保存至: {os.path.abspath(save_path)}")
    print(f"预测起始值: {final_preds[0][0]:.2f}")
    print(f"预测最大值: {np.max(final_preds):.2f}")
    print(f"=" * 30)

    return final_preds

    return final_preds





if __name__ == "__main__":
    # train()
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = load_data('load.csv', window_size=96)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # test_and_visualize('best_model.pth', x_test, y_test, scaler, device)
    data = pd.read_csv('load.csv', header=None, skiprows=1)
    power_data = data[1].values.reshape(-1, 1)
    last_96_points = power_data[-96:]
    predict_future_24h('best_model.pth', last_96_points, scaler, device)