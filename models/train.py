"""
LSTM 模型训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, TRAIN_SPLIT,
    MODEL_DIR, SEQUENCE_LENGTH, INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT
)
from models.lstm import create_model, count_parameters
from data.preprocess import load_and_prepare, train_test_split

# 回归模式：预测涨跌幅（而非方向）
REGRESSION_MODE = True


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, 
                batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """
    训练模型
    
    Args:
        model: LSTM 模型
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
    
    Returns:
        训练历史记录
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    print(f"训练模式：{'回归（预测涨跌幅）' if REGRESSION_MODE else '分类（预测涨跌方向）'}")
    
    model = model.to(device)
    
    # 转换为 Tensor
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # 创建 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 损失函数和优化器（根据模式选择）
    if REGRESSION_MODE:
        criterion = nn.MSELoss()  # 回归用 MSE
    else:
        criterion = nn.BCELoss()  # 分类用 BCE
    
    # AdamW + 权重衰减（减少过拟合）
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)
    
    # 早停
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    
    # 训练循环
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    best_val_loss = float('inf')
    
    print(f"\n开始训练（{epochs} 轮）...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += torch.abs(outputs - batch_y).mean().item()
        
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_mae = torch.abs(val_outputs - y_val_tensor).mean().item()
            
            # 如果是回归模式，计算方向准确率
            if REGRESSION_MODE:
                val_direction_acc = ((val_outputs > 0) == (y_val_tensor > 0)).float().mean().item()
            else:
                val_direction_acc = ((val_outputs > 0.5) == (y_val_tensor > 0.5)).float().mean().item()
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        # 打印进度
        if REGRESSION_MODE:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} MAE: {train_mae:.6f} | "
                  f"Val Loss: {val_loss:.6f} MAE: {val_mae:.6f} Dir_Acc: {val_direction_acc:.4f}")
        else:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_mae:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_mae:.4f}")
        
        # 保存最佳模型（回归用 MAE，分类用 loss）
        metric = val_mae if REGRESSION_MODE else val_loss
        if metric < best_val_loss:
            best_val_loss = metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_direction_acc': val_direction_acc,
                'regression_mode': REGRESSION_MODE,
            }, Path(MODEL_DIR) / 'best_model.pth')
            if REGRESSION_MODE:
                print(f"  → 保存最佳模型 (val_mae={val_mae:.6f}, dir_acc={val_direction_acc:.4f})")
            else:
                print(f"  → 保存最佳模型 (val_acc={val_direction_acc:.4f})")
        
        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n早停触发（{early_stopping.patience} 轮未改善）")
            break
    
    print("=" * 60)
    if REGRESSION_MODE:
        best_idx = history['val_mae'].index(min(history['val_mae']))
        print(f"训练完成！最佳验证 MAE: {min(history['val_mae']):.6f}")
        print(f"最佳方向准确率：{history.get('val_direction_acc', [0])[best_idx] if 'val_direction_acc' in history else 'N/A'}")
    else:
        print(f"训练完成！最佳验证准确率：{max(history.get('val_acc', [0])):.4f}")
    
    return history


def plot_history(history, save_path=None):
    """绘制训练历史"""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss 曲线
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy 曲线
        ax2.plot(history['train_acc'], label='Train Acc')
        ax2.plot(history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training & Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存：{save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("matplotlib 未安装，跳过图表绘制")


def main():
    """主训练流程"""
    print("=" * 60)
    print("BTC LSTM Predictor - 模型训练")
    print("=" * 60)
    
    # 加载和准备数据
    print("\n[1/4] 加载数据...")
    X, y, scaler = load_and_prepare()
    
    # 分割训练/测试集
    print("\n[2/4] 分割数据集...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, TRAIN_SPLIT)
    
    # 创建模型
    print("\n[3/4] 创建模型...")
    model = create_model(
        input_features=INPUT_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    print(f"模型参数量：{count_parameters(model):,}")
    
    # 训练模型
    print("\n[4/4] 训练模型...")
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # 保存训练历史
    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(model_dir / f'training_history_{timestamp}.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # ===== Bug2修复：持久化 scaler =====
    print("\n保存 scaler 到 models/scaler.pkl ...")
    joblib.dump(scaler, model_dir / 'scaler.pkl')
    print("✅ scaler 已保存")
    
    # ===== Bug3修复：保存预测值统计信息，用于概率归一化 =====
    print("计算训练集预测值统计信息...")
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        X_all_tensor = torch.FloatTensor(X).to(device)
        all_preds = model(X_all_tensor).squeeze().cpu().numpy()
    pred_mean = float(np.mean(all_preds))
    pred_std = float(np.std(all_preds))
    if pred_std < 1e-8:
        pred_std = 1.0  # 避免除零
    model_stats = {
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'regression_mode': REGRESSION_MODE,
        'trained_at': timestamp,
    }
    with open(model_dir / 'model_stats.json', 'w') as f:
        json.dump(model_stats, f, indent=2)
    print(f"✅ model_stats.json 已保存 (mean={pred_mean:.6f}, std={pred_std:.6f})")
    
    # 绘制训练曲线
    plot_history(history, save_path=model_dir / f'training_curve_{timestamp}.png')
    
    print(f"\n✅ 训练完成！模型已保存到 {model_dir}")
    
    return model, history


if __name__ == "__main__":
    main()
