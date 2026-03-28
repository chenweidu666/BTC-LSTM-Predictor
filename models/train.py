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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, TRAIN_SPLIT,
    MODEL_DIR, SEQUENCE_LENGTH, INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT
)
from models.lstm import create_model, count_parameters
from data.preprocess import load_and_prepare, train_test_split


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
    
    model = model.to(device)
    
    # 转换为 Tensor
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # 创建 DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=5)
    
    # 早停
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    
    # 训练循环
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    
    print(f"\n开始训练（{epochs} 轮）...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / len(y_train)
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_predictions = (val_outputs > 0.5).float()
            val_correct = (val_predictions == y_val_tensor).sum().item()
            val_acc = val_correct / len(y_val)
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 打印进度
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, Path(MODEL_DIR) / 'best_model.pth')
            print(f"  → 保存最佳模型 (val_acc={val_acc:.4f})")
        
        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\n早停触发（{early_stopping.patience} 轮未改善）")
            break
    
    print("=" * 60)
    print(f"训练完成！最佳验证准确率：{max(history['val_acc']):.4f}")
    
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
    
    # 绘制训练曲线
    plot_history(history, save_path=model_dir / f'training_curve_{timestamp}.png')
    
    print(f"\n✅ 训练完成！模型已保存到 {model_dir}")
    
    return model, history


if __name__ == "__main__":
    main()
