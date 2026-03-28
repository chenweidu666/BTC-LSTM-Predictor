"""
LSTM 模型定义
"""

import torch
import torch.nn as nn


class LSTMPricePredictor(nn.Module):
    """
    LSTM 价格预测模型
    
    输入：[batch_size, sequence_length, input_features]
    输出：[batch_size, 1] 涨跌概率（0-1）
    """
    
    def __init__(self, input_features: int = 15, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMPricePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力机制（可选，提升性能）
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出涨跌概率（0-1）
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入 [batch_size, sequence_length, input_features]
        
        Returns:
            输出 [batch_size, 1] 涨跌概率
        """
        # LSTM 输出
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 注意力加权
        attention_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        context = torch.sum(attention_weights * lstm_out, dim=1)  # [batch, hidden_size]
        
        # 全连接层
        output = self.fc(context)
        
        return output
    
    def predict_direction(self, x, threshold=0.5):
        """
        预测涨跌方向
        
        Args:
            x: 输入特征
            threshold: 概率阈值
        
        Returns:
            1=涨，0=跌
        """
        with torch.no_grad():
            prob = self.forward(x)
            return (prob > threshold).int()


def create_model(input_features: int = 15, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2) -> LSTMPricePredictor:
    """创建模型实例"""
    model = LSTMPricePredictor(
        input_features=input_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    model = create_model()
    print(f"模型参数量：{count_parameters(model):,}")
    
    # 测试前向传播
    batch_size = 32
    seq_len = 60
    features = 15
    x = torch.randn(batch_size, seq_len, features)
    output = model(x)
    print(f"输入形状：{x.shape}")
    print(f"输出形状：{output.shape}")
    print(f"输出范围：[{output.min():.4f}, {output.max():.4f}]")
