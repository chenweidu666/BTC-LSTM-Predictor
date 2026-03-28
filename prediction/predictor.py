"""
实时预测模块
使用训练好的 LSTM 模型预测 BTC 未来走势
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SEQUENCE_LENGTH, INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    PREDICT_THRESHOLD, MODEL_DIR
)
from models.lstm import create_model
from utils.indicators import calculate_indicators
from data.fetch_data import fetch_klines


class BTCPredictor:
    """BTC 价格预测器"""
    
    def __init__(self, model_path: str = None):
        """
        初始化预测器
        
        Args:
            model_path: 模型权重文件路径（默认使用最新的 best_model.pth）
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备：{self.device}")
        
        # 创建模型
        self.model = create_model(
            input_features=INPUT_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        )
        
        # 加载权重
        if model_path is None:
            model_path = Path(MODEL_DIR) / 'best_model.pth'
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"已加载模型：{model_path}")
            print(f"验证准确率：{checkpoint.get('val_acc', 'N/A')}")
        else:
            print(f"警告：模型文件不存在 {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # 存储历史数据用于预测
        self.historical_data = None
        self.scaler = None
    
    def load_historical_data(self, limit: int = 2000):
        """加载历史数据用于计算指标"""
        print("加载历史数据...")
        self.historical_data = fetch_klines(limit=limit)
        return self.historical_data
    
    def prepare_input(self, current_data: pd.DataFrame) -> torch.Tensor:
        """
        准备输入特征
        
        Args:
            current_data: 当前 K 线数据（包含最新价格）
        
        Returns:
            模型输入张量 [1, sequence_length, features]
        """
        if self.historical_data is None:
            self.load_historical_data()
        
        # 合并历史数据和当前数据
        df = pd.concat([self.historical_data, current_data])
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        
        # 取最近 SEQUENCE_LENGTH 根 K 线
        if len(df) < SEQUENCE_LENGTH:
            raise ValueError(f"数据不足：需要{SEQUENCE_LENGTH}根 K 线，实际{len(df)}根")
        
        recent_df = df.tail(SEQUENCE_LENGTH)
        
        # 计算技术指标
        recent_df = calculate_indicators(recent_df)
        
        # 选择特征列（与 preprocess.py 一致）
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'adx', 'adx_pos', 'adx_neg',
            'bb_pct', 'bb_high', 'bb_low',
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'price_change', 'price_change_5',
            'volatility', 'volume_change'
        ]
        available_cols = [col for col in feature_cols if col in recent_df.columns]
        
        features = recent_df[available_cols].values
        
        # 标准化（简单 MinMax）
        features_min = features.min(axis=0)
        features_max = features.max(axis=0)
        features_range = features_max - features_min
        features_range[features_range == 0] = 1  # 避免除零
        features_normalized = (features - features_min) / features_range
        
        # 转换为 Tensor
        input_tensor = torch.FloatTensor(features_normalized).unsqueeze(0)  # [1, seq_len, features]
        
        return input_tensor.to(self.device)
    
    def predict(self, current_price: float = None) -> dict:
        """
        预测未来走势
        
        Args:
            current_price: 当前价格（可选，用于获取最新数据）
        
        Returns:
            预测结果字典
        """
        # 获取最新 K 线数据
        print("获取最新市场数据...")
        latest_klines = fetch_klines(limit=SEQUENCE_LENGTH + 10)
        
        if latest_klines.empty:
            return {"error": "无法获取市场数据"}
        
        # 准备输入
        input_tensor = self.prepare_input(latest_klines)
        
        # 预测
        with torch.no_grad():
            probability = self.model(input_tensor).item()
        
        # 解析结果
        prediction = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'probability': probability,
            'direction': '涨' if probability > PREDICT_THRESHOLD else '跌' if probability < (1 - PREDICT_THRESHOLD) else '震荡',
            'confidence': '高' if probability > 0.7 or probability < 0.3 else '中' if probability > 0.55 or probability < 0.45 else '低',
            'current_price': latest_klines['close'].iloc[-1],
            'recommendation': self._get_recommendation(probability)
        }
        
        return prediction
    
    def _get_recommendation(self, probability: float) -> str:
        """根据概率给出操作建议"""
        if probability > 0.7:
            return "强烈建议做多"
        elif probability > 0.6:
            return "建议做多"
        elif probability > 0.55:
            return "谨慎做多"
        elif probability < 0.3:
            return "强烈建议做空/平仓"
        elif probability < 0.4:
            return "建议做空/平仓"
        elif probability < 0.45:
            return "谨慎做空"
        else:
            return "持仓观望"
    
    def predict_batch(self, data_path: str = None) -> list:
        """
        批量预测（用于回测）
        
        Args:
            data_path: 数据文件路径
        
        Returns:
            预测结果列表
        """
        from data.fetch_data import load_data
        
        if data_path:
            df = load_data(data_path)
        else:
            df = self.historical_data
        
        predictions = []
        
        for i in range(SEQUENCE_LENGTH, len(df)):
            window = df.iloc[i-SEQUENCE_LENGTH:i]
            input_tensor = self.prepare_input(window)
            
            with torch.no_grad():
                prob = self.model(input_tensor).item()
            
            predictions.append({
                'timestamp': df.index[i].isoformat(),
                'probability': prob,
                'actual_close': df['close'].iloc[i],
                'actual_direction': 1 if df['close'].iloc[i] > df['close'].iloc[i-1] else 0
            })
        
        return predictions


def main():
    """测试预测"""
    print("=" * 60)
    print("BTC LSTM Predictor - 实时预测")
    print("=" * 60)
    
    predictor = BTCPredictor()
    
    print("\n进行预测...")
    result = predictor.predict()
    
    if 'error' in result:
        print(f"错误：{result['error']}")
        return
    
    print("\n" + "=" * 60)
    print("预测结果")
    print("=" * 60)
    print(f"时间：{result['timestamp']}")
    print(f"当前价格：${result['current_price']:,.2f}")
    print(f"上涨概率：{result['probability']*100:.1f}%")
    print(f"预测方向：{result['direction']}")
    print(f"置信度：{result['confidence']}")
    print(f"操作建议：{result['recommendation']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
