"""
实时预测模块
使用训练好的 LSTM 模型预测 BTC 未来走势

v1.1 修复内容：
  - Bug1：统一训练/预测特征选择（由 config.FEATURE_SELECTION 控制）
  - Bug2：加载持久化的 scaler.pkl，替代局部 MinMax 归一化
  - Bug3：用 model_stats.json 中的 z-score 做概率归一化
"""

import torch
import math
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SEQUENCE_LENGTH, INPUT_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    PREDICT_THRESHOLD, MODEL_DIR, FEATURE_SELECTION
)
from models.lstm import create_model
from utils.indicators import calculate_indicators
from data.fetch_data import fetch_klines
from data.preprocess import prepare_features


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
            print(f"验证 MAE：{checkpoint.get('val_mae', 'N/A')}")
        else:
            print(f"警告：模型文件不存在 {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # ===== Bug2修复：加载持久化的 scaler =====
        scaler_path = Path(MODEL_DIR) / 'scaler.pkl'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print(f"✅ 已加载 scaler：{scaler_path}")
        else:
            self.scaler = None
            print(f"⚠️ 未找到 scaler.pkl，将使用局部 MinMax（不推荐，请重新训练）")
        
        # ===== Bug3修复：加载模型统计信息，用于 z-score 概率转换 =====
        stats_path = Path(MODEL_DIR) / 'model_stats.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                self.model_stats = json.load(f)
            print(f"✅ 已加载 model_stats：mean={self.model_stats['pred_mean']:.6f}, "
                  f"std={self.model_stats['pred_std']:.6f}")
        else:
            self.model_stats = None
            print(f"⚠️ 未找到 model_stats.json，将使用默认概率转换（请重新训练）")
        
        # 存储历史数据用于预测
        self.historical_data = None
    
    def load_historical_data(self, limit: int = 2000):
        """加载历史数据用于计算指标"""
        print("加载历史数据...")
        self.historical_data = fetch_klines(limit=limit)
        return self.historical_data
    
    def prepare_input(self, current_data: pd.DataFrame) -> torch.Tensor:
        """
        准备输入特征（v1.1：统一特征选择 + 使用持久化 scaler）
        
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
        
        # 取足够长的数据计算指标（避免头部 NaN）
        lookback = SEQUENCE_LENGTH + 100
        df = df.tail(lookback)
        
        if len(df) < SEQUENCE_LENGTH:
            raise ValueError(f"数据不足：需要{SEQUENCE_LENGTH}根 K 线，实际{len(df)}根")
        
        # ===== Bug1修复：使用 prepare_features 统一特征选择（与训练保持一致）=====
        features_scaled, _ = prepare_features(df, feature_selection=FEATURE_SELECTION)
        
        # 如果有持久化的 scaler，用它重新归一化（覆盖 prepare_features 内部的局部 scaler）
        if self.scaler is not None:
            # ===== Bug2修复：用持久化 scaler 做 transform =====
            raw_features, _ = prepare_features(df, feature_selection=FEATURE_SELECTION)
            # prepare_features 已经做了归一化，但 scaler 不同；
            # 这里重新取原始特征，用训练时的 scaler transform
            try:
                # 反归一化回原始值（用 prepare_features 内部的局部 scaler 逆变换）
                # 更简单的做法：直接重新计算技术指标，再用全局 scaler transform
                df_with_indicators = calculate_indicators(df)
                
                # 与训练时的特征列保持一致（由 FEATURE_SELECTION 决定）
                from data.preprocess import prepare_features as pf
                feature_cols_map = {
                    'core': [
                        'close', 'volume', 'rsi', 'macd', 'macd_diff',
                        'adx', 'bias_sma20', 'bias_sma50', 'bb_pct', 'bb_width',
                        'atr_pct', 'williams_r', 'roc_5', 'roc_10', 'volume_ratio',
                        'obv_change', 'candle_body', 'candle_range', 'momentum_accel', 'volatility'
                    ],
                    'momentum': [
                        'close', 'volume', 'rsi', 'macd', 'macd_diff', 'macd_signal',
                        'stoch_rsi', 'stoch_rsi_d', 'williams_r', 'roc_5', 'roc_10', 'roc_20',
                        'bias_sma20', 'bias_sma50', 'bias_ema12', 'bb_pct', 'volatility',
                        'volatility_ratio', 'volume_change', 'volume_ratio',
                        'candle_body', 'momentum_accel', 'ma_cross', 'adx', 'atr_pct'
                    ],
                }
                feature_cols = feature_cols_map.get(FEATURE_SELECTION, list(df_with_indicators.columns))
                available_cols = [c for c in feature_cols if c in df_with_indicators.columns]
                
                raw_values = df_with_indicators[available_cols].fillna(0).values
                
                # 检查 scaler 期望的特征数是否匹配
                if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ == len(available_cols):
                    features_values = self.scaler.transform(raw_values)
                else:
                    # 列数不匹配时，退回局部归一化
                    print(f"⚠️ scaler 特征数({getattr(self.scaler, 'n_features_in_', '?')}) "
                          f"与当前特征数({len(available_cols)})不匹配，使用局部归一化")
                    f_min = raw_values.min(axis=0)
                    f_max = raw_values.max(axis=0)
                    f_range = np.where(f_max - f_min < 1e-8, 1.0, f_max - f_min)
                    features_values = (raw_values - f_min) / f_range
                
                # 取最后 SEQUENCE_LENGTH 行
                features_seq = features_values[-SEQUENCE_LENGTH:]
                
            except Exception as e:
                print(f"⚠️ 使用持久化 scaler 失败（{e}），退回 prepare_features 归一化")
                features_seq = features_scaled.values[-SEQUENCE_LENGTH:]
        else:
            # 没有 scaler，直接用 prepare_features 的结果
            features_seq = features_scaled.values[-SEQUENCE_LENGTH:]
        
        input_tensor = torch.FloatTensor(features_seq).unsqueeze(0)  # [1, seq_len, features]
        return input_tensor.to(self.device)
    
    def _return_to_probability(self, predicted_return: float) -> float:
        """
        将模型输出的涨跌幅转换为上涨概率
        
        v1.1 Bug3修复：使用训练集统计的 z-score 做归一化，替代随意的 *10 系数
        """
        if self.model_stats is not None:
            # z-score 归一化后做 sigmoid
            mean = self.model_stats['pred_mean']
            std = self.model_stats['pred_std']
            z_score = (predicted_return - mean) / std
            probability = 1.0 / (1.0 + math.exp(-z_score))
        else:
            # 退回旧方式（仅无 model_stats 时兜底）
            probability = 1.0 / (1.0 + math.exp(-predicted_return * 10))
        return probability
    
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
        
        # 预测（回归模式：输出涨跌幅）
        with torch.no_grad():
            predicted_return = self.model(input_tensor).item()
        
        # ===== Bug3修复：使用 z-score 归一化概率 =====
        probability = self._return_to_probability(predicted_return)
        
        # 解析结果
        prediction = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'predicted_return': predicted_return,
            'probability': probability,
            'direction': '涨' if predicted_return > 0 else '跌',
            'confidence': '高' if abs(predicted_return) > 0.03 else '中' if abs(predicted_return) > 0.01 else '低',
            'current_price': latest_klines['close'].iloc[-1],
            'recommendation': self._get_recommendation_return(predicted_return)
        }
        
        return prediction
    
    def _get_recommendation_return(self, predicted_return: float) -> str:
        """根据预测涨跌幅给出操作建议"""
        if predicted_return > 0.03:
            return "强烈建议做多"
        elif predicted_return > 0.01:
            return "建议做多"
        elif predicted_return > 0.005:
            return "谨慎做多"
        elif predicted_return < -0.03:
            return "强烈建议做空/平仓"
        elif predicted_return < -0.01:
            return "建议做空/平仓"
        elif predicted_return < -0.005:
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
                predicted_return = self.model(input_tensor).item()
            
            probability = self._return_to_probability(predicted_return)
            
            predictions.append({
                'timestamp': df.index[i].isoformat(),
                'probability': probability,
                'predicted_return': predicted_return,
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
