"""
数据预处理和特征工程
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SEQUENCE_LENGTH, DATA_DIR
from utils.indicators import calculate_indicators, get_feature_columns


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    准备特征数据
    
    Args:
        df: 原始 K 线数据
    
    Returns:
        (features_df, scaler) 特征 DataFrame 和标准化器
    """
    print("计算技术指标...")
    
    # 计算技术指标
    df = calculate_indicators(df)
    
    # 选择特征列
    feature_cols = get_feature_columns()
    available_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"使用 {len(available_cols)} 个特征：{available_cols}")
    
    features = df[available_cols].copy()
    
    # 标准化特征
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index
    )
    
    print("特征标准化完成")
    return features_scaled, scaler


def create_sequences(features: pd.DataFrame, target_col: str = 'close', 
                     sequence_length: int = None) -> tuple:
    """
    创建时间序列样本
    
    Args:
        features: 特征 DataFrame
        target_col: 目标列名
        sequence_length: 序列长度
    
    Returns:
        (X, y) X: [samples, sequence_length, features], y: [samples]
    """
    sequence_length = sequence_length or SEQUENCE_LENGTH
    
    print(f"创建时间序列样本（序列长度={sequence_length}）...")
    
    feature_values = features.values
    target_values = features[target_col].values if target_col in features.columns else None
    
    X, y = [], []
    
    for i in range(sequence_length, len(feature_values)):
        # 输入：过去 sequence_length 个时间步的特征
        X.append(feature_values[i-sequence_length:i])
        
        # 输出：下一个时间步的涨跌（1=涨，0=跌）
        if target_values is not None:
            # 预测下一时刻是否上涨
            y.append(1 if target_values[i] > target_values[i-1] else 0)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"创建完成：X 形状={X.shape}, y 形状={y.shape}")
    print(f"样本分布：上涨={sum(y)} ({sum(y)/len(y)*100:.1f}%), 下跌={len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    
    return X, y


def train_test_split(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> tuple:
    """
    训练集/测试集分割（时间序列分割，不打乱）
    
    Args:
        X: 特征
        y: 标签
        train_ratio: 训练集比例
    
    Returns:
        (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(X) * train_ratio)
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"训练集：{len(X_train)} 样本")
    print(f"测试集：{len(X_test)} 样本")
    
    return X_train, X_test, y_train, y_test


def load_and_prepare(filename: str = None) -> tuple:
    """
    加载数据并准备特征
    
    Returns:
        (X, y, scaler) 准备好的数据和标准化器
    """
    from data.fetch_data import load_data, fetch_klines
    
    # 尝试加载本地数据，如果没有则获取新数据
    if filename:
        df = load_data(filename)
    else:
        df = fetch_klines(limit=2000)
    
    if df.empty:
        raise ValueError("无法获取数据")
    
    # 准备特征
    features, scaler = prepare_features(df)
    
    # 创建序列
    X, y = create_sequences(features)
    
    return X, y, scaler


if __name__ == "__main__":
    # 测试
    X, y, scaler = load_and_prepare()
    print(f"\n最终数据：X={X.shape}, y={y.shape}")
