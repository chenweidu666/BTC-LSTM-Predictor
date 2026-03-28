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


def prepare_features(df: pd.DataFrame, funding_df: pd.DataFrame = None, 
                     feature_selection: str = 'all') -> tuple:
    """
    准备特征数据
    
    Args:
        df: 原始 K 线数据
        funding_df: 资金费率数据（可选）
        feature_selection: 'all'=全部 42 个，'core'=核心 20 个，'momentum'=动量为主
    
    Returns:
        (features_df, scaler) 特征 DataFrame 和标准化器
    """
    print("计算技术指标 + 量化特征...")
    
    # 计算技术指标
    df = calculate_indicators(df)
    
    # 合并资金费率数据（如果有）
    if funding_df is not None and not funding_df.empty:
        print("合并资金费率数据...")
        df = df.join(funding_df, how='left')
        df['funding_rate'] = df['funding_rate'].fillna(0)
    
    # 选择特征列
    if feature_selection == 'core':
        # 核心特征（20 个，减少冗余）
        feature_cols = [
            'close', 'volume',
            'rsi', 'macd', 'macd_diff',
            'adx', 'bias_sma20', 'bias_sma50',
            'bb_pct', 'bb_width', 'atr_pct',
            'williams_r', 'roc_5', 'roc_10',
            'volume_ratio', 'obv_change',
            'candle_body', 'candle_range',
            'momentum_accel', 'volatility'
        ]
    elif feature_selection == 'momentum':
        # 动量特征为主（25 个）
        feature_cols = [
            'close', 'volume',
            'rsi', 'macd', 'macd_diff', 'macd_signal',
            'stoch_rsi', 'stoch_rsi_d', 'williams_r',
            'roc_5', 'roc_10', 'roc_20',
            'bias_sma20', 'bias_sma50', 'bias_ema12',
            'bb_pct', 'volatility', 'volatility_ratio',
            'volume_change', 'volume_ratio',
            'candle_body', 'momentum_accel',
            'ma_cross', 'adx', 'atr_pct'
        ]
    elif feature_selection == 'minimal':
        # 精简特征（15 个，只保留相对有效的）
        feature_cols = [
            'close',           # 需要用于计算 target
            'momentum_accel',  # 最相关 +0.073
            'candle_body',     # K 线实体 +0.057
            'candle_upper',    # 上影线 +0.046
            'volatility_ratio', # +0.035
            'volume_change',   # -0.024
            'stoch_rsi',       # +0.024
            'roc_20',          # -0.023
            'atr',             # -0.023
            'bb_width',        # +0.023
            'obv_change',      # +0.022
            'williams_r',      # 动量
            'roc_5',           # 短期动量
            'ma_cross',        # 均线交叉
            'bias_sma20',      # 乖离率
        ]
    else:
        # 全部 42 个特征
        feature_cols = get_feature_columns()
    
    available_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"特征模式：{feature_selection}")
    print(f"使用 {len(available_cols)} 个特征")
    
    # 特征重要性分析
    if feature_selection == 'all':
        print("\n特征相关性分析（与目标变量）...")
        target = df['close'].pct_change().shift(-1)  # 下一期收益率
        correlations = []
        for col in available_cols[:10]:  # 只分析前 10 个
            corr = df[col].corr(target)
            correlations.append((col, corr))
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        for col, corr in correlations[:5]:
            print(f"  {col}: {corr:+.4f}")
    
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
                     sequence_length: int = None, predict_return: bool = True) -> tuple:
    """
    创建时间序列样本
    
    Args:
        features: 特征 DataFrame
        target_col: 目标列名
        sequence_length: 序列长度
        predict_return: True=预测涨跌幅（回归）, False=预测涨跌方向（分类）
    
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
        
        # 输出：根据模式选择
        if target_values is not None:
            if predict_return:
                # 回归：预测未来涨跌幅度（连续值）
                if target_values[i-1] != 0:
                    ret = (target_values[i] - target_values[i-1]) / target_values[i-1]
                    # 限制异常值
                    ret = np.clip(ret, -0.5, 0.5)
                    y.append(ret)
                else:
                    y.append(0.0)
            else:
                # 分类：预测涨跌方向（0/1）
                y.append(1 if target_values[i] > target_values[i-1] else 0)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"创建完成：X 形状={X.shape}, y 形状={y.shape}")
    
    if predict_return:
        print(f"涨跌幅分布：min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}, std={y.std():.4f}")
    else:
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
    from config import FEATURE_SELECTION
    
    # 尝试加载本地数据，如果没有则获取新数据
    if filename:
        df = load_data(filename)
    else:
        df = fetch_klines(limit=2000)
    
    if df.empty:
        raise ValueError("无法获取数据")
    
    # 准备特征（使用特征选择）
    features, scaler = prepare_features(df, feature_selection=FEATURE_SELECTION)
    
    # 创建序列（使用回归模式）
    from models.train import REGRESSION_MODE
    X, y = create_sequences(features, predict_return=REGRESSION_MODE)
    
    return X, y, scaler


if __name__ == "__main__":
    # 测试
    X, y, scaler = load_and_prepare()
    print(f"\n最终数据：X={X.shape}, y={y.shape}")
