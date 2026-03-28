"""
技术指标计算工具
使用 ta 库计算各种技术指标 + 量化特征
"""

import pandas as pd
import numpy as np
import ta


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有技术指标 + 量化特征
    
    Args:
        df: DataFrame，包含 open, high, low, close, volume 列
    
    Returns:
        DataFrame，添加技术指标列
    """
    df = df.copy()
    
    # ========== 趋势指标 ==========
    # RSI (14)
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    # MACD
    macd_obj = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
    df['macd'] = macd_obj.macd()
    df['macd_signal'] = macd_obj.macd_signal()
    df['macd_diff'] = macd_obj.macd_diff()
    
    # ADX
    adx_obj = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx_obj.adx()
    df['adx_pos'] = adx_obj.adx_pos()
    df['adx_neg'] = adx_obj.adx_neg()
    
    # 移动平均线
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
    
    # 均线乖离率
    df['bias_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['bias_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    df['bias_ema12'] = (df['close'] - df['ema_12']) / df['ema_12']
    
    # 均线交叉信号
    df['ma_cross'] = (df['ema_12'] > df['ema_26']).astype(int)
    
    # ========== 波动率指标 ==========
    # 布林带
    bb_obj = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb_obj.bollinger_hband()
    df['bb_mid'] = bb_obj.bollinger_mavg()
    df['bb_low'] = bb_obj.bollinger_lband()
    df['bb_pct'] = bb_obj.bollinger_pband()  # 布林带位置百分比
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']  # 带宽
    
    # ATR (平均真实波幅)
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['atr_pct'] = df['atr'] / df['close']  # ATR 百分比
    
    # 波动率
    df['volatility'] = df['close'].pct_change().rolling(window=14).std()
    df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=20).mean()  # 波动率比率
    
    # ========== 动量指标 ==========
    # KDJ / StochRSI
    try:
        kdj_obj = ta.momentum.StochRSIIndicator(df['close'], window=14)
        df['stoch_rsi'] = kdj_obj.stochrsi()
        df['stoch_rsi_d'] = kdj_obj.stochrsi_d()
    except:
        # 如果 ta 库版本不支持，用简单计算
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['stoch_rsi'] = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
        df['stoch_rsi_d'] = df['stoch_rsi'].rolling(3).mean()
    
    # Williams %R
    df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()
    
    # ROC (变化率)
    df['roc_5'] = ta.momentum.ROCIndicator(df['close'], window=5).roc()
    df['roc_10'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()
    df['roc_20'] = ta.momentum.ROCIndicator(df['close'], window=20).roc()
    
    # ========== 成交量指标 ==========
    # 成交量变化
    df['volume_change'] = df['volume'].pct_change()
    
    # OBV (能量潮)
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['obv_change'] = df['obv'].pct_change()
    
    # 成交量比率
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # 量价关系
    df['price_volume_corr'] = df['close'].rolling(window=20).corr(df['volume'])
    
    # ========== 价格形态特征 ==========
    # 涨跌幅
    df['price_change'] = df['close'].pct_change()
    df['price_change_5'] = df['close'].pct_change(periods=5)
    df['price_change_10'] = df['close'].pct_change(periods=10)
    
    # K 线形态
    df['candle_body'] = (df['close'] - df['open']) / df['open']  # 实体
    df['candle_upper'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']  # 上影线
    df['candle_lower'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']  # 下影线
    df['candle_range'] = (df['high'] - df['low']) / df['low']  # 振幅
    
    # 价格位置
    df['high_low_ratio'] = (df['close'] - df['low']) / (df['high'] - df['low'])  # 在当日区间的位置
    
    # 动量加速度
    df['momentum_accel'] = df['price_change'].diff()  # 涨跌幅变化
    
    # 连续涨跌
    df['up_streak'] = (df['close'] > df['close'].shift(1)).astype(int).groupby(
        (df['close'] > df['close'].shift(1)).astype(int).diff().abs().cumsum()
    ).cumsum()
    
    # ========== 填充 NaN 值 ==========
    df = df.bfill().fillna(0)
    
    return df


def get_feature_columns() -> list:
    """返回所有特征列名（42 个特征）"""
    return [
        # 基础价格
        'open', 'high', 'low', 'close', 'volume',
        
        # 趋势指标
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'adx', 'adx_pos', 'adx_neg',
        'sma_20', 'sma_50', 'ema_12', 'ema_26',
        'bias_sma20', 'bias_sma50', 'bias_ema12',
        'ma_cross',
        
        # 波动率指标
        'bb_pct', 'bb_width',
        'atr', 'atr_pct',
        'volatility', 'volatility_ratio',
        
        # 动量指标
        'stoch_rsi', 'stoch_rsi_d',
        'williams_r',
        'roc_5', 'roc_10', 'roc_20',
        
        # 成交量指标
        'volume_change', 'volume_ratio',
        'obv_change', 'price_volume_corr',
        
        # 价格形态
        'candle_body', 'candle_upper', 'candle_lower', 'candle_range',
        'high_low_ratio', 'momentum_accel',
        
        # 外部数据（如果有）
        'funding_rate', 'long_short_ratio'
    ]
