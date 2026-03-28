"""
技术指标计算工具
使用 ta 库计算各种技术指标
"""

import pandas as pd
import numpy as np
import ta


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有技术指标
    
    Args:
        df: DataFrame，包含 open, high, low, close, volume 列
    
    Returns:
        DataFrame，添加技术指标列
    """
    df = df.copy()
    
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
    
    # 布林带
    bb_obj = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb_obj.bollinger_hband()
    df['bb_mid'] = bb_obj.bollinger_mavg()
    df['bb_low'] = bb_obj.bollinger_lband()
    df['bb_pct'] = bb_obj.bollinger_pband()  # 布林带位置百分比
    
    # 移动平均线
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
    df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
    
    # 涨跌幅
    df['price_change'] = df['close'].pct_change()
    df['price_change_5'] = df['close'].pct_change(periods=5)
    
    # 波动率
    df['volatility'] = df['price_change'].rolling(window=14).std()
    
    # 成交量变化
    df['volume_change'] = df['volume'].pct_change()
    
    # 填充 NaN 值
    df = df.bfill().fillna(0)
    
    return df


def get_feature_columns() -> list:
    """返回所有特征列名"""
    return [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'adx', 'adx_pos', 'adx_neg',
        'bb_pct', 'bb_high', 'bb_low',
        'sma_20', 'sma_50', 'ema_12', 'ema_26',
        'price_change', 'price_change_5',
        'volatility', 'volume_change'
    ]
