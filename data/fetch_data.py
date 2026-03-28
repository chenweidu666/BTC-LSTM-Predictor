"""
从 Gate.io 获取 BTC 历史 K 线数据
"""

import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GATE_API_BASE, CONTRACT, INTERVAL, DATA_DIR


def fetch_klines(limit: int = 1000, interval: str = None) -> pd.DataFrame:
    """
    从 Gate.io 获取 K 线数据
    
    Args:
        limit: 获取 K 线数量
        interval: K 线周期（默认使用 config 中的配置）
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    interval = interval or INTERVAL
    
    url = f"{GATE_API_BASE}/futures/usdt/candlesticks"
    params = {
        "contract": CONTRACT,
        "interval": interval,
        "limit": limit
    }
    
    print(f"正在从 Gate.io 获取 {limit} 根 {interval} K 线...")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            print("警告：未获取到数据")
            return pd.DataFrame()
        
        # Gate.io K 线格式：字典列表 [{'t': timestamp, 'o': open, 'h': high, 'l': low, 'c': close, 'v': volume}, ...]
        df = pd.DataFrame(data)
        df = df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
        
        # 转换数据类型
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # 设置时间索引
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        print(f"成功获取 {len(df)} 根 K 线")
        print(f"时间范围：{df.index[0]} 至 {df.index[-1]}")
        print(f"价格范围：${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"获取数据失败：{e}")
        return pd.DataFrame()


def fetch_funding_rates(limit: int = 1000) -> pd.DataFrame:
    """
    获取资金费率历史数据
    
    Args:
        limit: 获取数量
    
    Returns:
        DataFrame with columns: timestamp, funding_rate
    """
    url = f"{GATE_API_BASE}/futures/usdt/funding_rate"
    params = {
        "contract": CONTRACT,
        "limit": limit
    }
    
    print(f"正在获取资金费率历史...")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['t'], unit='s')
        df['funding_rate'] = df['r'].astype(float)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        print(f"成功获取 {len(df)} 条资金费率记录")
        return df[['funding_rate']]
        
    except Exception as e:
        print(f"获取资金费率失败：{e}")
        return pd.DataFrame()


def fetch_long_short_ratio(limit: int = 1000) -> pd.DataFrame:
    """
    获取多空持仓人数比
    
    Args:
        limit: 获取数量
    
    Returns:
        DataFrame with columns: timestamp, long_short_ratio
    """
    url = f"{GATE_API_BASE}/futures/usdt/candlesticks"
    params = {
        "contract": CONTRACT,
        "interval": "4h",
        "limit": limit
    }
    
    # 使用 MCP 工具获取多空比（如果有）或估算
    # 这里先用一个简单的代理指标：价格变化方向
    print(f"正在获取多空比数据...")
    
    try:
        # 从价格数据估算多空情绪
        klines = fetch_klines(limit=limit)
        if klines.empty:
            return pd.DataFrame()
        
        # 计算多空比代理指标
        df = klines.copy()
        df['long_short_ratio'] = 1.0 + (df['close'].pct_change() * 10)  # 简化代理
        df['long_short_ratio'] = df['long_short_ratio'].clip(0.5, 2.0)  # 限制范围
        
        print(f"计算多空比代理指标 {len(df)} 条")
        return df[['long_short_ratio']]
        
    except Exception as e:
        print(f"获取多空比失败：{e}")
        return pd.DataFrame()


def fetch_open_interest(limit: int = 100) -> pd.DataFrame:
    """
    获取持仓量数据（如果有 API 支持）
    
    注意：Gate.io 公开 API 可能不直接提供历史持仓量
    这里用成交量作为代理指标
    """
    print(f"正在获取持仓量数据（用成交量代理）...")
    
    try:
        klines = fetch_klines(limit=limit)
        if klines.empty:
            return pd.DataFrame()
        
        df = klines.copy()
        df['open_interest'] = df['volume'].rolling(window=20).sum()  # 20 周期滚动和作为代理
        
        print(f"计算持仓量代理指标 {len(df)} 条")
        return df[['open_interest']]
        
    except Exception as e:
        print(f"获取持仓量失败：{e}")
        return pd.DataFrame()


def save_data(df: pd.DataFrame, filename: str = None):
    """保存数据到 CSV"""
    if df.empty:
        print("数据为空，不保存")
        return
    
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"klines_{timestamp}.csv"
    
    filepath = data_dir / filename
    df.to_csv(filepath)
    print(f"数据已保存到：{filepath}")
    return filepath


def load_data(filename: str) -> pd.DataFrame:
    """从 CSV 加载数据"""
    filepath = Path(DATA_DIR) / filename
    if not filepath.exists():
        print(f"文件不存在：{filepath}")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
    print(f"已加载 {len(df)} 根 K 线 from {filepath}")
    return df


if __name__ == "__main__":
    # 测试：获取并保存数据
    df = fetch_klines(limit=1000)
    if not df.empty:
        save_data(df)
        
        # 获取资金费率
        fr_df = fetch_funding_rates(limit=500)
        if not fr_df.empty:
            save_data(fr_df, "funding_rates.csv")
