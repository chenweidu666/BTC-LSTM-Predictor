"""
BTC LSTM Predictor - Configuration
"""

# Gate.io API 配置
GATE_API_BASE = "https://api.gateio.ws/api/v4"

# 交易对配置
CONTRACT = "BTC_USDT"
SETTLE = "usdt"
INTERVAL = "4h"  # K 线周期

# 模型配置
SEQUENCE_LENGTH = 60   # 输入序列长度（60 根 K 线 = 10 天）
INPUT_FEATURES = 20    # 输入特征数量（核心 20 个特征）
HIDDEN_SIZE = 64       # LSTM 隐藏层大小
NUM_LAYERS = 2         # LSTM 层数
DROPOUT = 0.4          # Dropout 比例

# 特征选择模式：'all'=42 个，'core'=20 个核心，'momentum'=25 个动量
FEATURE_SELECTION = 'core'

# 训练配置
BATCH_SIZE = 64       # 增加 batch size
EPOCHS = 100          # 增加训练轮数
LEARNING_RATE = 0.0005  # 降低学习率
TRAIN_SPLIT = 0.8     # 训练集比例

# 数据配置
DATA_DIR = "data"
MODEL_DIR = "models"
HISTORY_DAYS = 730    # 获取 2 年历史数据

# 预测配置
PREDICT_THRESHOLD = 0.6  # 涨跌概率阈值（>0.6 看涨，<0.4 看跌）
