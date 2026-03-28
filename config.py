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
SEQUENCE_LENGTH = 60  # 输入序列长度（60 根 K 线 = 10 天）
INPUT_FEATURES = 23   # 输入特征数量（实际特征数）
HIDDEN_SIZE = 64      # LSTM 隐藏层大小
NUM_LAYERS = 2        # LSTM 层数
DROPOUT = 0.2         # Dropout 比例

# 训练配置
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8     # 训练集比例

# 数据配置
DATA_DIR = "data"
MODEL_DIR = "models"
HISTORY_DAYS = 365    # 获取历史数据天数

# 预测配置
PREDICT_THRESHOLD = 0.6  # 涨跌概率阈值（>0.6 看涨，<0.4 看跌）
