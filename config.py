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
INPUT_FEATURES = 42   # 输入特征数量（42 个量化特征）
HIDDEN_SIZE = 128     # LSTM 隐藏层大小（增加模型容量）
NUM_LAYERS = 3        # LSTM 层数（增加深度）
DROPOUT = 0.3         # Dropout 比例（防止过拟合）

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
