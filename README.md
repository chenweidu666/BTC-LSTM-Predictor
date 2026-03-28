# BTC LSTM Predictor 🦞

BTC 价格趋势预测 - 基于 LSTM 深度学习模型

## 📖 项目简介

使用 LSTM（长短期记忆）神经网络预测 BTC 未来 4 小时的价格走势，为量化交易提供决策支持。

**核心特点：**
- 🧠 双层决策：技术指标 + 深度学习
- 📊 15+ 特征输入：OHLCV + RSI/MACD/ADX/布林带等
- ⚡ 实时预测：支持实时市场数据预测
- 📈 完整训练流程：数据获取 → 特征工程 → 模型训练 → 实时预测

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd BTC-LSTM-Predictor
pip install -r requirements.txt
```

### 2. 获取数据

```bash
python data/fetch_data.py
```

从 Gate.io 获取 BTC 永续合约 4h K 线数据。

### 3. 训练模型

```bash
python models/train.py
```

训练流程：
- 自动获取历史数据（默认 2000 根 K 线）
- 计算技术指标
- 创建时间序列样本
- 训练 LSTM 模型
- 保存最佳模型权重

### 4. 实时预测

```bash
python prediction/predictor.py
```

输出预测结果：
```
============================================================
预测结果
============================================================
时间：2026-03-29T12:00:00+00:00
当前价格：$69,420.00
上涨概率：68.5%
预测方向：涨
置信度：中
操作建议：建议做多
============================================================
```

---

## 📁 项目结构

```
BTC-LSTM-Predictor/
├── data/
│   ├── fetch_data.py    # 从 Gate.io 获取 K 线数据
│   └── preprocess.py    # 特征工程和预处理
├── models/
│   ├── lstm.py          # LSTM 模型定义
│   ├── train.py         # 训练脚本
│   └── best_model.pth   # 训练好的模型权重
├── prediction/
│   └── predictor.py     # 实时预测模块
├── utils/
│   └── indicators.py    # 技术指标计算
├── config.py            # 配置参数
├── requirements.txt     # Python 依赖
└── README.md            # 项目说明
```

---

## 🧠 模型架构

### 输入特征（15 个）

| 类别 | 特征 |
|------|------|
| 价格 | open, high, low, close |
| 成交量 | volume |
| 动量 | RSI(14), MACD, MACD Signal, MACD Diff |
| 趋势 | ADX, ADX+, ADX- |
| 波动 | 布林带位置 %, 波动率 |
| 均线 | SMA20, SMA50, EMA12, EMA26 |
| 变化 | 价格变化率，成交量变化率 |

### 模型参数

| 参数 | 值 |
|------|-----|
| 序列长度 | 60（10 天 4h K 线）|
| 隐藏层大小 | 64 |
| LSTM 层数 | 2 |
| Dropout | 0.2 |
| 学习率 | 0.001 |
| Batch Size | 32 |
| 最大 Epochs | 50 |

### 网络结构

```
输入 [batch, 60, 15]
    ↓
LSTM (hidden=64, layers=2)
    ↓
注意力机制
    ↓
全连接层 (64 → 32 → 1)
    ↓
Sigmoid → 上涨概率 [0-1]
```

---

## 📊 配置参数

编辑 `config.py` 自定义参数：

```python
# 模型配置
SEQUENCE_LENGTH = 60      # 输入序列长度
INPUT_FEATURES = 15       # 特征数量
HIDDEN_SIZE = 64          # LSTM 隐藏层大小
NUM_LAYERS = 2            # LSTM 层数

# 训练配置
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8         # 训练集比例

# 预测配置
PREDICT_THRESHOLD = 0.6   # 涨跌概率阈值
```

---

## 🎯 预测结果解读

| 上涨概率 | 预测方向 | 操作建议 |
|---------|---------|---------|
| > 0.7 | 强烈看涨 | 强烈建议做多 |
| 0.6-0.7 | 看涨 | 建议做多 |
| 0.55-0.6 | 谨慎看涨 | 谨慎做多 |
| 0.45-0.55 | 震荡 | 持仓观望 |
| 0.4-0.45 | 谨慎看跌 | 谨慎做空 |
| 0.3-0.4 | 看跌 | 建议做空/平仓 |
| < 0.3 | 强烈看跌 | 强烈建议做空/平仓 |

---

## ⚠️ 风险提示

1. **预测 ≠ 稳赚** - 模型基于历史数据，无法预测黑天鹅事件
2. **过拟合风险** - 训练集表现好不代表实盘能赚钱
3. **市场风险** - 加密货币波动剧烈，请做好风险管理
4. **仅供参考** - 本项目仅供学习研究，不构成投资建议

---

## 🔧 进阶用法

### 自定义数据源

```python
from data.fetch_data import fetch_klines, save_data

# 获取更长时间的历史数据
df = fetch_klines(limit=5000)
save_data(df, "my_data.csv")
```

### 加载本地数据训练

```python
from data.preprocess import load_and_prepare

X, y, scaler = load_and_prepare("my_data.csv")
```

### 批量预测（回测）

```python
from prediction.predictor import BTCPredictor

predictor = BTCPredictor(model_path="models/best_model.pth")
predictions = predictor.predict_batch("data/klines_20260329.csv")

# 分析预测准确率
correct = sum(1 for p in predictions 
              if (p['probability'] > 0.5) == (p['actual_direction'] == 1))
accuracy = correct / len(predictions)
print(f"预测准确率：{accuracy*100:.2f}%")
```

---

## 📈 性能优化建议

1. **增加数据量** - 使用更多历史数据训练（建议 1 年以上）
2. **调整超参数** - 尝试不同的 hidden_size、layers、learning_rate
3. **特征工程** - 添加更多特征（资金费率、多空比、情绪指标）
4. **集成学习** - 训练多个模型取平均
5. **迁移学习** - 用其他币种数据预训练

---

## 📝 待办事项

- [ ] 添加资金费率特征
- [ ] 添加市场情绪特征
- [ ] 支持多币种预测
- [ ] 添加可视化界面
- [ ] 集成到实盘交易系统

---

## 🦞 关于

本项目由 AI 龙虾 🦞 开发，用于 BTC 趋势预测研究。

**GitHub**: https://github.com/chenweidu666/BTC-LSTM-Predictor

**License**: MIT

---

*投资有风险，入市需谨慎*
