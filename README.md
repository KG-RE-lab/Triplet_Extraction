# ：基于表格填充的联合关系三元组抽取

> 采用**表格填充（Table Filling）**范式，结合 BERT、跨注意力、图传播推理与多尺度表格增强模块的联合关系三元组抽取模型。

---

## 目录

- [项目结构](#项目结构)
- [环境与依赖](#环境与依赖)
- [数据准备](#数据准备)
- [预训练 BERT](#预训练-bert)
- [使用方法](#使用方法)
- [模型与评估说明](#模型与评估说明)
- [致谢](#致谢)

---

## 项目结构

```
CAMGT/
├── README.md              # 本说明
├── bert-large-cased/      # 预训练 BERT 模型
├── dataset/               # 数据集目录
│   ├── NYT/               # NYT 数据集
│   ├── NYT_star/          # NYT* 数据集
│   ├── WebNLG/            # WebNLG 数据集
│   └── WebNLG_star/       # WebNLG* 数据集
└── code/                  # 训练与推理代码
    ├── run.py             # 入口脚本（训练/测试）
    ├── main.py            # 训练与测试主逻辑
    ├── train.py           # 训练循环与优化器
    ├── model.py           # 模型定义
    ├── modules.py         # 跨注意力、推理、表格增强等子模块
    ├── dataloader.py      # 数据加载
    ├── evaluation_utils.py # 评估（严格三元组 / 实体对 / 关系识别）
    ├── util.py            # 工具函数
    └── requirements.txt   # 依赖
```

---

## 环境与依赖

进入代码目录并安装依赖：

```bash
cd code
pip install -r requirements.txt
```

主要依赖：**PyTorch**、**transformers**、**bert4keras** 等，详见 `code/requirements.txt`。

---

## 数据准备

| 数据集 | 参考来源 |
| :----- | :------- |
| **NYT / NYT\*** | [CasRel](https://github.com/weizhepei/CasRel)、[CopyRE](https://github.com/xiangrongzeng/copy_re) |
| **WebNLG / WebNLG\*** | [JointER](https://github.com/yubowen-ph/JointER)、[ETL-span](https://github.com/yubowen-ph/JointER) |

每个数据集目录下需包含：

- 必需：`train.json`、`dev.json`、`test.json`、`rel2id.json`
- 可选：`1.json`–`5.json`、`epo.json`、`normal.json`、`seo.json` 等切分

---

## 预训练 BERT

将 **BERT-Large-Cased**（或兼容 BERT）放在 `bert-large-cased/` 下，或通过参数指定路径。所需文件：

| 文件 | 说明 |
| :--- | :--- |
| `vocab.txt` | 词表 |
| `config.json` | 模型配置 |
| `pytorch_model.bin` | PyTorch 权重 |

可从 [Hugging Face · bert-large-cased](https://huggingface.co/bert-large-cased) 下载。

---

## 使用方法

> 所有命令均在 `code/` 目录下执行，或确保工作目录与 `run.py` 中的相对路径一致。

### 训练

**按数据集训练示例：**

```bash
# WebNLG
python run.py --dataset=WebNLG --file_id=WebNLG --train=train --cuda_id=0

# WebNLG*
python run.py --dataset=WebNLG_star --file_id=WebNLG_star --train=train

# NYT
python run.py --dataset=NYT --file_id=NYT --train=train

# NYT*
python run.py --dataset=NYT_star --file_id=NYT_star --train=train
```

**常用参数：**

| 参数 | 说明 | 默认值 |
| :--- | :--- | :----- |
| `--cuda_id` | GPU 编号 | `"3"` |
| `--train_batch_size` | 训练批大小 | `6` |
| `--learning_rate` | 学习率 | `2e-5` |
| `--num_train_epochs` | 训练轮数 | `100` |
| `--bert_model_path` | BERT 权重路径 | `../bert-large-cased/pytorch_model.bin` |
| `--base_path` | 数据集根目录 | `../dataset` |
| `--loss` | 损失函数 | `FocalLoss_plus`（可选 `CE`、`FocalLoss`） |
| `--ablate` | 消融：关闭的模块 | 不传为完整模型；可多选 `cross_attention`、`reasoning`、`table_enhance` |

**消融示例：**

```bash
# 仅关闭跨注意力
python run.py --dataset=WebNLG --train=train --ablate cross_attention

# 关闭跨注意力 + 表格增强
python run.py --dataset=WebNLG --train=train --ablate cross_attention table_enhance
```

### 测试 / 评估

```bash
python run.py --dataset=WebNLG --file_id=WebNLG --train=test
```

测试会在 `test`、`1`–`5`、`epo`、`normal`、`seo` 等划分上运行，输出三类指标：

| 指标 | 说明 |
| :--- | :--- |
| **严格三元组**（Strict Triple） | F1 / P / R |
| **实体对**（Entity Pair） | F1 / P / R |
| **关系识别**（Relation） | F1 / P / R |

结果写入 `dataset/.../results/<file_id>/out.txt` 及对应预测 JSON。

---

## 模型与评估说明

- **CAMGT**：在 BERT 编码基础上，通过 Subject/Object 双路表示、跨注意力交互、图传播推理（Reasoning）和多尺度表格增强（Table Enhance）生成表格表示，并解码实体对与关系。
- **评估指标**：严格三元组要求头尾实体与关系均正确；实体对与关系识别作为辅助指标。

---

## 致谢

- 部分实现参考 [bert4keras](https://github.com/bojone/bert4keras)。
- 数据集与设定参考 CasRel、CopyRE、JointER、ETL-span 等工作。
