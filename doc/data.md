# Avito ML Cup 2026：数据说明

中文整理日期：2026-11-01。主要来源：[官方数据页](https://ods.ai/competitions/avitotechrecsys2026/dataset)及[压缩包内 README](../data/readme.md)。比赛任务、指标和提交规则见 [overview.md](overview.md)。

## 下载与文件结构

| 文件 | 官方标注大小 | 内容 |
| --- | --- | --- |
| [competition.zip](https://storage.yandexcloud.net/avitotech-mlcup-2026/competition.zip) | 19.2 GB | train、val、test、商品向量、README、提交模板 |
| [sample_submission.parquet](https://storage.yandexcloud.net/avitotech-mlcup-2026/sample_submission.parquet) | 16.8 MB | 全部测试候选的提交模板，分数为 0 |
| [baseline.zip](https://storage.yandexcloud.net/avitotech-mlcup-2026/baseline.zip) | 37.2 KB | CatBoost、Transformer、评估代码和参考依赖 |

主压缩包的顶层为 `competition/`。本仓库去掉这一层，直接解压到 `data/`：

```text
data/
├── readme.md
├── sample_submission.parquet
├── train/
│   ├── impressions/
│   │   ├── part-*.parquet
│   │   └── _SUCCESS
│   └── item_embeddings/
│       ├── data.memmap
│       └── data.meta
├── val/                         # 与 train 相同的目录结构
└── test/                        # 与 train 相同的目录结构
```

`impressions` 的 Parquet 使用 ZSTD 压缩，**一行是一整条 feed**，不是一个候选广告。读取完整 split 必须包含它的全部分片。`_SUCCESS` 是空的写入成功标记，不是数据文件。

`data.memmap` 为商品向量矩阵，`data.meta` 为 JSON，记录 `dtype`、`shape`、`order`。隐藏 test 标签不在公开压缩包中。

本地已读取独立提交模板：共 **2,904,806 行**，字段类型为 `feed_id:int32`、`item_id:int32`、`score:float32`。这是候选行数，不是 feed 数。

本地全量扫描得到的规模如下。feed 数、分片数、向量行数与压缩包 README 一致；候选行数为共同展开候选列表后的行数，向量表还覆盖历史商品：

| split | feed 数 | 候选行数 | Parquet 分片数 | 商品向量行数 |
| --- | ---: | ---: | ---: | ---: |
| train | 3,151,208 | 246,745,836 | 512 | 81,557,944 |
| val | 14,423 | 1,389,971 | 64 | 2,516,711 |
| test | 28,595 | 2,904,806 | 64 | 4,710,948 |

候选列表长度全量统计（2026-11-04，仅读取各split的item_id列表长度，保留原始候选）：

| split | 最小 | 中位数 | P90 | P95 | P99 | 最大 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 5 | 48 | 213 | 253 | 281 | 300 |
| val | 5 | 67 | 247 | 277 | 291 | 299 |
| test | 5 | 72 | 256 | 279 | 292 | 300 |

train 有 5,446,872 个正例候选，val 有 21,983 个；test 标签不公开。651 个文件均通过 ZIP CRC 和大小检查，640 个 Parquet 分片均完成内容扫描，测试候选键与模板完全一致。

## 时间划分与样本构造

| split | 数据日期 | 用途 |
| --- | --- | --- |
| train | 2026-09-06 至 09-19 | 训练，含标签和原始位置 |
| val | 2026-09-22 至 09-24 | 本地验证和调参，含标签 |
| test | 2026-09-26 至 09-29 | 最终预测，不含标签 |

`timestamp` 是 feed 记录时间，使用秒级 Unix 时间戳，类型为 `int32`。

上表日期按官方页面保留。下面是原始 `timestamp` 的本地实测边界，明确转换为 **UTC+8**；不要再按页面日期的 UTC 或北京时间零点裁剪原始 split：

| split | 最早记录（UTC+8） | 最晚记录（UTC+8） |
| --- | --- | --- |
| train | 2026-09-06 05:00:48 | 2026-09-20 05:00:48 |
| val | 2026-09-22 05:00:48 | 2026-09-25 05:00:48 |
| test | 2026-09-26 05:00:48 | 2026-09-30 04:58:40 |

官方保留原始位置 `rank <= 300` 的候选；位置从 0 开始。目标行为合并为二值 `target`，然后保留至少 5 个候选、至少 1 个正例的 feed。

压缩包 README 另说明：各 split 的候选中排除了用户已经浏览过的商品。

val/test 专门取自**展示前已均匀随机打乱候选**的流量，标签来自展示后的反馈。生成公开文件时，又对候选及其全部对应字段施加同一个随机排列。train 才有原始位置 `pos`，且过滤后可能不连续；不能把 val/test 文件内顺序视为原始曝光位置。

## 字段字典

下面用 `M` 表示一条 feed 的候选数，用 `H` 表示历史事件数。`list<T>` 表示元素类型为 `T` 的列表；不同 feed 的 `M`、`H` 可以不同，且 `0 <= H <= 256`。

### Feed 级字段

| 字段 | 类型 | 每行长度 | 含义 |
| --- | --- | --- | --- |
| `feed_id` | `int32` | 标量 | feed 唯一标识，train/val/test 之间不重叠 |
| `timestamp` | `int32` | 标量 | feed 记录时间，Unix 秒 |
| `user_id` | `int32` | 标量 | 匿名用户标识，跨 split 编码一致 |
| `user_embedding` | `list<int8>` | 128 | 用户向量，属于整条 feed |

### 候选级字段

以下列表都长为 `M`，相同下标对应同一个候选。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `item_id` | `list<int32>` | 匿名商品标识，跨 split 及候选/历史编码一致 |
| `category_id` | `list<int8>` | 商品类别编码，与历史类别共用字典 |
| `f1`、`f2`、`f3`、`f4`、`f5` | 各为 `list<float32>` | 5 个已归一化数值特征，官方未公开业务语义 |
| `item_id__embedding_idx` | `list<int32>` | 当前 split 商品向量矩阵中的行号；`-1` 表示缺失 |
| `pos` | `list<int16>` | 原始曝光位置，从 0 开始；仅 train 有 |
| `target` | `list<int8>` | 二值目标标签；仅 train/val 有 |

`target=1` 表示至少发生一种目标行为，否则为 0。各目标行为权重相同，发生次数不会把标签变成大于 1 的值。`target` 不能作为输入特征。

### 历史事件字段

以下列表都长为 `H`，相同下标对应同一次历史事件，**顺序由新到旧**；历史可以为空。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `history__item_id` | `list<int32>` | 历史商品标识 |
| `history__category_id` | `list<int8>` | 历史商品类别 |
| `history__event_type` | `list<int8>` | 事件类型的类别编码 |
| `history__event_source` | `list<int8>` | 事件来源的类别编码 |
| `history__item_id__embedding_idx` | `list<int32>` | 历史商品在当前 split 向量矩阵中的行号；`-1` 表示缺失 |

候选和历史是两组不同的列表，`M` 不必等于 `H`。事件编码并非 Avito 原始事件编号，不能自行赋予“购买”“收藏”等未公开语义。

## ID、类别与向量的约定

- 用户 ID 和商品 ID 分属不同空间；正数表示已知对象，`0` 预留为缺失 ID。
- 商品 ID 在候选、历史、各 split 中共用编码；用户 ID 也跨 split 一致。feed 不重叠不代表用户或商品不重叠。
- 类别、事件类型和事件来源的编码跨 split 一致；`0` 表示未知或缺失类别。
- README 给出的编码范围：类别为 `1..53`，事件类型为 `1..10`，事件来源为 `1..30`；各 split 不保证出现全部编码。
- `f1`–`f5` 的变换只在 train 上拟合，再应用于 val/test，无需额外编码器文件。
- 每个 split 各有一套形状为 `(N,128)`、C 顺序、`int8` 类型的商品向量矩阵。同一商品在不同 split 的 `item_id` 相同，但 embedding 行号可以不同。
- **索引必须对应当前 split**。`-1` 不能直接传给 NumPy 索引，否则会读到矩阵最后一行。
- **先转成 float32 再计算相似度**，避免 int8 点积、范数等运算溢出。memmap 可按需读取，不必把整张向量表载入内存。

## 读取一条 feed 及候选向量

以下示例使用本仓库相对路径，并显式处理缺失向量：

```python
import json
from pathlib import Path

import numpy as np
import polars as pl

split = Path("data/train")
feed = pl.scan_parquet(str(split / "impressions/part-*.parquet")).head(1).collect().row(0, named=True)
folder = split / "item_embeddings"
meta = json.loads((folder / "data.meta").read_text())
embeddings = np.memmap(
    folder / "data.memmap",
    dtype=meta["dtype"],
    mode="r",
    shape=tuple(meta["shape"]),
    order=meta["order"],
)

indices = np.asarray(feed["item_id__embedding_idx"], dtype=np.int64)
present = indices >= 0
items = np.zeros((len(indices), 128), dtype=np.float32)
items[present] = embeddings[indices[present]].astype(np.float32)
user = np.asarray(feed["user_embedding"], dtype=np.float32)
```

需要把 feed 展开为候选行时，应把 `item_id`、类别、数值特征、embedding 索引及标签等**全部候选列表共同展开**，保留 `(feed_id,item_id)`，不能分别打乱或独立排序。

**训练集的候选键不保证唯一。** 全量扫描发现 train 有 1 条 feed 多次包含同一 `item_id`，共 4 个额外重复候选；val/test 未发现重复候选键。训练时保留官方候选行及顺序，不自动去重；需要关联候选时可额外保留列表内下标，避免直接按 `(feed_id,item_id)` 做未经检查的一对一连接。提交文件的键唯一要求仍然严格执行。

## 本地验证与提交

使用官方 val 做参数选择，按 `feed_id` 分组计算 NDCG，再等权平均；不要随机拆候选行。完整 NDCG 与 @10/@30 分开记录，指标表述差异见 [overview.md](overview.md)。

官方 `calculate_metrics.py` 的命令行接口要求答案具有 `leaderboard_split` 列，面向隐藏 test 的 Public/Private 计算，不能直接传入公开的嵌套 val Parquet。可先共同展开 val 候选，将预测与标签按键对齐，再调用其 `metrics(frame)` 或本项目 Notebook 的验证入口。

预测时覆盖提交模板的全部候选键，输出 `feed_id:int32`、`item_id:int32`、`score:float32`，不能包含遗漏、额外键、重复键、null、NaN 或 Inf。只上传该预测文件；本机环境和参考基线锁定版本不是提交内容。

## 完整train/val的向量缺失统计（2026-11-04）

逐分片完整扫描user_embedding及候选/历史embedding索引，未读取target或pos。user统计128维向量全零；商品统计索引-1，不把它等同于所有可能的有效索引零向量。

| 项目 | train | val |
| --- | --- | --- |
| 全零user向量/feed | 79,848 / 3,151,208（2.533885%） | 205 / 14,423（1.421341%） |
| 候选缺失索引/候选 | 7,924 / 246,745,836（0.003211%） | 214 / 1,389,971（0.015396%） |
| 历史缺失索引/历史事件 | 90,521,519 / 583,586,378（15.511246%） | 355,067 / 2,499,547（14.205254%） |

这是原始数据的缺失情况，不含训练时额外随机置零。当前旧配方对非零user/item官方向量分别以0.1概率置零；item规则也作用于历史商品。已准备0.02概率的新对照，保留网络dropout和共享embedding参数；真实缺失率本身不能证明更轻扰动必然提升。没有将新的缺失标记或有效向量比例加回模型。
