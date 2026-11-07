# Avito ML Cup 2026：个性化推荐排序

中文整理日期：2026-11-01。来源：[比赛主页](https://ods.ai/competitions/avitotechrecsys2026)、[FAQ](https://ods.ai/competitions/avitotechrecsys2026/faq)。字段与文件结构见 [data.md](data.md)。

## 任务

为 Avito 推荐信息流中的候选广告排序。每条信息流以 `feed_id` 标识，候选集合已经给定；需要为其中每个 `item_id` 预测一个 `score`，让更可能触发目标行为的广告排在前面。

`target` 是二值标签：发生过至少一种目标行为记为 1，否则为 0。目标行为权重相同，多次行为不增加标签值。只比较同一条 feed 内的分数，不要求不同 feed 的分数可比，也不要求分数是概率。

可用信息包括 5 个归一化数值特征、类别、128 维用户和商品向量，以及最多 256 条用户历史事件。官方按时间提供 train、val、test，test 标签不公开。

## 评估指标

主页声明主指标为 **NDCG**，越高越好。先计算每条 feed 的 NDCG，再对所有 feed 等权平均；候选多的 feed 不会获得额外权重。

对按预测分数降序排列的一条 feed，二值相关度下：

$$
\operatorname{DCG}@K=\sum_{i=1}^{\min(K,M)}\frac{\operatorname{target}_i}{\log_2(i+1)},\qquad
\operatorname{NDCG}@K=\frac{\operatorname{DCG}@K}{\operatorname{IDCG}@K}.
$$

其中，`M` 是候选数，位置 `i` 从 1 开始；IDCG 是把全部正例放在负例前得到的理想 DCG。完整 NDCG 使用所有候选。

- **相同分数采用悲观排序**：负例排在正例前。
- 没有正例的 feed，NDCG 记为 1；官方数据构造会筛选含正例的 feed。
- **截断位置存在表述差异**：主页给出完整 DCG 求和，却在解释中写了 `IDCG@30`。官方评估代码同时输出完整 NDCG、NDCG@10、NDCG@30，CatBoost 基线使用 @30 选择模型。因此本地应分别报告这些指标，不能将完整 NDCG 与 @30 混称。

指标实现可查本仓库 [官方评估代码](../official/calculate_metrics.py)。

## Public 与 Private

test 按 feed 时间划分：较早的约 30% 为 Public，其余约 70% 为 Private。划分点按 feed 数量尽量接近 30%；相同时间戳的 feed 不会被拆开，因此比例可能略有偏差。

一条 feed 整体属于其中一部分，但用户不保证互斥。比赛期间展示 Public 分数，最终排名使用 Private 分数。

## 提交文件

本比赛在 ODS 平台提交 **Parquet 预测文件**。无需平台执行推理代码，不要求使用官方基线的 Python 或依赖版本。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `feed_id` | `int32` | 测试 feed 标识 |
| `item_id` | `int32` | 候选广告标识 |
| `score` | `float32` | 排序分数，越大越靠前 |

使用 `sample_submission.parquet` 的全部 `(feed_id, item_id)` 键，只替换 `score`。每个键恰好一行，不得遗漏、增加、重复；字段不能含 null，分数不能是 NaN 或无穷。行顺序不影响结果，分数不必位于 `[0,1]`。

每天最多提交 5 次；组队后为全队共用 5 次，未使用次数不结转。主页还要求前 10 名在最终方案选择后提交解决方案材料，具体以正式规则为准。

## 官方基线

| 方法 | 使用的信息 | 本地选模方式 |
| --- | --- | --- |
| CatBoostRanker / YetiRank | 数值、类别、用户与候选向量的相似度 | val NDCG@30 |
| Transformer | 用户向量与同一 feed 的候选集合 | val Logloss |

两套基线均生成上述三列预测。基线及参考环境已保存在 `official/`。参考环境用于运行作者代码，不是提交要求；本项目使用 Jigsaw，自有 `avito.ipynb` 对 CatBoost 输入格式作了兼容适配。

## 时间、组队与奖励

以下为页面列出的时间，页面未明确时区，**不能直接当作北京时间**。

| 事项 | 页面信息 |
| --- | --- |
| 比赛开始、开放提交 | 2026-10-28 |
| 报名截止 | 2026-12-08 23:59，来自 FAQ |
| 组队截止 | 12 月 8 日，FAQ 未给出具体时刻 |
| 主页结束日期 | 2026-12-09 |
| 最终方案选择窗口 | 2026-12-09 00:00–12:00，来自主页 |

FAQ 将最终选择日期写为 2025-12-09，与本届主页不一致，属于明显年份冲突；实际截止时间应在操作前以平台最新说明核实。报名截止也不应直接视为预测文件提交截止。

最多 4 人组队。奖金为第一名 750,000 卢布、第二名 450,000 卢布、第三名 300,000 卢布。讨论渠道为[官方 Telegram 群](https://t.me/+3bJmHVSkojJkNGJi)。

FAQ 表示额外数据、训练/推理速度和模型大小没有额外限制，不提供额外计算资源。正式[参赛规则 PDF](https://storage.yandexcloud.net/ds-ods/files/data/docs/competitions/AvitoTechMLCup_2026_Recsys/Rules_Avito_tech_ML_cup_2026.pdf)第 2.1 条另有年满 18 岁、俄罗斯公民等资格要求；资格与第三方软件许可条件以该规则为准。
