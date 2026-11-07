# 本地 PyTorch 训推

当前正式配方为 TokenMixer + DIN，入口 `--model tokenmixer --din`；保留 OneTrans 和 MLP。使用 Jigsaw，从仓库根目录执行。模型与日志保存在 `output/<model>/<UTC+8 时间戳>/`。

## TokenMixer + DIN 当前配方

复用下表864维非序列特征（包括160维UTC时间embedding），追加历史mean128、history×item128、user×history128、DIN兴趣128，共1376维。1376恰好切成16×86，不需要补零，每块独立映射128维，经4层TokenMixer（每层额外保留输入输出残差x+Block(x)，训练逐block重计算激活，保留dropout RNG），flatten2048进入三个独立tower。候选/历史共享类别及QR商品ID表；保留所有有效历史事件。

历史每位置288维（item128+category64+type32+source32+QR32），共享MLP288→256→256，再线性投影128；masked mean是这些事件表示的均值。DIN将候选商品与事件表示分别投影32维，以[q,k,q-k,q*k]构造局部激活，MLP128→32→16→1、PReLU后对128维历史表示加权求和，不用softmax。query/key第一层线性项按候选/历史预计算，历史编码按feed复用；候选按2048分块，训练重计算局部激活。此实现为项目适配，不宣称完整复现DIN论文训练配方。

BCE + 0.1 ListMLE(pos) + 0.1 pairwise，pairwise使用独立tower；每feed均匀随机抽64对，pos差≥4。只有train有pos，验证与test只输出主tower。网络dropout0.1、AdamW LR0.001、5%warmup+cosine、2epochs，结束自动完整test推理。`--din`默认关闭以兼容旧TokenMixer checkpoint；本轮显式启用。

## 输入与结构

下表记录OneTrans统一causal attention的原配方。TokenMixer+DIN复用其864维非序列编码，增加上述历史特征；其token切分与输出维度以上一节为准。

| 非序列特征（按 concat 顺序） | 维度 | 编码 |
| --- | --- | --- |
| f1–f5 | 160 | 每项 train-only 分位数分128桶，各 nn.Embedding128×32 |
| category_id | 64 | nn.Embedding，与历史共享 |
| 官方 user_embedding | 128 | float32 后 L2 归一化 |
| 官方 item_embedding | 128 | split 内索引取128维向量，归一化 |
| user × item | 128 | 逐元素乘积 |
| 历史长度 | 32 | 10个桶，边界0/1/2/4/8/16/32/64/128 |
| 小时、分钟、星期、是否工作日、一周中的小时 | 160 | 24/60/7/2/168 桶，各32维；UTC；工作日为周一至周五 |
| UI cosine | 32 | 归一化user/item余弦相似度在[-1,1]分128桶 |
| item_id | 32 | QR双表相加，候选/历史共享 |
| **总计** | **864** | 连续切成4×216，各独立 Linear216→256 |

QR 将正 int32 ID 拆为商和余数，两张32维表相加。索引对唯一，但组合向量共享参数，表达能力不等同于每个ID独立一行。它与官方128维商品向量同时使用。

每个历史位置 concat `[官方item128, category64, event_type32, event_source32, item_id32]`，共 **288维**，经共享 `Linear288→256 → GELU → Linear256→256`。历史保留全部最多256条，官方新到旧顺序不变。候选/历史的缺失标记和有效向量比例不进入新版模型。

逻辑 token 顺序为 `[4个query; 可学习SEP; 历史事件]`，最大4+1+256个。采用向右可见的 causal mask：query可读自己及右侧query、SEP、历史；SEP可读自身及历史；历史只能读自身及更旧事件，不能读query。序列内部计算会反转有效部分来使用下三角内核，输出仍恢复新到旧。

每层query使用token-specific Q/K/V、FFN，SEP/历史使用共享Q/K/V、FFN。query与历史在同一次attention内联合softmax，不是独立self与cross结果相加；K/V取该层输入。最后query TokenMixer后展平4×256=1024，进入三套独立 `1024→256→128→1` tower。

Query及最终TokenMixer的PerTokenFFN使用SwiGLU：`down(SiLU(gate(x))×up(x))`。中间宽度704，接近原GELU1024的参数量。历史事件MLP和序列共享FFN保持GELU。旧检查点默认沿用GELU。

历史没有逐事件时间戳，无法计算真实距今时间；没有把序列下标冒充时间。

## 同组候选 KV cache

同一feed的SEP和历史每层只编码一次。BF16把全部有效事件和SEP按feed打包，K/V各为 `(sum(L_b+1),4,64)`，不计算历史padding；FP32保留padded参考实现。候选query独立。训练保留梯度，更新参数后重新构建缓存；推理同批feed可分候选块复用。

BF16 CUDA训练用两类FlashAttention tile覆盖query前缀和共享历史；独立query self每块最多8192个候选，通过共同log-normalizer组合，反向也使用共同输出和共同归一化，保持统一attention语义。FP32验证/推理使用显式联合keys的SDPA。`prepare_history_cache(batch)`返回所有层缓存。

## 学习率

最新要求为 **2 epochs + warmup/cosine**。AdamW 初始峰值 lr=0.001、weight_decay=0.0001；总 optimizer steps 前 5% 线性 warmup，之后 cosine 衰减至峰值的 5%（0.00005）。可用 `--warmup-ratio`、`--min-lr-ratio` 调整，`--lr-schedule constant` 切换固定学习率。

步数按每个 worker 的尾 batch 和各 epoch 分片顺序精确计算；调度计划与 global_step 随检查点保存。恢复沿用原计划，额外延长到原计划之外的步数维持最低 LR。TensorBoard 的工程指标中记录逐步 LR。

## 双卡与融合归一化

`--replica-device cuda:2` 启用第二张空闲卡；当前全局 batch1536 按完整 feed 拆成两份，各卡768。主 BCE 按候选数、辅助 ListMLE 按有效 feed 数加权，两卡梯度合并后只执行一次 AdamW，保留同样全局batch和LR步数。checkpoint只保存主模型，同时保存两卡CUDA RNG；恢复时可显式添加 replica-device。

本机GPU1→GPU2直接peer copy未通过数值检查，因此参数、梯度和结果全部经pinned CPU主存中转。初次forward/backward按卡顺序编译，后续并发，避免PyTorch2.5多线程FX tracing冲突。推理和验证使用主卡。

旧版128维使用fused RMSNorm；新版256维统一attention使用原生FP32 RMSNorm计算后转回BF16，避免双卡线程共享Triton autotuner的竞争。验证/test使用FP32。

单卡融合短测25.8万候选/秒、25.47GiB分配峰值。双卡正式第2轮持续记录42–44万候选/秒，约原19.8万的2.2倍，各卡峰值约13.4GiB；按吞吐折算9–10分钟一轮，完整耗时仍以最终run.json为准。

## 三任务与扰动

- 主tower：target BCE，提供验证指标和test推理分数。
- ListMLE tower：按train.pos升序排列，每feed按M−1归一化，再对有效feed平均。
- 独立pairwise tower：同feed内随机抽64对，要求原始pos差≥4，优化 `softplus(score_靠后−score_靠前)`。在全部符合条件的pair中均匀有放回抽样，不构造M×M矩阵。
- 总loss：`main_bce + 0.1×listmle + 0.1×pairwise`。pair数量、间隔、权重分别可配置。
- val/test没有pos，跳过两项顺序监督，不能用随机列表下标补标签。推理只取主tower。

训练默认user向量按user_embedding_dropout整条置零，官方item向量和QR商品ID向量均按item_embedding_dropout置零（默认各0.1，当前低扰动配方各0.02）。候选/历史的官方向量与QR输出分别独立抽mask；候选和历史共用同一套ID及类别表，同一feed的user扰动共享。eval/test关闭扰动。

## Train-eval

先确认设备余量并做少量吞吐、显存检查，再运行完整实验：

```bash
conda activate Jigsaw
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m src.train --mode train-eval --model onetrans --device cuda:1 --replica-device cuda:2 --epochs 2 --batch-size 4096 --workers 8 --tokens 4 --width 256 --attention-heads 4 --amp bf16 --compile --lr-schedule warmup_cosine --warmup-ratio 0.05 --min-lr-ratio 0.05
```

后续新训练默认跑 2 个 epoch；使用完整官方 train 训练，每个 epoch 后在完整 val 验证。完整训练结束自动用 `best.pt` 推理全部 test，在当前运行目录生成并校验 `submission.parquet`。默认按完整 NDCG 保存 `best.pt`，连续 2 个 epoch 未提升则停止；`--monitor ndcg_at_30` 可更改选模指标。`--model mlp` 切换为共享相同输入编码的 MLP 基线。

当前加速路径：Arrow 列式批量解码、8 workers、BF16、CUDA fused AdamW、feed 级 K/V 共享、变长 FlashAttention、局部 `torch.compile`（事件 MLP、序列/查询 FFN、最终 query TokenMixer）。Embedding lookup/backward 保留原生实现。PFFN bias 跟随激活 dtype，最终输出层保留 FP32；验证与 testinfer 默认关闭编译、使用 FP32，独立于训练精度。

OneTrans 初版单卡预检为19.3万候选/秒、34.49GiB；最新双卡融合版见上方双卡章节。短测不能替代完整 epoch 计时；每轮实际耗时及完整验证结果记录在 `run.json`。

此前 TokenMixer 基线于 2026-11-01 已测得首轮完整训练 **502.01 秒**（全部 3,151,208 feed、246,745,836 候选），完整验证 **9.81 秒**，合计约 **8 分 32 秒**。该结果来自 `output/tokenmixer/20261101_123451/`，不是有限样本外推。

## 全量训练与 testinfer

使用一个已完成、无样本/步数截断的 train-eval 运行目录：

```bash
python -m src.train --mode full --from-eval output/tokenmixer/EVAL_TIMESTAMP --device cuda:1
```

从随机初始化开始，用 train+val 重训；模型、学习率等配方继承该 eval 运行，默认训练 2 个 epoch；仅显式传入 `--epochs` 时更改轮数，不沿用旧运行的最佳轮数。训练后自动用主 tower 推理全部 test，输出并校验 `submission.parquet`。full 不再用 val 选模型。

也可显式固定配方和 epoch 数，例如 `--mode full --model onetrans --epochs 2`。单独推理：

```bash
python -m src.infer --checkpoint output/tokenmixer/RUN_TIMESTAMP/best.pt --device cuda:1
```

train-eval 和 full 完整训练结束均自动预测 test（分别使用 best.pt / last.pt），推理 batch 不超过 1024，以保留 FP32 显存余量；有限训练诊断不自动导出。只有预测和校验完成后，整次运行才标记 complete。

默认写入检查点所在运行目录，已有预测文件不会覆盖；可用 `--output` 指定新目录。完整推理严格检查三列类型、有限分数、候选键唯一及模板完整覆盖。

## 当前编译范围

带激活重计算的TokenMixer及DIN保持原生执行；仅独立事件MLP使用torch.compile，入口共享RLock。PyTorch2.5在“编译体外checkpoint”和“checkpoint整体编译”两种组合中均出现实际错误，这些路径已从入口移除。模型、完整历史、BF16、双卡分feed及全局batch保持原配方。

## 优化器

新训练默认所有可训练nn.Embedding表使用RMSprop（alpha0.99、eps1e-8、momentum0、weight_decay0），其他参数使用AdamW；官方user/item向量是输入数据。两组默认共用峰值LR和warmup/cosine进度，可用`--embedding-lr`独立设置峰值；TB在工程/优化器面板记录两组学习率。全局梯度裁剪后两个优化器各更新一次，两组状态均保存到同一checkpoint。

更换优化器必须新建运行、随机初始化模型及优化器，从第0步开始；`--resume`拒绝更换embedding优化器。旧checkpoint未记录embedding优化器时仍按全参数AdamW恢复，以保持同配方续训兼容；`--embedding-optimizer adamw`可用于新训练采用全AdamW配方。

## 独立优化开关与完整验证预测

以下特征开关默认关闭，主排序损失默认softmax；均记录进config/checkpoint/report。新配方从0训练，不在resume时修改：

| 开关 | 行为 |
| --- | --- |
| `--target-rank-weight 0.05` | 主塔在BCE之外学习按feed等权的正例集合softmax损失；只用target。原pos ListMLE和独立pairwise仍保留 |
| `--target-rank-kind ndcg_pairwise` | 替换主塔target-rank项：每feed抽64个正负target对，以当前完整排名的ΔNDCG加权、样本对权重归一化；默认softmax。权重仍由target-rank-weight控制，pos辅助pairwise不变 |
| `--numeric-residual` | 保留分桶embedding，另在完整DIN底座末尾追加原始f1–f5；1376=16×86变为1381→补11维→16×87 |
| `--scale-embedding-fields` | 每个学习embedding字段乘1/sqrt(d)，包括QR两表之和；官方unit user/item、raw数值、交互及历史聚合输出不额外缩放 |
| `--similarity-residual` | 仅DIN：追加连续UI cosine、历史max cosine及其逐feed zscore；复用同次扰动，按128-feed块FP32计算，raw5开启时总1384维补8维到16×87 |
| `--history-position-dim 32` | 仅DIN：每个历史事件增加32维位置embedding，位置1为最新、最多256、padding=0；事件MLP输入288→320，输出仍为256维。表示新旧序号，不是时间间隔。该表使用RMSprop，并遵循字段缩放开关 |
| `--recent-history-window 16` | 需开启similarity-residual：复用同次候选×历史cosine矩阵，追加最近16位置的max及其逐feed zscore；无有效历史时max为0。仅作为主网络输入，不修改预测分数。F底座1384→1386，补零8→6，仍16×87，参数量不变 |

`--device cuda:0 --replica-device cuda:0 --batch-size 4096`会顺序计算两个2048-feed半batch，按全局候选数/有效feed数合并梯度，仅更新一次；保持有效batch4096，配置标记`sequential_replica=true`。跨卡时仍经CPU中转，保持原有并行方式。单卡与双卡CUDA随机流不同，不能宣称逐参数相同的随机对照。

新训练每个epoch完整val保存`validation_epoch{epoch}.parquet`，包含feed_id/item_id/target/score；完整覆盖与指标计算成功才原子落盘。已完成的旧运行可补充best.pt的完整val预测，不更新TB current或重训：

```bash
python -m src.infer --split val --checkpoint output/tokenmixer/RUN_TIMESTAMP/best.pt --device cuda:1 --batch-size 128 --workers 2 --no-tensorboard
```

结果为该运行目录的`validation_predictions.parquet/json`，记录checkpoint、epoch、全部指标和覆盖。已经存在的文件不会覆盖。

2026-11-03用户要求只迭代单模型。后续只使用同一模型主tower的输出进行完整val评估和test预测，不执行多模型融合或预测后相似度分数叠加。已有ensemble文件仅保留作历史记录，不纳入单模型目标成绩。`--similarity-residual`作为网络输入特征由该模型统一学习，与已停用的预测后分数校正不同。

### 主任务NDCG pairwise选项

`ndcg_pairwise`仅使用target：每个同时有正例与负例的feed独立均匀有放回抽64对；当前score从大到小排序，同分时负例优先，padding不参与抽样。每对使用完整NDCG交换权重`delta = abs(D(rank_pos)-D(rank_neg))/IDCG`，其中`D(r)=1/log2(r+2)`；在该feed的64对中将delta归一化，再乘`softplus(score_neg-score_pos)`。排名、抽样和权重不反传，最后按有效feed等权平均；无正例或无负例feed跳过该项，仍保留BCE。

这是受[LambdaLoss论文](https://marc.najork.org/papers/cikm2018.pdf)启发的抽样、feed内归一化变体，不是论文完整复现；归一化也改变了原始损失尺度，不声称拥有其理论保证或已提升项目NDCG。双卡按有效feed数汇总；CUDA抽样使用已随checkpoint保存的各卡RNG。每轮完整val/test仍只使用主tower输出；ListMLE和独立pos-pairwise塔保持原定义。

新选项仅完成代码审查、Black120及语法检查，未运行tests或额外benchmark。实际数值/吞吐与效果由新的正式完整train-val确认，不能把排队当作已验证。

## TokenMixer 训练与推理加速（2026-11-03）

`--fast-execution`启用按feed分组的DIN矩阵乘法及批量token投影。保留原参数名/形状、完整历史、特征、三塔、损失及优化器；仅重排算术执行，浮点舍入不保证逐位相同。仅支持TokenMixer；默认关闭以兼容旧检查点，可显式从同配方last.pt恢复启用，保留权重、优化器、LR、RNG和数据游标。后续自动test显式沿用本次执行开关，即使best.pt来自启用前的epoch。

```bash
python -m src.train --resume output/tokenmixer/RUN_TIMESTAMP/last.pt --fast-execution
python -m src.infer --checkpoint output/tokenmixer/RUN_TIMESTAMP/best.pt --output output/tokenmixer/NEW_TIMESTAMP --device cuda:1 --batch-size 1024 --workers 8 --fast-execution
```

验证和test只计算主tower，仍为eager FP32。TokenMixer test不再误用OneTrans的256-feed限制；新训练入口的TokenMixer val至多1024feeds/batch，降低多实验并发时的峰值显存。单卡顺序副本直接在同GPU复制权重和累加梯度；跨GPU仍经CPU。DIN/Mixer保持原生执行，仅event MLP编译，未重新引入不稳定的编译/checkpoint组合。新训练可在每卡batch较小时省略`--checkpoint-mixer`，保存Mixer中间激活以减少反向重算；DIN重算保留，编译范围不随此开关扩大。需在正式运行确认显存余量，当前大batch运行不因此重启。

独立infer CLI默认设置expandable_segments（保留用户已有配置），减少不同候选数造成的显存碎片。不可仅按allocated峰值安排额外同卡推理；还须留出reserved及运行中训练峰值空间。额外完整推理安排空闲GPU，避免抢占训练显存。

完整val/test逐候选对齐证据及实际耗时见doc/experiments.md；共享GPU的两次耗时不构成公平加速比。BF16训练算术重排可能改变后续轨迹，切换点记录在各run.log，不能称为逐参数一致的旧路径续训。

## 日志与恢复

| 文件 | 内容 |
| --- | --- |
| `run.log` | UTC+8 时间、训练/验证/推理进度和异常 |
| `metrics.jsonl` | 可程序读取的逐次指标 |
| `tensorboard/` | Loss / 效果指标 / 工程指标三个分组，不写 start 和重复计数 |
| `config.json`、`run.json` | 固定配置、阶段状态、epoch 历史和最佳结果 |
| `last.pt`、`best.pt` | 最近安全检查点与最佳验证检查点；full 只保存 last |
| `report.md` | 每轮验证和最终推理后更新，包含全部效果、loss及工程指标 |
| `inference.json` | 推理耗时、输出路径和覆盖检查结果 |

训练监控总损失、主 BCE、辅助 ListMLE、独立pairwise、抽样pair数量、有效辅助 feed 数、学习率、梯度范数、正例比例、均值预测、数据/计算耗时、feed/候选吞吐、epoch ETA、进程 RSS 和本进程显存。验证报告完整 NDCG、@10、@30、Logloss、ROC-AUC、AP、Brier；NDCG 按 feed 等权、并列时负例优先。

```bash
python -m src.train --resume output/tokenmixer/RUN_TIMESTAMP/last.pt --device cuda:1
```

默认每 1000 个训练步和每个 epoch 保存 last，可用 `--checkpoint-every` 调整。恢复包括优化器、scaler、随机状态、epoch 和 batch 游标；会回放读取已完成的批次定位游标，不重复优化。保持 batch size、workers、数据与软件设置，才能保持原数据顺序；CPU 中断恢复的逐参数一致性已测试。

固定 TensorBoard tmux：`avito-tensorboard`，地址 `http://10.176.40.144:6006/#custom_scalars`。`src.tensorboard_service` 监视 `output/current`（各模型也保留自己的 current），新正式运行启动时会自动切换，并重启 TB 子进程清除旧运行缓存。只显示当前运行；完整日志继续保存在各自目录。

## 诊断边界

`--max-feeds`、`--val-max-feeds`、`--max-steps` 用于有限数据诊断，运行会标记 `limited=true`，不能作为 `--from-eval` 的正式依据。`infer --max-feeds` 只产生 `diagnostic_predictions.parquet`，标记 `submission_ready=false`。

用户最新要求：不再编写或运行tests测试代码、不再额外跑benchmark，直接正式train-val；训后test集推理仍保留。

## 历史OneTrans速度配置

用户已接受4个query tokens、batch4096、双卡BF16、8workers、2epochs的配置，保持其跑完，不再为10分钟目标重启。实际稳态23–24万候选/秒，折算17–18分钟/epoch；10分钟未达成。运行目录为 `output/onetrans/20261101_170112_769404/`，完整计时、验证及推理结果以该目录报告为准。
