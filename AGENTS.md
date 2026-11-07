# Avito ML Cup 2026 — 个性化推荐排序

## 最新执行要求与模型配方（优先于下方旧运行记录）

- **11-07 12:08窄FFN当前PID3081136/tmux avito-rmsnorm-small-ffn**：同目录`20261107_115828_007886`，日志/tmp/avito-rmsnorm-small-ffn-int8.log，从核对完整checkpoint400恢复，新增执行选项`--item-vector-transfer int8`，模型/优化器/LR/RNG/数据游标保持。官方三split商品表data.meta均为int8，候选/历史向量在worker、拼batch、pin memory及H2D时保留原int8，到目标设备后显式转FP32再进入模型；不重新量化，非int8表回退FP32，混合dtype历史padding使用类型提升。默认float32兼容旧运行，验证/推理按checkpoint配置继承，新路径完整val/test尚未执行。正式step420 profile及至少445步更新正常，工程监控显示商品向量逻辑载荷约208MiB/5步、相同张量FP32参考的0.25；仅该部分载荷，不是全部PCIe流量。主进程RSS约3.33GiB，旧路径约5.7GiB；吞吐7.72–7.78万候选/秒与旧路径重叠，不能声称额外训练加速。旧PID3050103已退出，GPU0全量PID2797616与GPU2原宽RMSNorm PID3030616均未重启，当前仍三项；TB跟随GPU0。恢复时移除/更新AVITO_PROFILE_STEP=420。Black120/语法检查通过，无tests/独立benchmark，目标未达到。

- **11-07 12:01新增第三项：窄FFN单模型对照已从0更新至至少85步**：GPU1/PID3050103/tmux avito-rmsnorm-small-ffn，目录`20261107_115828_007886`，日志/tmp/avito-rmsnorm-small-ffn.log。相对当前RMSNorm仅学习结构expansion4→2：4层Mixer及2层候选attention的FFN中间维512→256，token16/输出128维/完整历史/全部特征/三塔保持；参数14412262→10069990。seed42、2epochs、batch1024、8workers、BF16、17表RMSprop/其余AdamW、LR0.001+warmup/cosine、完整train/val和自动test。新目录随机初始化无resume；checkpoint0两个优化器为空/history为空，checkpoint1的129组AdamW和17表RMSprop全step1、权重有限。仅复用train-only分桶边界，已核对与原宽版相同。正式step20 profile形状[77146,16,256]；step65–85约7.77–7.82万候选/秒、峰值20.9GiB，同期原宽版约6.5万/28.2GiB；跨GPU/共享负载/批次不同，不当严格因果加速或质量非劣证明。估算约53分钟/epoch，尚无完整val/test，0.4和10分钟目标均未达到。当前三项各卡一个，GPU0全量PID2797616和GPU2原宽RMSNorm PID3030616均未重启；TB跟随GPU0。AVITO_PROFILE_STEP=20已消费，若恢复须移除或更新。没有tests/独立benchmark，未上传比赛。

- **11-07 11:54当前RMSNorm PID3030616/tmux avito-rmsnorm**：同目录`20261107_110833_028236`，日志/tmp/avito-rmsnorm-worker-pairs.log，从已核对checkpoint1600恢复。pairwise的stable排序/可抽样区间/累计对数在DataLoader worker预计算，主进程仍按原feed顺序用checkpoint CPU Torch RNG逐feed randint；抽样数量、gap、相对索引和loss保留。只在启用pairwise的训练loader预取，val/test不增加该工作；计划消费后不进入模型或双卡切分。正式step1620 profile完成，至少1635更新正常；后续窗口65651/66160/66765候选每秒，仅略高于原约6.5万，不能排除共享负载/批次差异。新增TB engineering/timing/pair_sampling_seconds，主进程抽样约0.109–0.113秒/5步、占约1.8%，数据等待约0.002秒/5步，CPU抽样不是主要剩余瓶颈。旧PID3011572已退出，GPU0全量PID2797616保持运行，TB跟随GPU0；无新完整val/test，目标仍未达到。恢复时移除或更新已消费的AVITO_PROFILE_STEP=1620。Black120/语法检查通过，无tests/独立benchmark。

- **11-07 11:49当前RMSNorm PID3011572/tmux avito-rmsnorm**：同目录`20261107_110833_028236`，日志/tmp/avito-rmsnorm-packed-metrics.log，从checkpoint1400同配方恢复；模型/优化器/LR/RNG/数据游标及所有执行后端不变。每步5个loss标量与日志步梯度范数/预测均值合成一次CPU传输，保留更新前loss/梯度有限性检查，计时仍等待GPU更新完成。正式step1420 profile的cudaStreamSynchronize由8次降为4次，step1425/1430吞吐65229/65838候选每秒，与之前约6.5万重叠，尚无明确整体加速；loss/梯度有限，无新完整val/test。旧PID2986568已退出；GPU0全量训练PID2797616保持运行，TB跟随GPU0。AVITO_PROFILE_STEP=1420已消费，恢复时须移除或设未来步骤。Black120/语法检查通过，无tests/额外benchmark。

- **11-07 11:42当前RMSNorm PID2986568/tmux avito-rmsnorm**：同目录`20261107_110833_028236`，日志/tmp/avito-rmsnorm-fused-block.log，从checkpoint1100同配方恢复；mixer_norm_backend=parallel_triton_fused_block，保留DIN局部编译和compiled_dropout。新增融合FFN后的残差相加、RMSNorm与可选block skip，明确保留两处BF16舍入，权重/梯度FP32归约；反向复用现有RMS公式并向两个加法输入返回相同dz、向skip返回原gradient。正式step1120 profile及至少1135更新成功，1110/1115/1125窗口旧64236/64713/64562、新65077/65717/65591候选/秒，约1–2%；loss有小幅差异，未完成完整val/test，不声称逐参数一致或10分钟达标。aten::add调用36→28，采样批次不同，不以总profile时间跨批次作严格收益证明。旧PID2951080已退出，GPU0全量任务PID2797616不中断，TB仍跟随GPU0。
- **瓶颈补充核对**：融合前20窗数据等待比例RMSNorm0.0363%、全量基线0.0260%；GPU0/2无活动throttle原因，功耗约207/201W、限额300W，温度72/61℃；不支持将慢归因于触发功耗/温度降频。GPU2在NUMA1，线程和大部分worker已在本地CPU执行，未强改亲和性或迁移共享页缓存。以上仅当时采样，不代表长期硬件状态。

- **11-07 11:33当前RMSNorm吞吐研发PID2951080/tmux avito-rmsnorm**：同目录`20261107_110833_028236`，日志/tmp/avito-rmsnorm-ffn-dropout.log，从checkpoint700同配方恢复。先在checkpoint500启用DIN确定性group_activation局部编译（native分组/池化、无重算），正式step520 profile已完成；稳定约6.1万候选/秒。再启用mixer_ffn_kernels=compiled_dropout：保留原生native_dropout的随机mask生成，将dropout backward、GELU backward与bias梯度归约独立编译；无新参数，模型/优化器/LR/RNG/数据游标按检查点恢复，存在浮点变化，不声称逐参数或mask逐值等价。正式step720 profile及后续更新成功，step715/725旧60745/60457、新64383/64295候选/秒，约+6%；峰值约26.8GiB，loss/梯度有限。现有运行须移除或更新过期AVITO_PROFILE_STEP后才能恢复。旧PID2926908/2897204已退出，不重启。
- **用户要求profile及尽量提高SM有效利用率**：step520形状采样定位主要热点为Mixer FFN，单层[79780,16,512] BF16激活1.217GiB；4层inner dropout前反向合计131.087ms、GELU前反向117.820ms、表中两组主要FFN bmm129.290ms，不能把父算子与kernel重复求和。step720观察到融合dropout/GELU backward kernel4次合计80.040ms（不同批次，仅局部采样）。数据等待约每5步2–3ms，非主要瓶颈；从算子/形状推断访存开销显著，尚不能严格认定整模型纯带宽或纯算力受限。Nsight可用但RmProfilingAdminOnly=1、当前UID1011/CapEff0，不具备硬件计数器读取权限；不改变系统权限。当前同GPU还有其他计算任务，总util100%不代表本任务独占SM。目标0.4/10分钟仍未达到。
- **仓库已上传**：origin=`git@github.com:mango7789/Avito.git`，初始提交baaeac9已推送main并核对远端SHA；数据/output/model/凭据已忽略。22/443直连超时，仓库本地core.sshCommand使用现有127.0.0.1:18888代理连接ssh.github.com:443，未改全局SSH配置。用户要求关闭misconception-download已完成，其shell/下载子进程均退出；Avito训练/TB保持运行。

- **11-07 11:17 RMSNorm继续融合优化，当前两项正常运行**：GPU2当前PID2897204/tmux avito-rmsnorm，同目录`20261107_110833_028236`、日志/tmp/avito-rmsnorm-tiled.log；从同配方checkpoint200恢复，保留优化器/LR/RNG/数据游标，已更新到225。后端parallel_triton_fused_mix把token_mix+BF16残差加法+RMSNorm融合；保留BF16加法舍入，P为自逆置换，反向dz+P(dz)。首次融合虽copy_约101.9→59.1ms但吞吐仍5.7–5.8万；进一步对同候选完整token tile一次读取、片内transpose，避免分散二次读取。相同215/220/225窗口旧57796/57532/57570、新59891/59826/59839候选/秒，约+3.6–4.0%；loss有微小差异，不能称逐参数一致/严格因果/完整验证。当前约6万，估算完整epoch69分钟，10分钟未达成。FP32 eval/test仍原生RMSNorm。Black120/语法检查通过，无tests/benchmark。
- GPU0原最佳全量训练PID2797616保持不变，11:17 step965/首轮31.2%、约4.7万候选/秒，训后自动test；当前尚无新submission。TB指针仍跟随该全量运行。RMSNorm此前PID2866207和2885602已正常SIGINT退出，不能按旧PID恢复或重启重复任务。

- **11-07 11:10用户授权RMSNorm及算子优化，已从0实际开训**：GPU2/PID2866207/tmux avito-rmsnorm，目录`output/tokenmixer/20261107_110833_028236/`，日志/tmp/avito-rmsnorm.log。来源最佳AB学习配方，仅Mixer内8处LayerNorm改RMSNorm（epsilon1e-5，无bias），候选attention/任务塔仍原LayerNorm；16tokens/width128/depth4、主rank0.05、dropout0.05、17表RMSprop/其余AdamW、完整train-val/2epochs/自动test。随机初始化无resume，首checkpoint两优化器均step1、历史为空、8个norm weight有限；参数14412262。训练使用新增parallel_triton stride-aware RMSNorm前反向、FP32归约及权重、BF16输出；直接按输入/梯度stride读写，不做归一化内部reshape/contiguous矩阵复制。反向每程序16行并行，权重梯度FP32分块再归约；FP32 val/test用原生nn.RMSNorm。支持旧LayerNorm checkpoint默认值，不改变其结构。
- **RMSNorm初步速度证据**：step15–40约5.8万候选/秒，较上一排序增强研发任务约5.2万高约12%，当前共享GPU下折算约70–71分钟/epoch，非整轮实测。正式step10 profile RMS反向8次合计31.110ms，此前LayerNorm profile为187.417ms；模型/批次不同，仅局部初步证据。全局copy_仍约101.9ms，不能宣称消除全部模型复制。loss/梯度有限，尚无RMS完整val/test，不声称精度等价或已达到0.4/10分钟。Black120/语法检查通过，无tests/额外benchmark。
- **当前仅两项**：GPU0原最佳train+val全量训练PID2797616继续，11:10约step700/首轮22.6%，训后完整test；GPU2新RMSNorm任务。旧主rank0.1吞吐研发PID2842805已SIGINT正常退出，保留last.pt step1700，不自动恢复。TB6006仍跟随GPU0全量任务，其余查询增强/context3/LCFN保持停止。类别-only LCFN尚未开训。

- **11-07 11:03最新状态**：最佳基线train+val全量训练仍GPU0/PID2797616/tmux avito-best-full，目录`output/tokenmixer/20261107_104934_340753/`；已过step450，2epochs后自动用last.pt完整test、校验全部官方键与有限分数，生成同目录submission.parquet，当前文件尚未生成。用户再次明确要求infer test；test_infer=true/test_max_feeds=0已核对。TB6006返回HTTP200，指针跟随此全量任务。GPU2吞吐研发PID2842805/tmux avito-speed，从同配方step1500恢复，日志/tmp/avito-speed-ranking-layout.log，已实际更新到1520；其余实验仍停止，当前两项。
- **DIN分块与排序loss索引优化的实际证据**：研发任务将din_chunk_size由2048升8192，profile bmm335→167、kernel4379→2861，但稳态窗口中位5.127万→5.071万候选/秒，没有明确吞吐增益；保留8192继续单独观察。随后把主rank/ListMLE/pairwise共用的整数索引和有效feed选择在前向前一次pinned异步传输，避免输出后的重复传输和target有效数同步；损失公式、采样、模型参数不变。正式step1510 profile同步调用21→8（不同批次，仅算子证据）；相同step1515/1520窗口约5.066/5.056万→5.217/5.197万，约+3%，共享负载及浮点差异存在，不宣称逐参数等价或整轮达标。峰值约27.3GiB，loss/梯度有限；新路径尚无完整val/test。全量训练未为此重启，未加载这两项新改动。Black120/语法检查通过，无tests/额外benchmark。

- **11-07 10:53当前实际运行两项**：GPU0最佳基线train+val全量训练PID2797616/tmux avito-best-full，目录`output/tokenmixer/20261107_104934_340753/`，已到step120，约4.7万候选/秒；2epochs后自动用last.pt完整test，尚无新submission。GPU2吞吐研发沿用主rank0.1实验，PID2806325/tmux avito-speed，目录`output/tokenmixer/20261106_224512_781608/`，从step1300恢复并已到1350，约5.1万候选/秒；日志/tmp/avito-speed-gather.log。其余DIN查询增强/候选context3/LCFN均停止，类别-only LCFN尚未训练。目标active、最多3项，不融合；TB6006固定后台跟随最佳基线全量训练。
- **DIN统一gather+split工程优化已完成实际更新、未完成整轮验证**：每个query/query_part/key/key_part/values源张量统一index_select，再用split给各分组，避免逐组IndexBackward0反复生成整批源梯度；历史前缀裁剪/分组/公式/候选顺序保留，outlier仍原有有界分块。profile新step1310 cudaLaunchKernel4379次，对旧step1260的6624次减少约34%（不同批次，仅算子数量证据）。同step1315旧吞吐46936.12、新51049.95候选/秒，约+8.77%，后续约5.1万；共享负载、恢复及浮点差异存在，不声称严格因果/逐参数一致/整轮达标。最佳基线全量训练在改动前启动，未加载此路径，不重启。下一步可评估增大DIN执行分块以用显存换减少小kernel，但尚未实现或调度。AVITO_PROFILE_STEP为绝对未来global_step，未来恢复需移除该环境变量或设新的未来步骤，不能原样复用已过去的1310。Black120/语法检查通过，无tests/额外benchmark。

- **11-07用户已明确恢复优化，并另行授权最佳基线train+val全量训练及test推理**：停止限制被此次新指令覆盖，目标active，仍优先吞吐、显存不设人为限制、最多3项。GPU0的最佳AB全量重训已从0启动：`output/tokenmixer/20261107_104934_340753/`，PID2797616/tmux avito-best-full；from-eval=AB，mode=full，2epochs，3165631 feeds/epoch，batch1024/BF16/8workers、17表RMSprop/其余AdamW、原主rank0.05、无LCFN/无丰富DIN query。首个checkpoint两优化器step均1，test_infer=true，训后用last.pt完整test，不上传。执行使用DIN不重算、FFN局部编译及当时已有的索引传输合并；尚未加载随后统一gather/split改动。不得把全量train+val后的成绩当官方独立val指标。
- **正式训练性能采样已完成**：GPU2主任务排序增强版从step1250恢复，step1260采样已写原run.log（CPU1.633s、CUDA聚合1.531s，含采集开销，不是稳态计时）。出现6624次cudaLaunchKernel、316次IndexBackward0，代码审查发现DIN逐组高级索引使反向反复生成整批源张量梯度。已改为每个源张量统一index_select后split，组内仍原公式/历史裁剪/候选顺序；仅Black120/语法检查通过，正准备从step1300同配方恢复验证。全量训练保持运行，不随研发小改动重启。

- **2026-11-07 10:44用户再次要求停止全部运行，优先把吞吐优化好，显存不设人为限制**：已停止DIN查询增强（旧PID2765903）、排序损失增强（旧PID2729386）、LCFN（旧PID2748158）和TensorBoard进程组；核对无Avito训练/TB tmux及GPU进程，其他用户进程保留。三项run.json均interrupted，各last.pt和日志保留；目标状态paused，不自动恢复正式训练或排队实验。后续并行上限仍为3。当前工作优先代码/执行优化，实测受48GiB物理容量及其他用户占用约束。
- **用户要求去掉商品ID LCFN，代码已改、类别版尚未启动**：新CLI默认lcfn_fields=[category]，只建hit(2×32)/count(257×32)两表并输出64维；原QR item_id embedding保留。基于AB的底座1389+64=1453，补3维到16×91。旧两字段LCFN检查点由ModelConfig默认(category,item)兼容，字段列表统一tuple保证恢复比较；类别版为新结构，必须从0，不复用旧LCFN优化器/权重。未运行tests/benchmark或新训练，Black120/语法检查通过。
- **速度优先的准备改动**：新训练CLI的din_recompute默认off，旧checkpoint缺失字段仍按原on解释；保留已有FFN编译、BF16与索引合并。新增AVITO_PROFILE_STEP=<未来global_step>，仅在正式训练指定更新中采集CPU/CUDA算子表并写原run.log，profile步骤计时含采集开销；尚未实际采集，不能声称定位到新瓶颈或提速。后续应先采集正式步骤，再决定计算/传输/调度优化，避免小改动反复重启。当前最佳已完成基线仍AB，本地0.3773363280029708、用户反馈线上0.3678834868；历史整轮约32.7分钟，当前共享GPU下此前稳态估计90–96分钟，不混淆为同条件退化或10分钟达标。

- **11-07相似度/相对数值索引传输合并已接入DIN查询特征增强版**：原每128feeds传输counts/offsets改为一次pinned metadata传输并切片，相对数值分支同样合并counts/starts；整数索引、FP32相似度/zscore计算顺序及网络配方保持。AC在step1300检查点完整落盘后退出并同配方恢复，当前PID2765903/tmux avito-din-query，日志/tmp/avito-din-query-layout.log；step1315已更新，loss/梯度有限。恢复后稳态两窗4.23–4.32万候选/秒，与之前约4.2–4.4万重叠，尚无明确额外提速证据，不据此重启另外两项。AE仍PID2729386、LCFN仍PID2748158，三项继续；AD保持停止。TensorBoard指针恢复到LCFN运行以便查看匹配监控。Black120/语法检查通过；不跑tests/benchmark。尚无新完整val结果，0.4和10分钟目标均未达到。

- **LCFN累计监控核对到step100**：训练checkpoint候选数、LCFN分母、逐窗日志分母之和均为8046102；类别命中6950017（86.3774409024%），平均出现47.6327006543次；item_id命中与次数为0。21张RMSprop表均更新到step100。仅为首100步，不是全量train或val统计。

- **LCFN历史匹配embedding版已正式启动（第三项）**：`output/tokenmixer/20261107_103505_438292/`，GPU1/PID2748158/tmux avito-lcfn。基于最佳AB从0、seed42、2epochs、batch1024、8workers/BF16，21张nn.Embedding用RMSprop、其余AdamW。参考用户飞书LCFN选区revision1124/全文1126；Avito使用category/item_id两字段，各自hit(2×32)和count(257×32)独立表，拼128维到all embedding；底座1389→1517，补3维切16×95→16×128，4层Mixer/候选context2/三塔保留，总参数14446246。主rank0.05、DIN query仍128→32，未叠加其他两项学习配方。启用已运行的索引合并传输、DIN不重算、FFN局部编译。step20正常更新，约4.42万候选/秒（共享GPU），当前窗口类别匹配86.1281%、item精确匹配0；只是早期训练窗口，不是全量统计或效果结果。匹配监控按全部候选加权累计，train窗口/epoch、完整val和test均写日志/TB/报告，checkpoint保留epoch原始计数；新增表零项可学习，无padding_idx。无tests/benchmark、未上传。后续报告用中文实验名称。

- **11-07最新并行上限改为三个训练任务**：用户明确覆盖此前两个的限制。DIN查询特征增强版与主任务排序损失增强版继续运行；新增LCFN历史匹配embedding版作为第三项，从当前最佳AB配方随机初始化、2epochs、完整train/val和训后test，不上传、不融合。匹配率监控的分母为全部候选，分别记录类别/item_id是否在有效历史出现及平均出现次数，零ID和padding不构成匹配。

- **历史：2026-11-07 10:24恢复训练，当时上限为两个，现已改为三个**：覆盖11-06停止指令。北京时间10:24恢复AC（GPU0/PID2708527/tmux avito-din-query，last.pt step850）和AE（GPU2/PID2708522/tmux avito-rank10，last.pt step700），同配方保留模型、优化器、LR、RNG及数据游标。当时AD保持停止；当前第三项为LCFN，详见上方最新记录。AE保留din_recompute=off，并启用mixer_ffn_kernels=compiled，启动记录compile_scope=event_mlp_and_mixer_ffn；AC为event_mlp_only。两项已记录resumed=true，恢复后AC至少更新到step860、AE至少到step705，loss和梯度有限；AE编译路径已完成前后向更新，尚无整轮速度或完整val证据。当前GPU另有其他用户计算进程，吞吐受到共享负载影响。TensorBoard重新启动，端口6006的HTTP和scalars tags均已验证可用，NO_PROXY绕过本地gRPC代理，跟随output/current（AC）。单模型完整val≥0.4和完整训练≤10分钟/epoch均未达到。

- **11-07 DIN分组索引合并传输已开始正式验证**：`attend_grouped`先在CPU构造全部分组/候选/还原顺序索引，再合成一个pinned buffer异步传输；保留原分组、完整有效历史、候选顺序与网络公式。AC从step900恢复，PID2721852、tmux avito-din-query，日志/tmp/avito-din-query-packed.log；至少更新至920，loss/梯度有限。新路径step910–920吞吐4.19–4.28万候选/秒；此前同机共享负载下稳态中位约3.26万。相同step910旧3.2667万、新4.2836万，但恢复和共享负载会影响结果，且loss有微小浮点差异，不宣称逐参数一致或整轮达标。主任务排序损失增强版（AE）已从step800同配方恢复接入，PID2729386/tmux avito-rank10，日志/tmp/avito-rank10-packed.log；FFN编译及DIN不重算保留，step810短窗约4.58万候选/秒，loss/梯度有限。DIN查询特征增强版（AC）已至step980，约4.26万候选/秒。两项均继续正式训练，AD保持停止；仍无新的完整val/test结果。Black120/语法检查通过，无tests或额外benchmark。

- **实验对用户报告使用中文含义**：AC=DIN查询特征增强版（候选item128查询扩为user128/item128/category64/QR32，共352→32）；AE=主任务排序损失增强版（target组内softmax loss权重0.05→0.1）。AC/AE仅为目录索引，不是架构名称，不能只报字母。两项均以AB为基线、独立单模型，不融合。

- **P相对数值特征已完整完成**：`output/tokenmixer/20261104_012738_091573/`，best epoch2 NDCG **0.3681849863567782**，比G最佳高0.000283558295；两轮完整train1356.669/1346.541秒、val11.288/11.448秒。完整test17.581秒、28595feeds/2904806候选、submission_ready=true，未上传；PID2793887已退出。G配方+relative_numeric，主softmax0.05、aux/pair各0.1，GPU1+2/global batch1024，从0/2epochs，fast_execution、checkpoint_mixer=False。报告和逐轮val文件均已核对。

- **只允许迭代单模型，目标为单模型主输出在完整官方 val 上 NDCG ≥ 0.4。** 禁止后续模型融合、预测后的相似度分数叠加或其他预测融合。已有 `output/ensemble/` 仅作历史记录，不纳入当前目标成绩。并行独立单模型对照已获授权，旧“不追加实验/不并行调参”限制已覆盖。
- 新配方从0开始，默认2epochs、完整官方train/val，训后自动用best.pt完成完整test预测并校验；不上传线上。不要写或跑tests，也不跑额外benchmark。普通中断恢复保留同配方模型、优化器、LR、RNG、数据游标；更换优化器必须新实验。
- 截至2026-11-06 22:40，W/X/Y/Z/AA/AB均完整完成两轮及test；最高完整单模型为AB epoch2 **0.3773363280029708**，@10=0.25313128661567913、@30=0.32163316090050714，目标0.4差0.0226636719970292。AB目录`output/tokenmixer/20261104_101342_330911/`，完整test ready且未上传。最新用户报告AB线上为0.3678834868（11-06反馈，助手未上传/未直接查榜）；首次原基线线上为0.349878157691628；下方旧“运行中/排队”文字仅属历史快照。

- **K已完整完成，近期16条特征未取得增益**：best epoch1 NDCG0.3640238972980395，epoch2为0.36345987424789405，低于F最佳0.001339879448；两轮train6023.195/5236.431秒（前期共享GPU及恢复/混合执行），val20.409/11.474秒。完整test17.305秒、28595feeds/2904806候选、ready且未上传。每轮完整train3151208/246745836、val14423/1389971；PID2557712已退出。


- R/S均已完整完成，原始配方记录：R `20261104_021345_298073`，GPU1/batch1024，J主softmax升0.2、relative_numeric=False/context0，参数13656166；S `20261104_021345_264773`，GPU2/batch1024，P+候选attention1层、relative_numeric=True/softmax0.05，参数14215014。均seed42/2epochs/BF16/8workers、fast_execution、存Mixer激活、event_mlp_only/expandable_segments；不重启。

- **M已完整完成**：best epoch2 NDCG0.3604908572483788、@10=0.23040937829582428、@30=0.2989316934815341、AUC0.6927739631608072、AP0.03869858837609868、Logloss0.07860116081152654、Brier0.015475589023606717；低于G最佳0.007410570814。两轮train1351.208/1342.108秒；完整test17.035秒、28595feeds/2904806候选、ready，未上传。PID2735538已退出，P已接替双卡；该主pairwise替换暂不采用。

- **训练入口内存设置已修正，R/S实际加载**：src.train.train在CUDA初始化前setdefault expandable_segments:True，并记入config/start/report；尊重显式环境覆盖。R/S显式env及config已核对；P未加载，reserved曾43.36GiB，未重启自动回收到15.94GiB，无观察到OOM，不能归因于新代码。Black120/语法检查通过，未跑tests/benchmark。

- **候选attention已通过S两轮完整train/FP32 val和test**：默认candidate_context_depth0，仅TokenMixer；原flatten2048投影128，经同feed双向self-attention/FFN，拼接成2176进三塔。无位置编码/target/pos输入、无跨feed或预测后加分。4heads×32，BF16 CUDA varlen训练；FP32按≤128feeds分组SDPA恢复候选顺序，完整14423feeds/1389971候选验证成功。depth1新增558848参数；不声称BF16与FP32逐值一致。J/K/P/N/R保持depth0。

- **训练位置统计已核对，未启用纠偏loss**：完整train pos/target/f1扫描覆盖3151208feeds/246745836候选/5446872正例；pos0正例率11.8145645%，pos200–300为1.0084688%；前10位置占候选10.8463488%、正例30.0251043%。观察率混合相关性、位置和筛选，不能直接当propensity；未使用val/test标签拟合权重。官方val/test展示前随机且不公开pos，不能用文件顺序替代。参考作者论文及分层表见doc/experiments.md；当前未实现/排队IPS或位置加权loss，J/K/P/N/R/S不变。

- **验证/推理传输已精简**：to_device新增features_only，eval/test将target/pos及pairwise监督留在CPU，消除无用H2D和val标签D2H回传；模型前向不读这些字段，主塔FP32/候选顺序/全部覆盖不变。Black120/语法检查通过，自N起新进程加载；不能把更早运行的计时归因于此改动。

- **T已完整完成**：`output/tokenmixer/20261104_032820_518635/`，R配方+融合BF16 Mixer LayerNorm，best2 NDCG0.3714766651879457，较R+0.000242927434；两轮train1898.550/1886.142秒（固定后31.6/31.4分钟），较R第二轮2205.765秒减少14.49%耗时。峰值22.46GiB，完整test16.601秒、ready且未上传。首次native混合dtype失败发生step0前，已验证checkpoint0无更新；改用现有Triton层归一化并从零进度恢复，FP32参数/归约、BF16输出，eval原生FP32。不是逐值等价证明；后续新实验采用该加速开关，未达到10分钟目标。

- **J/G/P第二轮配对诊断已记录**：J−G完整val NDCG均差0.002050807227，按用户簇bootstrap2000次95%区间[0.000978316227,0.003072964695]；J−P为0.001767248932，区间[0.000100998206,0.003481431133]。J比G在4个北京时间日期及正例数/候选数组均正向；不覆盖训练种子及同val选模偏差，不视为线上或因果收益。继续观察R的主rank0.2。

- **U已完整完成**：`output/tokenmixer/20261104_033008_869480/`，S+归一化DIN上下文128，底座1517补3至16×95，参数14231398。best2 NDCG0.37498315372126234，较S+0.000439769080；两轮train2301.806/2298.335秒，完整test17.612秒、ready，未上传。原始未归一化DIN保留，共享logits/values，并非模型融合。

- **并行调度采用每卡一组**：K与J共享GPU0时02:10–02:19吞吐中位数41574候选/秒，J退出后的13窗90274/秒，约2.17倍（正式不同批次窗口，不是额外benchmark）。前两批均已成功结束，当前W/X/Y分别使用GPU0/1/2；不再同卡堆叠新实验。

- **并行启动的TB指针写入已修正**：point_current_run临时symlink由固定.current.tmp改为含PID的独立文件，再原子replace，避免两个新进程互删临时指针。run.json/checkpoint原子落盘保留；这是代码审查发现的并发风险，不是已证实的历史FX冲突原因。Black120/语法检查通过，已在后续训练使用，不为此重启旧运行。



- **V已完整完成，当前单模最佳**：`output/tokenmixer/20261104_034531_022749/`，S的user/item置零降至0.02，relative_numeric=True/context1、主softmax0.05、无softmax DIN及BF16 norm，参数14215014。best2 NDCG0.3757933909872471，较S+0.001250006346；两轮train2278.454/2269.979秒，完整test18.004秒、ready，未上传。

- **W/X/Y于09:06从0实际并行开训**：均来源V，seed42/2epochs/batch1024/BF16/8workers，fast_execution、存Mixer激活、融合BF16 norm（FP32 eval/test）、event_mlp_only、RMSprop embedding/AdamW其余参数、完整val和自动test。W GPU0/PID70030/tmux avito-normalized-low，新增softmax DIN，参数14231398；X GPU1/PID70031/tmux avito-context-rank20，主softmax0.05→0.2，参数14215014；Y GPU2/PID70032/tmux avito-context-depth2，候选attention1→2层，参数14413286。TokenMixer仍16tokens/128维/4层，user/item置零0.02。首轮完整val已完成，第二轮进行中；完整test尚未生成。没有继承权重或融合，未运行tests/benchmark。

- **Z/AA/AB已完整完成**：均实际来源Y配方，从0/seed42/2epochs、din_history_bucket64；Z batch512最佳0.37558713188752574，AA主LR0.0005最佳0.37436038283164425，AB网络dropout0.05最佳0.3773363280029708。完整val/test及report均已核对；旧等待器和训练进程已退出，不恢复旧任务。下一轮采用AB配方。

- **DIN历史padding优化已完成正式验证，但未取得明确整轮加速**：din_history_bucket64已用于Z/AA/AB全部两轮及FP32 val/test，默认0兼容旧检查点；只移除mask=False后缀，保留有效历史和候选顺序。静态val位置数减少约20%，但AA/AB第二轮1961.270/1959.523秒，对Y的1946.322秒没有明显提速；同时存在超参数变化，不能单独归因。不得将静态计算量减少称作实测20%提速。

- **扰动范围按现有代码更正**：UnifiedFeatures中user_embedding_dropout作用官方user向量，item_embedding_dropout同时作用候选/历史的官方item向量和QR ID向量；四次item调用分别抽独立mask，并非绑定置零。N/V的0.1→0.02因此也改变QR输出扰动，不能只归因于官方向量。时间/类别等表不使用此整向量置零；eval/test全部关闭。此次只改说明，未改代码、RNG或正在训练的配方。

- **W/X/Y首轮配对诊断完成**：完整val逐候选键/target对齐；相对V首轮NDCG均差−0.0009054/−0.0007752/+0.0004800，用户簇bootstrap2000次95%区间均跨0。Y短历史≤64组+0.0042286，但四个北京时间日期仅两天正向，最后1047feeds日期−0.0041288；不能据首轮继续加深或宣称稳定收益。完整分层表见doc/experiments.md；当前三组继续第二轮，下一批仍为batch/LR/dropout。

- **商品频率只读诊断完成，未加入模型**：全train3151208feeds/246745836候选/583586378历史位置，仅读item_id与history__item_id；不同候选ID38950205、历史ID94705754，最大ID111319563/111319574，负ID和ID0计数均0。train候选/历史次数对val候选的非零覆盖65.4337%/58.7427%，单特征NDCG0.297543307293/0.302735242221、AUC0.52744994/0.54408991。不是条件增益或模型结果；未保存统计缓存/改预测。若后续实现，train每行须只用过去记录，不能回填全train最终次数。

- **T/U/V完整val配对不确定性已记录**：按用户聚类bootstrap2000次，T−R NDCG均差0.000242927434的95%区间[-0.000161465540,0.000642988926]；U−S为0.000439769080，区间[-0.001114419270,0.001918948121]；V−S为0.001250006346，区间[-0.000096750830,0.002505333352]。三者区间均跨0，不宣称独立泛化或严格非劣性；V−S四个北京时间日期均正向。当前仍按完整val选择单模，W/X/Y验证组合/调参，不能把小幅点估计当确定收益。

- **W/X/Y首轮完整val已核对**：W NDCG0.37118764250958386、X0.3713178220095837、Y0.37257308837529873；相对V首轮0.37209305814398025，分别−0.000905415634/−0.000775236134/+0.000480030231。三组每轮train3151208feeds/246745836候选、val14423/1389971，epoch1逐候选文件及全部指标成功落盘；均继续第2轮。当前最高完整pipeline仍V0.3757933909872471，0.4未达成，不能把较低首轮当最终结果。

- **最新线上反馈已记录**：用户11-06报告当前最佳AB提交线上0.3678834868，对应本地完整val0.3773363280029708；比首次线上0.349878157691628高0.018005329108372。非助手上传/查榜证据，目标仍为单模型完整官方val≥0.4，尚未达到。

- **AC/AD/AE于11-06 22:45启动、22:47记录step0**：来源已完成最佳AB配方，从0/seed42/2epochs/batch1024/BF16/8workers、dropout0.05、user/item扰动0.02、历史分桶64，RMSprop embedding/AdamW其余，完整val及自动test。AC GPU0/PID557864/tmux avito-din-query，目录20261106_224512_765619，DIN query复用user128/item128/category64/QR32共352→32，新增7168参数，总14420454；AD GPU1/PID557865/tmux avito-context3，目录20261106_224512_774671，候选attention3层，14611558参数；AE GPU2/PID601305（原557867已退出）/tmux avito-rank10，目录20261106_224512_781608，主rank0.1，14413286参数。冷启动依赖/数据等待尚未取得完整val，不当作已完成；预读三split向量表至共享页缓存，无复制。TB服务重新启动，HTTP可用性仍待确认。

- **11-06最新速度要求：完整训练≤10分钟/epoch**，与单模型完整val NDCG≥0.4一起持续优化，当前未达成。用户明确要求继续提速，允许工程执行调整后同配方检查点恢复；保留完整历史/样本、优化器和LR/RNG/数据游标，不将局部吞吐估算当整轮达标。

- **AE工程优化恢复已实际更新**：旧PID557867已SIGINT正常退出，最近完整last.pt为global_step500/epoch0/batch_cursor500，RMSprop/AdamW均保留。新增din_recompute on/off，默认on兼容旧模型；off保存DIN激活而不在反向重算，内部无随机算子，参数/损失/分组不变。AE同目录/tmux avito-rank10以--resume last.pt --din-recompute off恢复，启动日志/tmp/avito-rank10-recovery.log；新PID601305已从step500恢复并更新超过step545，loss/梯度有限；step510–545相同批次窗口约12.7万→13.4万候选/秒（约+6%），峰值约26GiB，仍未达到10分钟。恢复后数值有小幅差异，不声称逐参数/随机轨迹完全一致；AC/AD不重启。源码Black120/语法检查通过，没有tests/额外benchmark。该轮包含执行切换，不能算纯优化完整epoch；第二轮才能给纯路径整轮时间。

- **AC/AD/AE已实际更新、启动核对完成**：三个last.pt第1步均history空、AdamW状态step1（AC/AE137组、AD149组）、RMSprop17表全部step1；已记录的loss/梯度均有限。每卡实际训练显存约23–24GiB，三split向量表顺序预读已完成，22:51最近20窗数据等待占比约0.1%、中位吞吐AC/AD/AE约12.19/12.48/12.66万候选每秒；不是完整epoch计时。TB 6006已返回HTTP200，跟随AC。

- **每卡实际一个Avito训练实例**：GPU0额外列出GPU1/2训练PID各262MiB，为默认设备CUDA上下文，非额外模型。device_from_name已改为在autocast/backend初始化前set_device；Black120/语法检查通过，新进程加载，当前进程不重启，未声称已实测消除。其他用户unisearch进程独立存在，不改动。

当前实验索引（目录均位于 `output/tokenmixer/`；运行状态后续须读活进程与run.json，不能把快照当当前事实）：

| 实验 | 目录 | 配方变化 | 状态 | 最佳epoch / 完整val NDCG | 完整test |
| --- | --- | --- | --- | --- | --- |
| A | `20261103_092847_192231` | 原DIN仅embedding LR降至0.0003 | complete | 2 / 0.352813219216237 | ready |
| B | `20261103_093730_240215` | 原DIN + 主target-rank0.05 | complete | 2 / 0.358064582526040 | ready |
| C | `20261103_102843_883489` | 原DIN + raw f1–f5 | complete | 2 / 0.353173190763577 | ready |
| D | `20261103_113958_737085` | B + embedding字段缩放 | complete | 2 / 0.360085316719806 | ready |
| E | `20261103_114414_395532` | B + raw数值 + 连续相似度输入 | complete | 1 / 0.361193981613878 | ready |
| F | `20261103_132230_880737` | E + embedding字段缩放 | complete | 1 / 0.365363776745913 | ready |
| G | `20261103_141544_349710` | F改为单卡batch1024 | complete | 2 / 0.367901428062111 | ready |
| H | `20261103_145647_324726` | F主target-rank改为0.1 | complete | 1 / 0.366637167862204 | ready |
| I | `20261103_225136_504077` | F + 32维历史位置embedding | complete | 1 / 0.363787391035576 | ready |
| J | `20261103_230946_686897` | G主target-rank改为0.1 | complete | 2 / 0.369952235288615 | ready |
| K | `20261103_231929_460988` | F + recent_history_window16；GPU0顺序副本 | complete | 1 / 0.364023897298040 | ready |
| L | `20261103_234900_877500` | H的ListMLE/pairwise权重均降至0.02；fast执行 | complete | 2 / 0.366340905922189 | ready |
| M | `20261104_004142_396038` | G主rank改为ΔNDCG抽样pairwise；双卡小batch，存Mixer激活 | complete | 2 / 0.360490857248379 | ready |
| N | `20261104_023053_452577` | J的user/item向量置零降至0.02；独占GPU0存Mixer激活 | complete | 2 / 0.373588631337624 | ready |
| P | `20261104_012738_091573` | G + f1–f5组内zscore；双卡小batch，存Mixer激活 | complete | 2 / 0.368184986356778 | ready |
| R | `20261104_021345_298073` | J主softmax权重0.2；单卡存Mixer激活 | complete | 2 / 0.371233737754210 | ready |
| S | `20261104_021345_264773` | P + 1层同feed候选attention；单卡 | complete | 2 / 0.374543384640801 | ready |
| T | `20261104_032820_518635` | R + 融合BF16 Mixer norm；修复后从step0恢复 | complete | 2 / 0.371476665187946 | ready |
| U | `20261104_033008_869480` | S + 归一化DIN上下文128 | complete | 2 / 0.374983153721262 | ready |
| V | `20261104_034531_022749` | S的user/item向量置零降至0.02 | complete | 2 / 0.375793390987247 | ready |
| W | `20261104_090646_618271` | V+融合BF16 norm+归一化DIN | complete | 2 / 0.375938009452569 | ready |
| X | `20261104_090646_624477` | V+融合BF16 norm+主rank0.2 | complete | 2 / 0.375300319208548 | ready |
| Y | `20261104_090646_631217` | V+融合BF16 norm+候选attention2层 | complete | 2 / 0.377029051958765 | ready |
| Z | `20261104_101342_315799` | Y + batch512 + DIN历史分桶64 | complete | 2 / 0.375587131887526 | ready |
| AA | `20261104_101342_323360` | Y + 主LR0.0005 + DIN历史分桶64 | complete | 2 / 0.374360382831644 | ready |
| AB | `20261104_101342_330911` | Y + dropout0.05 + DIN历史分桶64 | complete | 2 / 0.377336328002971 | ready |
| AC | `20261106_224512_765619` | AB + DIN query352→32 | resumed | 尚无完整val | 未完成 |
| AD | `20261106_224512_774671` | AB + 候选attention3层 | interrupted | 尚无完整val | 未完成 |
| AE | `20261106_224512_781608` | AB + 主rank权重0.1 | resumed | 尚无完整val | 未完成 |

- **G已完整完成**：PID2261974已退出，原GPU0/batch1024；两轮完整train/val，epoch2最佳。test覆盖28,595feeds/2,904,806候选，18.520秒，submission_ready=true，未融合、未上传。**I已完整完成**：GPU1+2/batch4096，best epoch1 NDCG0.3637873910355759，epoch2为0.3632478512996115，未超过F。两轮训练1677.760/1628.416秒（第2轮为旧/新执行混合，不是纯加速轮次）；完整test fast_execution/FP32/batch1024，17.558秒，ready且未上传。PID2536388已退出，不重启；后续不追加位置embedding。
- **J已完整完成**：`output/tokenmixer/20261103_230946_686897/`，best epoch2 NDCG0.369952235288615、@10=0.24206358823354002、@30=0.3113226298291903、AUC0.7070750234851496、AP0.04103261233133102、Logloss0.07706900502272356、Brier0.01541491472949769。两轮train5425.795/5911.166秒，GPU0与K共享且首轮含恢复/执行切换；不可当独占卡加速计时。完整val每轮14423feeds/1389971候选；best epoch2完整test35.285秒，28595/2904806、ready，未上传。PID2514980已退出。
- K `output/tokenmixer/20261103_231929_460988/` 已完整结束，不重启。N已接替GPU0，R/S分别占用GPU1/2，每卡一个训练任务。
- **N已完整完成**：`output/tokenmixer/20261104_023053_452577/`，J的user/item官方向量置零0.1→0.02，其余学习配方保留；从0/2epochs/GPU0/batch1024，fast_execution、存Mixer激活。best epoch2 NDCG0.3735886313376244，完整test17.262秒、ready且未上传，PID3101276已退出；V已接替GPU0。
- **完整缺失统计已核对**：train全零user向量79,848/3,151,208=2.533885%；candidate缺失索引7,924/246,745,836=0.003211%；history缺失索引90,521,519/583,586,378=15.511246%。val分别1.421341%/0.015396%/14.205254%。商品统计仅索引-1，不把它当全部有效索引零向量统计；未读target/pos。这只支持轻扰动对照，不证明它必然有益，不新增缺失标记或有效向量比例。
- **L已完整完成**：best epoch2 NDCG0.3663409059221892、@10=0.2380610950038841、@30=0.3066422959709585、AUC0.6995305697034553、AP0.03799371573549275、Logloss0.07730996516551626、Brier0.015428005370846194；未超过H最佳。两轮train1556.413/1543.347秒。完整test17.118秒、28,595feeds/2,904,806候选、ready，未上传。PID2537949已退出，M已接替双卡。
- **H已完整完成**：best epoch1为0.36663716786220396，epoch2为0.36640413949847367，较F最佳提高0.001273391116；完整test 28,595feeds/2,904,806候选，20.099秒，submission_ready=true。PID2261979已退出，不重启。
- **特征与维度**：原DIN底座1376=16×86；加raw f1–f5为1381，补11维到16×87；再加连续UI/history-max/history-max组内zscore为1384，补8维。I的history_position_dim32使事件输入288→320，事件MLP仍输出256后投影128，底座维度不变；位置1代表最新、256最旧、0为padding，不是时间间隔。recent_history_window16仅在网络输入追加近期max和组内zscore，1384→1386、补6维，token切分和参数量不变；不叠加预测分数。
- **配方**：F/G/I主target-rank0.05，H为0.1；其余保留BCE、0.1 ListMLE(pos)、0.1独立pairwise(pos差≥4随机64对/feed)。nn.Embedding用RMSprop，其余AdamW，峰值LR0.001，5%warmup/cosine至5%峰值；网络dropout及user/item置零各0.1。候选与历史共享类别及QR表。保留原历史mean、history×item、user×history和DIN。
- **执行加速已部署**：I/J/K启用fast_execution（DIN按feed分组bmm、批量token投影），同GPU顺序副本直接复制/合并；验证/test只算主塔，保持FP32。G同best.pt完整val NDCG/@10/@30完全一致，val/test最大score差均1.90735e-6，完整test位于`output/tokenmixer/20261103_233555_080590/`且ready，未上传。舍入和BF16后续轨迹可能变化，不称逐参数一致。I稳态短窗口约15.9万候选/秒，对照切换前约15万；未取得整轮加速计时。
- **K验证显存修复已部署**：23:42保存step150后同配方恢复，PID2557712，已加载val batch上限1024。`avito-val-cap-rollout`已正常退出，无重复训练进程。不是新调参任务，I/J未为此重启。
- **23:36显存事件**：额外完整test与I同GPU，推理reserved34.73GiB导致I第一次恢复OOM；推理结束后从原step1200恢复，当前PID2536388，队列已重新绑定。额外推理以后安排空闲GPU，独立infer CLI默认expandable_segments。新训练入口TokenMixer val上限1024，自动test显式继承fast_execution；当前进程是否加载这些后续修复须以启动时点为准，不为未加载的改动宣称已生效。
- **工程限制**：DIN/Mixer原生执行；按配置重计算（M仅DIN重计算），仅event_mlp独立compile；FX编译入口共用RLock。所有跨GPU参数和梯度必须经CPU，禁止已知数值错误的P2P。新模型不重用优化器状态，运行中不为调参重启。TB固定tmux `avito-tensorboard`、端口6006，跟随output/current；不重新生成架构图。
- **22:41中断恢复记录**：G/H旧进程和tmux消失、run.json为零字节，原因未确认。last.pt完整可读，分别从step4850/1450同配方恢复。最初16个worker在机械盘随机读向量表时等待；利用可用内存顺序预读train/val/test官方memmap进入共享页缓存（不复制、不改数据）后恢复GPU计算，近期G约8.1万/H约15万候选每秒。TB已验证HTTP200。恢复计时包含新的数据等待，不包含进程退出期间的停机时间。
- **完整val诊断**：最近16历史max直接排序NDCG0.3498913141223604，全部max0.3422826254668367，最近64为0.34534263300103635。这是单特征诊断，不是训练模型成绩；同val探索选择不能当独立泛化收益。各event_type子集未显示一致更强信号，不猜测编号语义。G/H与F逐feed分析见doc/experiments.md；G在正例≥3组仍负向，不声称小batch全面更优。
- **23:53只读特征审查**：完整val历史类别频率/近期频率/最近同类位置单特征NDCG约0.2473/0.2462/0.2533；G的同类别数量分组softmax质量与target归一质量总体接近，暂不追加类别计数。非零官方向量L2范数中位数约256，1%–99%约255.35–256.69；原始UI点积比cosine仅+0.000243 NDCG，candidate范数AUC0.50083，暂不追加幅值特征。属于同val诊断，不证明条件增益为零，不修改预测分数。
- **已审查但不加入的特征**：非零训练用户82.84%仅一条feed，非零val/test feed的train用户覆盖31.86%/30.38%；本轮未加user_id embedding。val candidate/history exact item匹配为0。数据无历史事件时间戳，不编造距今时间；不加入用户已拒绝的缺失标记和有效历史向量比例。
- **完整预测与报告**：新训练每轮原子保存validation_epoch{epoch}.parquet，键/target/score逐候选对齐，完整覆盖及全部指标成功才落盘；每轮和test结束刷新同目录report.md。A加载旧代码无逐轮文件；B的best完整val预测已补齐。原TokenMixer/DIN/OneTrans完整val预测也已保存。单卡顺序副本与双卡CUDA随机流不同，不称逐参数同随机对照。全部loss、效果和工程指标以各run报告为准。

以下DIN基线及OneTrans记录均为历史，不代表当前活跃训练：

- **最新运行已于北京时间21:09完整完成**：`output/tokenmixer/20261101_200042_126547/`，status=complete，best epoch2；完整val NDCG0.35271545665943、@10=0.220486297791773、@30=0.290227949576804、AUC0.6871645139781767、AP0.03747561100222456、Logloss0.07757993938218416、Brier0.015424976259532687。比原TokenMixer最佳0.356918365623154低0.004202908963724。两轮训练1761.715/2273.081秒（29.36/37.88分钟），各覆盖3,151,208feeds/246,745,836候选；val均14,423feeds/1,389,971候选。test使用epoch2 best.pt、eager FP32，22.121秒，submission_ready=true，覆盖28,595feeds/2,904,806候选；submission.parquet/report.md均保留，未上传线上。训练tmux已退出。先前提出的DIN扩大分块、部分层取消重计算、按feed分组矩阵乘法仅完成代码审查，本轮没有启用，不能宣称已获得相应加速。

- **用户最新纠正：更换优化器要从0开始新实验。** 已删除续训时AdamW→RMSprop状态迁移代码，CLI拒绝在--resume时更换embedding优化器。更换优化器、需要干净对照时随机初始化模型、优化器与LR进度；不能把混合优化器历史当成全程新优化器实验。普通中断恢复仍保留同配方模型/优化器/RNG/数据游标。
- **本轮已完成的训练配方** `output/tokenmixer/20261101_200042_126547/`，原tmux `avito-tokenmixer`已退出。新目录、seed42随机初始化、global_step0，无--resume，不继承旧模型/优化器/历史；已完成至少3步更新，首次checkpoint里111组AdamW和17组RMSprop状态的step均为1；2epochs、batch4096、8workers、GPU1+2、BF16。TokenMixer 16tokens / width128 / depth4 / GELU，block级x+Block(x)残差；事件MLP288→256→256后投影128，历史mean及history×item、user×history保留，新增DIN128。底座1376=16×86，无补零，各块86→128，输出16×128，flatten2048进入三塔。DIN32维query/key，[q,k,q-k,q*k]局部激活128→32→16→1、PReLU、无softmax加权和；候选2048分块并重计算激活，同feed历史编码共享。
- **从第1次更新起**17张nn.Embedding表使用RMSprop（峰值LR0.001、alpha0.99、eps1e-8、momentum0、weight_decay0），其余参数AdamW（峰值LR0.001、weight_decay0.0001）；共享5%warmup+cosine至5%峰值，梯度全局裁剪后分别更新一次，两个优化器状态均写checkpoint。网络dropout0.1、user/item置零各0.1。BCE+0.1ListMLE+0.1独立pairwise，每feed随机64对且pos差≥4。完整val与全部工程指标写report，训后自动完整test推理；不跑tests或额外benchmark，不上传线上。
- **当前编译范围仅event_mlp_only**：Mixer和DIN使用原生激活重计算，避免PyTorch2.5已实际出现的FX并发冲突、checkpoint元数据不一致和AOT partition断言。事件MLP独立编译，所有编译入口共享RLock；CUDA执行仍异步。所有跨卡参数/梯度经CPU中转，禁止直接P2P。时间/分桶/QR/UI相似度等增强特征全部保留，数据没有历史事件时间戳。固定TB跟随output/current，6006端口；不重新生成架构图。新一轮速度/分数只读新运行日志。
- **旧混合优化器运行已停止** `output/tokenmixer/20261101_185042_092119/`，status=interrupted，保留原checkpoint及报告。前550步全AdamW，之后embedding RMSprop；已完成epoch1完整train/val：NDCG0.3523827615、@10=0.2202425069、@30=0.2900349201、AUC0.6817130085、AP0.0360029295、Logloss0.0778502790、Brier0.0154384994；训练1642.366秒（含此前启动/恢复开销，不含进程退出停机等待）。未完成两轮和test，不能当成新一轮全程RMSprop结果。下方旧“当前运行/预计完成”均为历史快照。

以下为此前OneTrans及基线历史记录：

- **最新运行结果已完成**：`output/onetrans/20261101_174526_905299/` 状态complete，2epochs、完整train/val及test均完成。best epoch2：NDCG0.3539223051、@10=0.2224824294、@30=0.2921969792、AUC0.6922009709、AP0.0402078219、Logloss0.0772995055、Brier0.0154053932。两轮训练970.039/905.226秒；每轮3,151,208 feeds/246,745,836候选。
- 本轮是去pairwise+LR0.0005/dropout0.05组合调优；比上一轮NDCG0.3565077758低0.0025854707，不能单独归因于pairwise。自动test已完成并校验2,904,806候选/28,595 feeds，61.122秒，`submission.parquet`与`report.md`在该运行目录；未上传线上。训练tmux已正常退出，不需要自动重启或追加新实验。下方“训练中/当前运行”属于先前快照。

- **最新授权已覆盖此前固定旧运行的要求**：用户明确沿用当前OneTrans，关闭pairwise，并同时调学习率/dropout，启动一轮新训练。当前运行 `output/onetrans/20261101_174526_905299/`，tmux `avito-onetrans`；2epochs、batch4096、8workers、GPU1+2、BF16、4query/width256/depth3。pairwise_weight=0、pairwise_tower=False，随机pair抽样关闭；保留BCE+0.1 ListMLE。峰值LR=0.0005、网络dropout=0.05、user/item置零各0.1、weight_decay=0.0001；5%warmup+cosine至0.000025，seed42。记录为“去pairwise+LR/dropout组合调优”，不能把分数变化单独归因于pairwise。参数18,328,802。
- 前一轮 `output/onetrans/20261101_170112_769404/` **已完成2epochs和完整test推理**：best epoch2，NDCG0.3565077758、@10=0.2254957713、@30=0.2954782801、AUC0.6976179397、AP0.0414076970、Logloss0.0770575592、Brier0.0153953646。两轮训练1130.706/1062.400秒。submission.parquet覆盖28,595 feeds/2,904,806候选，submission_ready=true，推理61.191秒；未上传线上。下方旧“当前运行/三塔”描述只属于上一轮历史记录。
- 新训练默认pairwise权重已改为0；保留旧pairwise代码和旧checkpoint配置，保证旧模型可加载。新训练仍自动记录全部val/report并生成完整test提交文件；不跑tests/benchmark，不同时开启额外调参任务。

- 用户明确：不再写或跑tests测试代码，不再额外运行benchmark；直接正式train-val。tests与test集预测不同，**训后完整test推理保留**。
- 用户已同意非序列query从8减到**4个**，优先速度；随后接受当前约**23–24万候选/秒、17–18分钟/epoch**的运行速度。**保持当前配置跑完两轮，不继续改训练配置或为10分钟目标重启。**10分钟目标没有达成，不能宣称达标。
- 当前正式运行 `output/onetrans/20261101_170112_769404/`，tmux `avito-onetrans`。GPU1+2，全局batch4096（每卡2048），8workers，BF16，局部编译，2epochs，AdamW峰值LR0.001，5%warmup+cosine。每卡实测峰值约38GiB。阶段、步数和分数只读当前run.json/run.log。
- OneTrans为unified_prefix：width256、3层、4heads、4query tokens。非序列底座864→4×216→4×256；历史每位置288→256→256；query后插入可学习SEP，统一右向causal attention。最终query TokenMixer后flatten1024，三个独立tower。参数18,624,995。
- category64、type/source32、f1–f5各32维分桶embedding；32维QR item ID候选/历史共享。小时/分钟/星期/工作日/一周小时各32维；UTC，工作日仅周一至周五。历史长度32维分桶、UI cosine128桶32维。无向量缺失标记或有效历史向量比例；无历史事件时间戳，不编造距今时间。
- Query与最终TokenMixer的PerTokenFFN为SwiGLU（hidden704），up/gate合并投影；QKV亦合并，保持原Parameter与checkpoint键。序列共享FFN和事件MLP保持GELU。融合RMSNorm的Triton启动/自动调参用进程内锁保护，GPU异步执行，保留BF16输出；FP32验证/推理保持原生。
- BF16历史侧把全部有效事件和SEP按feed打包成旧→新→SEP，varlen causal attention和共享FFN不计算padding。每层K/V取该层输入，flat_key/flat_value仅每feed保存一份；key/value在打包缓存内可为None。FP32验证/推理保留padded参考实现。全部历史事件保留。
- Query self按8192候选分块，避免原先大批query反向出现非有限梯度；与共享历史仍使用共同softmax归一化，不能改成两个独立attention相加。
- 三独立tower：主BCE；ListMLE(pos)权重0.1；pairwise权重0.1，同feed均匀随机64对、pos差≥4。val/test无pos不提供顺序监督。仅主tower用于最终指标和test预测。
- 本机CUDA peer copy数值错误，所有跨卡参数和梯度传输必须经CPU。旧版128 cross已完整完成，epoch2 NDCG0.3567997440、@30=0.2959534149、训练598.76秒，test已生成；不能把该成绩当新版4-query结果。
- 旧8-query运行 `output/onetrans/20261101_154436/` 未完成完整val。CUDA Graph尝试因当前分块训练/运行时内存追踪问题退出，**已从入口删除该不稳定路径**；当前只用正常局部torch.compile，切勿宣称CUDA Graph已启用。
- 数据恢复在worker端跳过已读行，保持shuffle RNG及worker轮转；checkpoint保存逐worker行游标。全局batch调整时也保留样本覆盖，LR按reference_batch_size逻辑进度延续。完整epoch结束严格核对3,151,208 feeds。
- 每轮及推理结束更新同运行目录report.md，包含全部loss、效果和工程指标。固定TB仍跟随output/current，端口6006。不得重新生成已删除的架构图。

## 项目与授权

- 项目目录：`/home/zhengxiang/Avito`。
- 比赛：https://ods.ai/competitions/avitotechrecsys2026
- 官方数据说明：https://ods.ai/competitions/avitotechrecsys2026/dataset
- 中文说明：`doc/overview.md` 为比赛概览，`doc/data.md` 为文件结构和字段字典。整理来源与官方表述差异写在文档内。
- 本比赛为预测文件提交，不是平台执行代码；无需提交运行环境，也无需与官方基线锁定版本一致。优先沿用 Jigsaw，按实际使用的代码检查兼容性。
- 用户已授权实现 PyTorch 训推 pipeline 并启动 TokenMixer 正式 train-eval。full 重训/testinfer 入口已实现；官方 train/val 评估已完成，用户随后授权用最佳检查点推理完整 test 并生成本地提交文件。上传、线上提交和公开发布仍需明确授权。
- 使用本地 Git；用户未提供远端地址，暂不设置 `origin`。不主动使用 `gh`。
- 正式 train-eval 已完成 3 轮，最佳为第 2 轮，完整 val NDCG 0.3569183656，用户反馈首次线上分数 0.349878157691628，详见 `doc/experiments.md`；有限数据预检和组件检查不等于完整验证。

## 初始化完成状态（2026-11-01）

- 官方数据全部下载并解压，651 个文件通过 ZIP CRC/大小检查；640 个 Parquet 分片完成全量扫描，列表对齐、标签、跨 split feed ID、embedding 索引及模板键检查通过。临时下载/校验文件已清理。
- train：3,151,208 条 feed / 246,745,836 个候选行；val：14,423 / 1,389,971；test：28,595 / 2,904,806。
- train 正例候选 5,446,872，val 正例候选 21,983；test 标签不公开。数值特征 f1–f5 全量未发现 null/NaN/Inf。
- train 有 1 条 feed 含重复 item，共 4 个额外候选；val/test 候选键均唯一。提交模板与 test 的全部键一致。
- 保留官方 split。页面日期与原始 timestamp 的 UTC/UTC+8 边界不同，实测 UTC+8 时间见 `doc/data.md`，不要按页面日期零点重新裁剪。

## 目录与环境

- 官方数据直接放 `data/`：`train/`、`val/`、`test/`、`sample_submission.parquet`、官方 `readme.md`。不增加 `raw/` 或外层 `competition/`。
- 自有模型及 pipeline 源码位于 `src/`；官方基线当前位于根目录 `official/`。官方参考代码的方法改动须另行说明。
- 独立模型资源放 `model/` 并忽略 Git；训练检查点、eval、推理结果统一放 `output/<model>/<timestamp>/`，时间戳用 UTC+8。
- 项目上下文维护在本文件；仅在确需独立文档时使用 `doc/` 下 Markdown，不自动堆积说明、快照、日志或重复文件。
- 本项目使用 conda 环境 `Jigsaw`：`conda activate Jigsaw`。非交互命令可用 `conda run --no-capture-output -n Jigsaw ...`。
- 初始化时已安装 Python 3.11.13、NumPy 1.26.4、Polars 1.34.0、PyArrow 21.0.0、CatBoost 1.2.8、PyTorch 2.5.1（CUDA 12.4）。
- 初始化仅补装缺失的纯 Python 依赖 `loguru==0.7.3`，未升级现有包；两套官方基线 CLI 导入及评估器边界样例检查已通过。CUDA 可见不代表已验证正式训练。
- 自有 Notebook 的默认 inspect 已在真实数据上运行；8 条真实 train feed、833 个候选的特征与官方实现完全一致，适配后的 CatBoost Pool 加载通过。另有合成数据的缺失 embedding/int8 转换、指标一致性、提交格式及 Transformer CPU 前向检查；均未训练，不代表模型效果。
- 官方 `official/pyproject.toml`/`uv.lock` 是参考环境，固定 CatBoost 1.2.10、Torch 2.10.0、Polars 1.14.0 等，与现有 Jigsaw 不一致。不要直接 `uv sync` 或擅自升级共享环境；初始兼容性检查不构成精确环境复现。
- 如后续确定存在 Python/依赖冲突，直接创建项目独立 conda 环境并记录原因，不破坏 Jigsaw。
- Python 及 Notebook Python 单元必须 Black 格式化，行长 120；Notebook 清除无用输出。格式检查：`conda run -n Jigsaw python -m black --check --line-length 120 src tests official`。
- 自有训推一体 Notebook 为根目录 `avito.ipynb`，默认 `MODE="inspect"`，支持本地/Kaggle 路径、`eval`、`full` 和 `predict`，代码自包含，不依赖本机绝对路径或附带仓库模块。创建 Notebook 不代表授权启动训练。
- Notebook 沿用官方 CatBoost 特征和训练参数；为兼容 Jigsaw 的 CatBoost 1.2.8，在 `Pool` 边界将 Polars 表转成 Pandas、标签/分组转 NumPy。官方原脚本只验证过 CLI 导入，直接运行训练仍会因其 Polars 输入失败，不能将其描述为训推已兼容。
- `eval` 在 train 训练并用 val 的 NDCG@30 选择树数，同时报告完整 NDCG/@10/@30；`full` 要求指定一个完整 eval 运行目录，固定该树数在 train+val 重训；`predict` 要求显式指定模型路径并自动检查模板键和格式。正式运行前先做吞吐/内存预检。

## 数据与验证约束

- 每行 Parquet 是一条 feed，候选字段为等长列表，必须共同展开，保持 `(feed_id, item_id)` 对齐。
- 已实测 train 存在同一 feed 内重复 `item_id`，训练候选键不保证唯一。保留官方行和列表顺序，需要关联时保留候选下标，不擅自去重或套用提交键唯一约束。test/提交键唯一仍须严格检查。
- 训练时间：2026-09-06 至 09-19；val：09-22 至 09-24；test：09-26 至 09-29。使用官方 val 验证，不随机拆候选行。
- train 的 `pos` 是原始曝光位置；val/test 没有此字段，其候选在曝光前均匀随机打乱，发布文件又共同打乱字段。默认不得将 `pos` 作为跨 split 推理依赖。
- 每条 feed 至少 5 个候选，train/val 按官方构造含正例；历史最多 256 条事件，顺序由新到旧。
- `target` 是二值标签，不能当特征；`feed_id` 跨 split 不重叠，但用户和商品可重叠，不能声称 user/item-disjoint。
- 各 split 有独立 `item_embeddings/data.memmap` 和 `data.meta`；按对应 split 的索引取向量，不能跨 split 复用 embedding 索引。
- 用户/商品向量为 128 维 int8；计算相似度前转 float32。商品 embedding 索引 `-1` 表示缺失，禁止直接负索引。
- 主页面声明主指标为每 feed 等权平均的 NDCG；官方评估代码同时计算完整 NDCG、NDCG@10、NDCG@30。主页面公式混有 `@30` 字样，CatBoost 用 @30 选模型，后续报告应并列保留完整 NDCG 和 @30，不能擅自混称。
- 分数相同采用悲观排序：负例排在正例前；无正例 feed 的 NDCG 记为 1。
- `official/calculate_metrics.py` 的 CLI 需要带 `leaderboard_split` 的隐藏答案文件，不能原样用在公开 val；其 `metrics(frame)` 可处理展开后的 `feed_id,target,score` 数据。
- 提交 Parquet 仅含 `feed_id:int32`、`item_id:int32`、`score:float32`，与模板候选键完全一致、无重复/遗漏/额外键、无 null/NaN/Inf。
- Public/Private 按测试 feed 时间约 30%/70%；每天最多提交 5 次，最多 4 人组队。
- 主页面列出 2026-12-09 结束及当天 00:00–12:00 选择最终方案；FAQ 存在年份笔误，精确截止时区/时间后续提交前需复核，不直接推断 UTC+8。
- 官方规则 PDF 第 2.1 条列有俄罗斯公民资格要求；已提醒用户核实。不要代用户注册或确认身份。

## 训推入口

- 用户最新要求：后续每次完整训练结束自动预测完整 test 并校验生成 `submission.parquet`。train-eval 使用 `best.pt`，full 使用 `last.pt`；有限样本/步数诊断不自动生成正式提交。推理失败则整次 pipeline 标记失败，模型检查点保留。自动推理沿用 eager FP32，batch 至多 1024；本地生成已获持续授权，上传/线上提交仍需用户明确授权。
- 用户最新要求：后续所有新训练默认跑 2 个 epoch（train-eval/full 均适用，含 `--from-eval`）；除非用户另行指定，不复用旧运行的多轮设置。已有 checkpoint 恢复保留原计划，本次 test 推理使用已选好的 epoch 2 best.pt。

- 用户最新明确授权启动定制 OneTrans 正式训练，改为 **2 epochs + warmup/cosine**。CLI 默认模型 onetrans，默认 2 epochs；AdamW lr 0.001，前 5% 总步数线性 warmup，cosine 至 5% 峰值 LR。调度计划/global_step 随 checkpoint 保存，恢复逐参数一致性测试通过。
- OneTrans：非序列 `[dense8, category16, user128, item128, user×item128]` concat=408，8×51 独立投影为 8×128 query。历史位置 `[item128,cat16,type8,source8,missing1]` concat=161，共享 MLP161→128→128，每位置一个 token，保留完整 L≤256，反转有效部分为旧→新。3 层“序列 causal self-attn/共享 FFN；query cross 序列/token-specific FFN”，最后 query TokenMixer，flatten1024→双 tower。attention 4 heads×32，参数 6,377,778。属于按用户要求的 OneTrans 定制，不是完整论文复现。
- 同组候选按 feed 复用每层历史 K/V，各 `(B,4,L,32)`；BF16 使用已安装的 FlashAttention varlen，序列不按候选复制。显式 `prepare_history_cache` 支持推理分候选块复用。训练保留梯度，仅当前 forward/backward 复用，更新参数后重建。CPU/真实 CUDA BF16 缓存与分块语义检查通过。
- 当前正式 OneTrans：`output/onetrans/20261101_144904/`，tmux `avito-onetrans`，全局 batch1536、8 workers、BF16、局部编译。第1轮于15:17前后完成：完整 train 1311.38秒，val13.52秒，NDCG 0.3510416631。当前从该轮检查点恢复第2轮，GPU1+GPU2各处理768条feed，保留原全局batch、主/辅损失归一化、优化器和warmup/cosine进度（总4112步，warmup206步）。结束仍自动用best.pt推理完整test。
- 最新速度优化已部署：Fused RMSNorm训练保持BF16输出（FP32累积），验证/test沿用native FP32；序列KV按feed共享。双卡参数、梯度、结果全部经pinned CPU主存中转，因为本机GPU1→GPU2直接peer copy虽报告支持但arange值检查失败。不要改回直接跨卡`.copy_`/`.to`。每层K/V仍只存一份/每feed，模型结构和历史长度保持原定义。
- 单卡融合预检25.8万候选/秒、25.47GiB峰值；双卡正式第2轮连续记录42–44万候选/秒，约原19.8万的2.2倍，单卡峰值约13.4GiB。按稳态吞吐折算约9–10分钟/epoch，仅为估计；不能将第1轮的21.9分钟或当前估计冒充完整优化轮次结果。
- PyTorch2.5的FX tracing为进程全局状态：ReplicaTrainer对新的feed/history shape先顺序预热两卡forward/backward，再并发已编译图，防止多线程同时编译报错。每卡CUDA RNG随checkpoint保存；仅主模型一次AdamW更新。CPU分feed梯度与损失加权等价检查、双卡BF16真实前后向检查通过；GPU组件loss差8.94e-7，KV梯度最大差2.67e-5，FP32 eval/infer路径不变。

- 固定 TB 已改为跟随 `output/current`，所有模型的正式运行都会更新它；端口仍 6006，tmux 仍 `avito-tensorboard`。训练状态与 ETA 以当前 run.json / run.log 为准。源码、测试、official 参考代码及 Notebook 单元已 Black 120 检查。
- 用户要求减少文件拆分：学习率计划及 warmup/cosine 函数已合回 `src/train.py`，删除 `src/lr.py`。
- 用户要求删除架构图，已删除 drawio 和三张 PNG，并移除相关链接。12 项 CPU 测试已通过，含双卡分feed损失/梯度归一化、warmup/cosine、worker尾batch规划及恢复。

当前主入口为 `python -m src.train`、`python -m src.infer`，具体命令和日志字段见 `doc/pipeline.md`。

- 历史 TokenMixer 基线：共享类别/type/source embedding；历史每个位置 concat 后过共享两层 MLP，再做 masked mean pooling。全部底座特征 concat、分块投影、TokenMixer 提取，最后双 tower。
- 主 tower 学 target（BCE），辅助 tower 按 train.pos 升序学标准 ListMLE，默认权重 0.1；每 feed 按 M−1 归一化。pos 不进入特征。val/test 无 pos，不补造监督；full 中 val 只训练主任务。推理只用主 tower。
- user/item 官方向量训练时各以 0.1 概率整条置零，可配置 random 高斯替换；历史商品使用同样 item 策略。eval/test 禁用扰动，真实缺失保持零。
- 模式 `train-eval` 使用完整 train/val，每 epoch 验证并默认按完整 NDCG 选 best；`full --from-eval` 继承一个已完成且无限制的 eval 配方，默认训练 2 个 epoch，随机初始化在 train+val 重训，自动完成 testinfer 和模板键检查。
- run.log、metrics.jsonl、TensorBoard、config/run.json、last/best.pt 均保存在单个运行目录。支持步级检查点及优化器/scaler/RNG/数据游标恢复，回放定位时不重复优化。CPU 恢复逐参数一致性已测试。
- 本轮通过 8 项 CPU 集成测试（含双塔损失、共享编码/mask、扰动、官方 NDCG 计算一致性、恢复、full/testinfer、TensorBoard）；GPU 有限数据预检 4 步通过，仅作为吞吐/显存检查。
- 用户随后明确要求 width=128、8 workers、半精度、大 batch、尽量接近 10 分钟/epoch，以及 TB 按 Loss/效果/工程分组并去掉 start。
- 旧运行 `output/tokenmixer/20261101_121829/` 已按硬件/模型重配需求中断并保留检查点，不能把它当作当前活跃运行或完整验证结果。
- 已完成正式运行：`output/tokenmixer/20261101_123451/`，原训练 tmux `avito-tokenmixer` 已退出，Jigsaw / GPU 1，3 epochs、4096 feeds/batch、8 workers、width 128、lr 0.001、BF16；全部样本/步数限制为 0，每 10 步记录指标、每 100 步保存 last。已授权，不要误当作未授权任务停止。
- 优化包括 Arrow 列式批量解码、fused AdamW、BF16、局部编译 TokenMixer block/事件 MLP（保留原生 embedding backward）、expandable CUDA segments。编译缓存位于当前运行的 `compiler_cache/`，训练/恢复仍需要，不当成无用下载缓存删除。
- 验证和 testinfer 统一使用 eager FP32 主 tower；训练侧编译 BF16 的分数与 eager BF16 存在小幅偏差，不能将其当作严格推理分数等价。完整评估和最终导出沿用同一 FP32 路径。
- 性能预检（不是完整 epoch）：局部编译 batch4096 约 47 万候选/秒、35 GiB 分配峰值；新正式运行预热后约 52 万候选/秒，完整 epoch 计时仍以 run.json 为准。9 项 CPU 集成测试通过，含 Arrow 批量/逐行编码逐值一致性。
- 固定 TB：tmux `avito-tensorboard`，`src.tensorboard_service` 跟随 `output/current`，`http://10.176.40.144:6006/#custom_scalars`。Loss、效果、工程三组，自定义面板已验证；start 只留文本/JSON，不写 TB。效果指标在完整 val 后出现。
- 已完成原生完整 epoch 1：3,151,208 feed / 246,745,836 候选，训练 502.01 秒，完整 val 9.81 秒，合计约 8 分 32 秒。不是短测外推，已达到 10 分钟/epoch 目标。
- epoch 1 完整 val（14,423 feed / 1,389,971 候选、eager FP32）：NDCG 0.3517883、NDCG@10 0.2197133、NDCG@30 0.2893727、AUC 0.6779251、AP 0.0338343、Logloss 0.07824245。全部 3 轮已于北京时间 13:16:54 完成，训练每轮分别 502.01/484.11/482.69 秒。最佳 epoch 2：NDCG 0.3569183656、NDCG@10 0.2258994905、NDCG@30 0.2958706095、AUC 0.6893627867；best.pt 保存该轮，不当作线上成绩。

- 2026-11-01 test 推理已完成：使用上述 epoch 2 best.pt，eager FP32 / GPU 1 / batch1024 / 8 workers，耗时 23.51 秒；`output/tokenmixer/20261101_123451/submission.parquet` 完整覆盖 28,595 feed / 2,904,806 候选，格式、键唯一、有限分数与模板全覆盖检查通过，`inference.json` 标记 `submission_ready=true`。仅生成本地文件，尚未上传或提交。

`avito.ipynb` 为此前自包含 CatBoost 参考入口，默认 inspect，不与当前 PyTorch 训练混用：

| 模式 | 配置 | 行为 |
| --- | --- | --- |
| `inspect` | `DATA_DIR` 可选，默认自动寻找 `data/` | 只检查数据结构 |
| `eval` | `TRAIN_FEEDS=VAL_FEEDS=0` 表示完整数据 | train 训练、val 选树数和评估 |
| `full` | `EVAL_RUN` 指定完整 eval 运行目录 | 固定树数在 train+val 重训 |
| `predict` | `MODEL_PATH` 指定检查点 | 预测 test 并检查提交文件 |

有限 feed 的运行只作为诊断，不能标成完整验证；`full` 会拒绝使用有限数据 eval 的树数。Notebook 默认相对路径 `output/catboost/<UTC+8 timestamp>/` 保存每次结果。

官方脚本位于 `official/`，其 `--help` 可查看 CLI。CatBoost 原脚本对 Polars `Pool` 输入依赖更新版本，在 Jigsaw 中请使用已适配的 Notebook，不直接照搬作者环境安装命令。

- 官方 Transformer 使用 val Logloss 选检查点，并非 NDCG；改选择规则前需明确区别于官方复现。
- 数据包：https://storage.yandexcloud.net/avitotech-mlcup-2026/competition.zip
- 基线包：https://storage.yandexcloud.net/avitotech-mlcup-2026/baseline.zip
- 规则 PDF：https://storage.yandexcloud.net/ds-ods/files/data/docs/competitions/AvitoTechMLCup_2026_Recsys/Rules_Avito_tech_ML_cup_2026.pdf

## 通用工作约定

- 外部数据如确有需要，清洗、去重、字段对齐后优先统一到 `data/external.csv`，额外添加 `source`；保留文本大小写、标点等有效风格信息。
- 大型下载使用 tmux 后台和断点续传；按需测试 proxy/hf-mirror 吞吐，明确区分下载中、校验中、完成。成功校验后清掉压缩包、控制文件和旧日志。
- 优先获取必要基座/encoder；默认不下载其他选手 adapter/LoRA，可以研究其公开代码和数据。
- 数据、权重、凭据、实验产物不得进 Git。公开发布、上传和比赛提交遵循明确授权。
- 如需 Kaggle 离线运行，使用 uv 安装预下载的、兼容目标 Python/CUDA/PyTorch 的 wheels；本比赛本身是 ODS 文件提交，不能照搬 Kaggle 的运行限制。
- 正式训推前实测吞吐、显存；加速方案须验证语义兼容及排序/分数一致性。不要把公开作者成绩、样本跑通或组件检查描述为自有比赛得分。
