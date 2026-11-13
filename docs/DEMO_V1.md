# Federated Home Energy — 可运行 Demo

MAIE 5532 · CAO, Jun (21342592) / ZHOU, Guanxu (21313412)

这是基于原 proposal 的第一版 CPU 仿真，实现了真实数据上的本地训练、等权 FedAvg、资源调度、预测评估和可交互结果回放。不是最终项目提交版本。原 proposal 和项目 sources 均未修改。

## 先看效果

直接用浏览器打开 `results/dashboard.html`。报告已嵌入真实运行结果，不需要联网、Python 或后台服务。

可以切换家庭查看 24 小时预测，切换调度策略、拖动轮次滑块或点击“播放”，查看客户端参与与掉线，比较各模型误差和模型压缩效果。这里是已完成实验的回放，不是实时训练画面。

Windows 在本目录打开 PowerShell，重新训练并打开报告：

```powershell
.\start.ps1
```

脚本优先使用本目录虚拟环境，然后使用本机 Codex 已有 Python，再尝试 PATH 的 Python。若 PowerShell 执行策略不允许运行脚本，直接用下面的 Python 命令即可；无需更改系统策略。

其他电脑（Python 3.11+）：

```text
python -m venv .venv
```

Windows 激活：`.venv\Scripts\Activate.ps1`；macOS/Linux 激活：`source .venv/bin/activate`。

```text
python -m pip install -r requirements.txt
python run_demo.py --privacy-demo --open
python -m unittest discover -s tests -v
```

只有 NumPy 一个第三方训练依赖，无需 PyTorch、GPU、Flower 或 Streamlit。附带准备好的数据，因此运行时不需要重新下载原始归档。

## 数据集更换

| 项目 | 原 proposal | 本版 demo |
|---|---|---|
| 数据 | Low Carbon London，5,567 户 | Ausgrid Solar Home Electricity Data，原始 300 户 |
| 范围 | 计划挑选 20–30 户 | 10 户完整实际观测，2012-07-01 至 2013-06-30 |
| 下载 | 需要另行处理大型原始库 | 本次镜像 ZIP 14,618,329 字节，含用电与其他辅助数据 |
| 运行所需数据 | 尚未实现子集 | 10 个 NPZ 合计 180,058 字节，约 176 KiB |
| 任务 | 过去 48 小时 → 未来 24 小时 | 保持不变，每日 00:00 发起一次预测 |
| 目标含义 | 家庭小时用电 | **仅 GC 一般用电**，不含 CL 可控负载和 GG 光伏发电 |

数据集换到澳大利亚装有光伏的家庭，最终报告需要同步修改地域、样本来源、代表性与预测目标，不能继续称为 London 数据。GC 并不等于 GC+CL 总负载，也不是扣除 GG 的净购电。

按户编号选取完整一年、有限、非负且 Row Quality 为空（实际读数）的前 10 户：1、3、4、5、6、7、8、9、10、11。不是随机或具有代表性的抽样；完整性筛选本身也可能造成选择偏差。

每两个半小时 **kWh 电量求和** 得到小时电量；不进行功率平均。源时间列是区间结束时间，转换后按小时区间开始标记；保留原始本地 48 槽日历，不转 UTC、不纠正夏令时，后续跨地区实验需专门处理时区。缺值不插补，缺行或非实际读数会使该户不满足完整性要求。

`data/manifest.json` 记录源地址、镜像地址、归档及每户文件 SHA-256、数据范围和处理规则。Ausgrid 官网旧地址已返回 404；本版使用公开归档镜像并核验固定 SHA-256。上游改变文件时程序会报错，不会静默接受新数据。`data/ausgrid_source_notes.pdf` 是附带的原始字段说明。

扩展为 30 户（生成新目录，保留现有 10 户样例）：

```text
python prepare_data.py --download --clients 30 --output data30
python run_demo.py --data data30 --clients-per-round 10 --output results30
```

已经有归档时可以用 `--archive PATH` 代替 `--download`。不会将邮编或光伏装机容量写入小数据包。数据权利仍属于原提供方，引用和后续使用应保留 Ausgrid 归属及原说明。

## 模型与实验设计

- 每户训练期均值/标准差只由前 70% 时间计算，并留在本地。其余 15% 验证、15% 测试，严格按时间排序。每天一个样本，标签覆盖未来 24 小时；训练、验证、测试标签不交叉。验证/测试可以使用预测时刻之前已知的历史。
- 默认每户 253 个训练窗口、55 个验证窗口、55 个测试窗口；测试从 2013-05-07 开始。
- MLP 为 `52 → 32 ReLU → 24`，2,488 个 FP32 参数。输入为 48 小时归一化用电，加星期、年内日期的 sin/cos 共 4 个日历特征；输出反归一化后截断为非负电量。
- SGD 学习率 0.03，默认 30 轮、每轮最多 5 户，每户 2 个 epoch。低档设备 batch=16，其余 batch=32。代码直接实现前向和反向传播，并用数值梯度检验。
- 同步中央拓扑，聚合使用**家庭等权**，符合 proposal 的变体。没有按样本数加权。每轮至少 2 个成功客户端才更新，带版本检查拒绝旧更新。
- 三档算力 0.5/1/2 倍、内存标记 256/1024/4096 MB、带宽 1/5/20 Mbps、在线概率 0.60/0.80/0.95。内存仅是配置标签，尚未实现实测内存或可行性筛选。
- 随机和资源调度共用由同一种子生成的在线、8% 传输掉线和耗时抖动轨迹。资源调度预留一个名额给成功贡献最少的在线设备，其余优先选估算耗时较少的设备；低档客户端降为 1 epoch。这是“调度+训练量调整”的组合对照，不能独立归因给某一个机制。
- 仿真训练耗时基准为每 100 样本每 epoch 0.30 秒，再按算力调整；传输按张量大小和带宽估算。默认每轮截止 5 秒，超时/掉线的客户端不参与聚合。**这些是合成假设，未做真机校准**。
- 通信统计包括向选中户下发和从成功户接收的 FP32 参数字节；不含失败部分包、协议、加密或 TLS。运行耗时另行用 `perf_counter` 实测。
- 非隐私联邦模型选择验证集 MAE 最小轮次，测试集只作最终报告。集中式参考和 local-only 各最多训练 60 epoch，分别按验证集选权重；不是严格等总计算量实验。
- MAE、RMSE 先逐户计算再平均（macro），worst-quintile MAE 是最差 20% 家庭的平均 MAE。

## 已附带的真实运行结果

种子 42，30 轮，以上默认参数；不要将一次运行解释为统计结论。

| 方法 | 测试 MAE / kWh | 测试 macro RMSE / kWh |
|---|---:|---:|
| 前日同期 persistence | 0.386218 | 0.673086 |
| Local-only MLP | 0.332314 | 0.547760 |
| 集中式参考 | 0.339232 | 0.545659 |
| Uniform FedAvg | 0.344155 | 0.552402 |
| 资源感知 FedAvg | 0.345462 | 0.552806 |
| Gaussian/RDP 教学模式 | 0.503881 | 0.727799 |

资源感知相对 Uniform 的 MAE 上升约 0.38%，30 轮累计仿真时间从 121.04 秒变为 94.07 秒，降低约 22.28%。**这不是达到固定验证精度的 time-to-target，不能据此声称 proposal 的 20% 目标已经完成。** 两个策略最终均覆盖全部 10 户，但贡献次数不均，资源调度更偏向高档设备。

FP32 文件 10,918 字节，INT8 存储文件 4,414 字节，缩小 2.47 倍，MAE 增加约 0.029%。量化为逐张量对称权重量化，加载后恢复 FP32，仍由 NumPy 执行；不是 ONNX/TFLite 整数推理引擎，也不减少联邦训练上行流量。报告中的延迟不包括数据预处理和文件加载。

## 隐私演示的边界

默认 `python run_demo.py` 不使用 DP。加 `--privacy-demo` 才额外跑裁剪和 Gaussian 噪声实验。

本地增量整体 L2 裁剪到 C=0.05；m 个成功户取等权平均后加独立坐标 Gaussian 噪声，标准差为 `sigma * 2C / m`，sigma=2。采用替换一个家庭的邻接定义。RDP 每次发布为 `alpha/(2*sigma^2)`，线性组合，按 `min_alpha(RDP + log(1/delta)/(alpha-1))` 转换，delta=1e-5，使用离散阶数网格，不使用抽样放大。

发布前检查下一次 ε 是否超过预算 10，默认在 12 次成功更新后停止，形式会计 ε≈9.838。噪声模式用最终发布权重，避免按未保护验证误差选择 checkpoint。

**这是教学用机制测试，不是可部署的 DP 系统**：噪声采用公开固定种子以复现；服务器在同一进程能读取个体更新与数据；结果中有未保护的逐户误差和实例预测；归一化参数也不属于发布模型。因此公式不能作为本报告或整个程序的隐私保证。当前没有 SecAgg+ 或密码学安全聚合，不伪造其通信开销。

## 文件与分工建议

| 文件 | 用途 |
|---|---|
| `prepare_data.py` | 验证归档、筛户、半小时转小时、生成清单 |
| `engine.py` | 数据切分、MLP/SGD、聚合、调度、会计公式、量化 |
| `run_demo.py` | 运行基线和实验、导出报告与指标 |
| `dashboard_template.html` | 报告模板；需要打开的是 results 下的已生成报告 |
| `results/dashboard.html` | 可直接打开的完整离线演示 |
| `results/results.json` | 配置、环境、每轮结果和详细指标 |
| `results/metrics.csv` | 最终指标，Excel 可直接查看 |
| `results/*_rounds.csv` | 每轮结果 |
| `results/*/latest_checkpoint.npz` | 最新权重、轮次、版本快照；不是完整断点恢复状态 |
| `results/resource/model_int8_storage.npz` | 可恢复为 FP32 的量化权重 |
| `tests/test_engine.py` | 9 项关键正确性验证 |

CAO Jun 可继续负责数据、预测模型、真实整数运行时和多种子质量实验；ZHOU Guanxu 可继续负责 Flower 接入、真实安全聚合、调度、公平性、可恢复状态与实时监控。实际贡献应按你们真实完成内容记录。本 demo 由 Codex 协助生成，应按课程要求说明外部工具使用情况。

## Proposal 可使用的数据替换段落（待你们确认后修改原稿）

> We will replace Low Carbon London with the Ausgrid Solar Home Electricity Data (2012–2013), which contains half-hourly readings from 300 residential customers with rooftop solar in Australia. Our initial prototype uses 10 households with complete actual general-consumption (GC) readings; subsequent experiments will expand to 20–30 households. We sum adjacent half-hour energy readings into hourly kWh and forecast the next 24 hours from the previous 48 hours and calendar features. The target is general consumption, excluding controlled load and solar generation. Household identities define federated clients, with chronological train/validation/test splits and local training-only normalization. The selected cohort is not assumed to represent the wider residential population.

## 下一阶段与本版未完成部分

1. 替换 NumPy 客户端为 PyTorch/Flower，维持现有数据与模型接口，接入 SecAgg+、身份认证和 TLS。
2. 在小客户端数下验证安全聚合阈值、掉线恢复、量化边界和真正的隐私会计，去掉固定公开噪声种子；明确哪些监控输出可公开。
3. 接入 ONNX Runtime / TFLite 真正的 INT8 运算，测内存、median/p95 延迟与误差，按能力分配推理模型。
4. 加完整 checkpoint 恢复、实时界面与资源可行性检查；扩展 30 户，至少三种子，按固定验证目标比较耗时。

## 资料与归属

- Ausgrid, *Solar home electricity data – notes (August 2014)*，原始说明附在 data 目录。原官网旧链接：https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data
- 公开归档镜像：https://github.com/rockey-abhishek/Ausgrid-Dataset （仅使用数据与原始说明，未复制其分析程序）
- UNSW CEEM 数据结构说明：https://unsw-ceem.github.io/DARTH-docs/ausgrid300/
- McMahan et al. (2017), *Communication-Efficient Learning of Deep Networks from Decentralized Data*. https://proceedings.mlr.press/v54/mcmahan17a.html
- Mironov (2017), *Rényi Differential Privacy*. https://arxiv.org/abs/1702.07476
- 后续 Flower 官方参考：https://flower.ai/docs/examples/fl-dp-sa.html 与 https://flower.ai/docs/framework/main/en/explanation-ref-secure-aggregation-protocols.html
