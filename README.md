# Federated Home Energy — Flower 系统版

MAIE 5532 · CAO Jun (21342592) / ZHOU Guanxu (21313412)

30 户真实 Ausgrid 数据的家庭用电预测系统：真实 Flower SecAgg+、可选中央差分隐私、资源与公平调度、断点恢复、实时监控和 CUDA 训练。采用单机多进程仿真，设备性能和网络仍为模拟配置。

本次补齐 **ZHOU Guanxu 的系统实现部分**。系统功能已完成并测试，性能目标需按实测结果判断：当前 30 户资源调度尚未达到 20% 加速目标，DP 模型的效用损失也较大。不是宣称整个团队项目已达到最终提交标准。

## 运行与实时监控

Python 3.12，在仓库目录运行：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-system.txt
.venv\Scripts\python.exe -m energy_fl.runner --output system_runs/demo --workers 1
```

另开终端看实时监控（仅监听本机）：

```powershell
.venv\Scripts\python.exe -m energy_fl.monitor --output system_runs/demo --open
```

打开 <http://127.0.0.1:8767/>。网页每秒更新已提交轮次、MAE、通信载荷、隐私预算、参与公平性和训练记录。输出目录由控制器锁定；已有训练须加 `--resume` 或使用新目录。

Windows 可用 `./start_system.ps1 -Open` 同时运行训练和监控；训练结束后按 Enter 关闭监控。若脚本策略不允许运行，直接使用 Python 命令，无需修改系统策略。macOS/Linux 将 Python 路径换成 `.venv/bin/python`。

## 使用你的 GPU

本机已安装并验证 **RTX 5060 Laptop GPU / PyTorch 2.7.1+cu128**。其他机器按适配驱动安装，当前测试环境的安装命令是：

```powershell
.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m energy_fl.runner --backend cuda --workers 1 --output system_runs/gpu
```

也可用 `./start_system.ps1 -Backend cuda -Output system_runs/gpu -Open`。CUDA 固定一个工作进程以控制显存。该 MLP 仅 2,488 个参数，预热后 CPU 约 **0.50 ms/epoch**，GPU 约 **13.23 ms/epoch**，默认使用 CPU 更快。两种后端的一次训练参数最大差异约 1.5e-8，见 [GPU 验证](evidence/gpu_correctness.json)。

## 安全聚合、DP 与恢复

默认真实执行 Flower SecAgg+ 的密钥交换、加密份额、掩码上传和掉线恢复。训练更新先 L2 裁剪，再等户平均。Flower protobuf 经本机进程管道传输，不是 SuperLink/TLS 网络部署。

```powershell
.venv\Scripts\python.exe -m energy_fl.runner --sigma 4 --clip 0.05 --epsilon-budget 8 --workers 1 --output system_runs/private
```

默认 δ=1e-5，RDP 不使用采样放大；下一次发布将超预算时停止。DP 运行隐藏未保护的验证/测试精度。数学定义、有限精度限制和可信服务器边界见 [隐私说明](docs/PRIVACY.md)。安全聚合不会自动让全部日志具有 DP。

中断后，用**相同数据、参数、轮数和输出目录**加 `--resume`：

```powershell
.venv\Scripts\python.exe -m energy_fl.runner --output system_runs/demo --workers 1 --resume
```

恢复已提交模型、轮次、参与计数和隐私预留；未完成轮次重新交换密钥。`--stop-after 5` 可在第 5 轮保存后退出，用来演示恢复，保留原定 `--rounds 30`。

## 数据来源与大小

| 项目 | 当前系统 |
|---|---|
| 来源 | Ausgrid Solar Home Electricity Data，镜像和哈希见 manifest |
| 正式子集 | `data30/`，30 户 × 8,760 小时；NPZ **531,267 字节，约 0.51 MiB** |
| 小样例 | `data/`，10 户；NPZ 共 180,058 字节 |
| 原始下载归档 | 14,618,329 字节，约 13.94 MiB，不上传归档 |
| 时间 | 2012-07-01 至 2013-06-30 |
| 任务 | 48 小时历史 → 未来 24 小时 GC 一般用电 |
| 本地样本 | 每户 253 训练 / 55 验证 / 55 测试窗口 |

GC 不含 CL 可控负载或 GG 光伏发电，也不等于总负载或净购电。选择完整实际读数的前 30 户，不是代表性随机样本。两段半小时 kWh 求和为小时值，保留原始本地日历槽位，未校正夏令时。数据归属 Ausgrid，保留 [来源字段说明](data/ausgrid_source_notes.pdf) 和 [30 户 manifest](data30/manifest.json)。

已附运行子集，不必重新下载。重建命令：

```text
python prepare_data.py --download --clients 30 --output data30_rebuilt
```

## 验证与实验

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m energy_fl.check_gpu
```

GPU 检查可选，普通测试不依赖 PyTorch。覆盖真实安全聚合、两个阶段掉线、份额不足、陈旧更新、预算停止、崩溃预留、目录锁和恢复一致性。

已运行 10/30 户、种子 42/43/44、30 轮实验。包含明文/安全均匀聚合、纯资源调度、资源+自适应训练、不同 DP 噪声及 persistence/local/centralized 参考，共 **54 行结果**。

- [系统评估报告](docs/EVALUATION.md)：均值、标准差、配对 time-to-target、目标是否达成。
- [逐次 CSV](evidence/benchmark.csv)、[实验配置](evidence/experiment_design.json)、[学习曲线](evidence/learning_curves.json)。
- [系统架构与工作分工](docs/SYSTEM_DESIGN.md)。
- [隐私与安全假设](docs/PRIVACY.md)。

复现实验，保留现有结果：

```powershell
.venv\Scripts\python.exe -m energy_fl.benchmark --output benchmark_runs/repeat --evidence repeat_evidence --calibration evidence/cpu_calibration.json
```

DP 噪声来自 OS 随机源，相同种子不会逐位复现；公开离线效用评估不属于 DP 输出。报告区分本机实际时间、protobuf 字节和模拟边缘时间。3 个种子属于探索性重复。

## 文件和团队接口

`energy_fl/` 是新版系统；`system_runs/`、`benchmark_runs/`、`.venv/` 和 checkpoint 均忽略上传。`evidence/` 保留公开基准结果和校准，无需云账号或付费服务。

CAO 的模型/部署工作通过共同 FP32 参数布局接入。真正 INT8 推理和团队最终报告仍需整合；旧版 INT8 只是压缩存储。原 CPU demo 继续由 `run_demo.py` / `start.ps1` 运行，`results/dashboard.html` 是旧版离线回放，说明存档于 [DEMO_V1.md](docs/DEMO_V1.md)。框架和算法引用见 [PRIVACY.md](docs/PRIVACY.md)。未修改原 proposal 或只读项目资料。
