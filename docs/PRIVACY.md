# 隐私定义、实现与测试边界

## 威胁模型

家庭身份、设备档位、上线/掉线、选择名单和训练配置为公开元数据。固定家庭成员集合，邻接关系是替换其中一户的整个本地数据。客户端诚实执行本地训练和裁剪，服务器按协议执行且不与客户端串谋。数据无关的调度不获得隐私放大；对全部已发布模型按最坏情况顺序组合。

SecAgg+ 隐藏单户向量，但服务器仍看得到未加噪的总和/均值。中央 DP 进一步保护服务器发布的模型，**不能保护数据免受该可信加噪服务器本身窥视**。更强威胁模型需要分布式噪声、可信执行环境、恶意安全协议等，当前未实现。

同机模拟中的多个逻辑客户端共用工作进程；拥有该进程或主机调试权限的人可读取客户端数据。无 TLS、身份签名、跨主机隔离或恶意客户端裁剪证明。日志清洗和哈希检查不构成抵抗任意主动攻击的证明。

## 敏感度与会计

1. 客户端更新 `u = local_model - broadcast_model`，裁剪至 `||u||₂ ≤ C`。
2. Flower 每坐标裁剪范围为 `[-C,C]`，量化范围 `Q=2²²`，等户权重为 1、max_weight=1。
3. 随机舍入每坐标误差不超过 `2C/Q`，因此维度 d 下使用保守界 `C_eff = C + √d·2C/Q`。这里 d=2,488。固定量化舍入种子只用于复现；敏感度按最坏误差计，不依赖该种子提供隐私。
4. m 个实际上传且成功恢复的贡献，替换一户的平均更新敏感度 `Δ₂ = 2C_eff/m`。
5. 服务器加入独立 `N(0, (σ·Δ₂)² I)`，再加到已有模型并转 FP32。噪声用 OS 支持的 `SystemRandom` 生成，未使用实验种子，未保存随机状态。
6. 使用 Google `dp-accounting==0.5.1` 的 GaussianDpEvent 和 RdpAccountant；有限阶集合包含 1.25 至 256。不使用 subsampling amplification；每次发布以 `α/(2σ²)` 的 Gaussian RDP 组成计费，再使用库的 RDP→(ε,δ) 转换。默认 δ=1e-5。

实现以常见有限精度浮点 Gaussian 采样运行，未进行抗浮点侧信道或生产密码随机采样审计；报告中的 ε 是该数学机制模型下的会计值，不宣称经过形式化验证的生产 DP。

## 预算和可观测输出

每轮开始前核查下一次发布的 ε，超预算即停止。协议中止不计费；成功聚合后、噪声生成前先写预留日志。恢复时未知是否已经发布的预留也计费，所以可能保守地多扣一次。每个模型 checkpoint 均是已加噪的模型；未加噪平均值仅短暂存在于服务器内存。

DP 运行禁止客户端返回验证/测试损失，监控不输出精度和精确训练 wall time。公共配置、参与数量、字节数和阶段仍可显示。操作系统运行时间、进程内存访问、外部控制台观测等侧信道不在模型发布 DP 保证内。

`energy_fl.benchmark` 专用于公开的 Ausgrid 数据，它独立载入数据评估 DP 模型，生成非 DP 的 post-hoc 精度表。该表不属于受保护输出，**不能将这一公开基准评估流程直接用于私人住户数据**。同一数据上的本地/集中式基线也只用于公开研究。

## 可复核安全测试

- 真实 SecAgg+ 恢复均值与直接求取裁剪平均值一致，误差限制在一个量化步长以内。
- 掩码上传消息的 ArrayRecords 为空，存在 masked_params；没有传出单户梯度、均值、标准差或私钥 Context。
- 密钥分享后掉线，正确排除掉线者更新；上传后 unmask 阶段掉线，仍恢复已经上传的贡献。
- 份额不足不返回聚合、模型不更新、ε 不增加。
- 轮次重放和参数摘要错误被拒绝；DP 评估入口直接拒绝。
- 恢复后模型与未中断的非 DP 执行逐位一致；配置变化拒绝复用 checkpoint。
- 预留后崩溃不会回退隐私计数；目录锁拒绝并发写入；预算在发布前停止。

这些检查验证实现行为，不等于成员推断攻击的经验测试，也不证明所有安全性质。

## 参考

- Flower 官方示例：[Secure Aggregation](https://flower.ai/docs/examples/flower-secure-aggregation.html)、[Differential Privacy with Secure Aggregation](https://flower.ai/docs/examples/fl-dp-sa.html)。本项目固定旧版 API 1.23.0，按安装源码验证，不依赖最新文档中的接口变化。
- [SecAggPlusWorkflow API](https://flower.ai/docs/framework/main/en/ref-api/flwr.server.workflow.SecAggPlusWorkflow.html)。
- [Google Differential Privacy / dp-accounting](https://github.com/google/differential-privacy/tree/main/python/dp_accounting)。
- I. Mironov, “Rényi Differential Privacy,” IEEE CSF, 2017. [arXiv:1702.07476](https://arxiv.org/abs/1702.07476)。
