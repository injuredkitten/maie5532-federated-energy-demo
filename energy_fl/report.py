"""Summarize the saved public experiments without rerunning or selecting seeds."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics as stats
from .state import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',default='evidence')
    args=p.parse_args();root=Path(args.evidence)
    rows=json.loads((root/'benchmark_rows.json').read_text(encoding='utf-8'))
    groups=defaultdict(list)
    for r in rows: groups[(r['households'],r['method'])].append(r)
    summaries=[]
    fields=['test_mae','test_rmse','worst_quintile_mae','target_seconds','modeled_training_seconds',
            'measured_training_seconds','training_wire_bytes','evaluation_wire_bytes',
            'coverage','jain_index','epsilon','privacy_releases']
    for (n,method),group in sorted(groups.items()):
        out={'households':n,'method':method,'seeds':[r['seed'] for r in group]}
        for field in fields:
            values=[r[field] for r in group if r.get(field) is not None]
            if values:
                out[field]={'mean':stats.mean(values),'std':stats.stdev(values) if len(values)>1 else 0.,'n':len(values)}
        summaries.append(out)
    paired=[]
    for n in (10,30):
        uniform={r['seed']:r for r in groups[(n,'secure_uniform')]}
        for variant in ('secure_resource','secure_adaptive'):
            for r in groups[(n,variant)]:
                baseline=uniform[r['seed']]
                speed=(1-r['target_seconds']/baseline['target_seconds'])*100 if r['target_seconds'] is not None and baseline['target_seconds'] else None
                paired.append({'households':n,'seed':r['seed'],'method':variant,
                               'time_to_target_improvement_pct':speed,
                               'mae_degradation_pct':(r['test_mae']/baseline['test_mae']-1)*100})
    atomic_json(root/'aggregate_metrics.json',{'groups':summaries,'paired':paired})
    lines=['# 系统评估报告（公开 Ausgrid 基准）','',
        '实验日期：2026-11-13。所有 3 个种子（42/43/44）均纳入，未挑选最好的一次。',
        '10/30 户各运行 30 轮；每轮最多 5/10 户，默认 2 epoch、batch=32、LR=0.03。',
        '统一 CPU 校准、相同在线/掉线轨迹；每个 FL 方法使用最终已提交模型。',
        '时间目标在正式实验前按早期验证集试跑固定：10 户 MAE≤0.26、30 户 MAE≤0.29 kWh。',
        '本地/集中式最多 60 epoch，选择验证集最佳轮次，不是严格总 FLOP 匹配的算法比较。','',
        '原始逐次数据：[benchmark.csv](../evidence/benchmark.csv)；配置：[experiment_design.json](../evidence/experiment_design.json)；学习曲线：[learning_curves.json](../evidence/learning_curves.json)。','',
        '## 预测质量与系统效率','',
        '均值 ± 样本标准差，n=3。MAE 单位 kWh；时间为模拟边缘训练 time-to-fixed-validation-target；非真实网络时间。','',
        '| 户数 | 方法 | 测试 MAE | 到目标的模拟秒数 | 训练载荷 MiB | 成功覆盖率 | Jain 指数 |',
        '|---:|---|---:|---:|---:|---:|---:|']
    def val(g,field,d=4,scale=1):
        a=g.get(field)
        return f"{a['mean']/scale:.{d}f} ± {a['std']/scale:.{d}f}" if a else '—'
    for g in summaries:
        if g['method'].startswith('private_') or g['method']=='secure_clip005': continue
        lines.append(f"| {g['households']} | {g['method']} | {val(g,'test_mae')} | {val(g,'target_seconds',2)} | {val(g,'training_wire_bytes',2,2**20)} | {val(g,'coverage',3)} | {val(g,'jain_index',3)} |")
    lines += ['',
        'plain_uniform 未使用密码聚合；secure_uniform 使用同样的均匀选择，但份额不足可能额外中止。安全聚合包含密钥和份额载荷，因此训练通信量高于明文基线。',
        'secure_resource 只改变选择策略；secure_adaptive 再同时改变低档 epoch 和 batch。二者分开列出，不能将组合效果全部归因于调度。','',
        '## Proposal 的性能目标核查','']
    for n in (10,30):
        group=[r for r in paired if r['households']==n and r['method']=='secure_resource']
        speeds=[r['time_to_target_improvement_pct'] for r in group if r['time_to_target_improvement_pct'] is not None]
        errors=[r['mae_degradation_pct'] for r in group]
        achieved=len(speeds)==3 and stats.mean(speeds)>=20 and stats.mean(errors)<=5
        lines.append(f"- {n} 户：逐种子配对后，平均 time-to-target 改善 **{stats.mean(speeds):.2f}%**，MAE 相对变化 **{stats.mean(errors):+.2f}%**。20% 加速且 MAE 劣化≤5% 的目标：**{'达到（仅当前仿真）' if achieved else '未达到'}**。")
    lines += ['',
        '负的加速数值表示更慢。三个种子仅提供探索性重复，不能据此宣称统计显著或真实硬件加速。资源策略保留公平名额，但成功次数分布仍可能不均；覆盖 100% 不等于次数公平。',
        '当前模型非常小，带宽和掉线等待主导仿真时间；偏向快设备会改变非 IID 家庭贡献分布，达到相同误差阈值不一定更快。后续改进需要重新预定验证实验，不应根据测试集选择策略。','',
        '## 隐私强度对照','',
        '30 户、C=0.05、δ=1e-5、ε 上限 8；σ 是相对于替换一户敏感度的噪声倍数。C=0.05 无噪声行用于隔离裁剪自身的影响。','',
        '| 方法 | 测试 MAE（公开离线评估） | 最终 ε | 发布次数 |',
        '|---|---:|---:|---:|']
    for g in summaries:
        if g['method'].startswith('private_') or g['method']=='secure_clip005':
            lines.append(f"| {g['method']} | {val(g,'test_mae')} | {val(g,'epsilon',3)} | {val(g,'privacy_releases',0)} |")
    lines += ['',
        'DP 控制器本身禁止未保护的损失/精度发布。上表由独立 public benchmark 工具评估公开 Ausgrid 数据得到，不是受 DP 保护的发布流程。噪声使用 OS 随机源，所以即使相同训练种子也不会逐位复现 DP 模型。',
        '预算停止、份额不足中止、崩溃预留均已测试；隐私会计生效并不意味着模型仍有实用精度。本轮噪声精度损失较大，不能宣称已经达到隐私与效用兼顾的最终目标。完整假设见 [PRIVACY.md](PRIVACY.md)。','',
        '## CPU / GPU','']
    cpu=json.loads((root/'cpu_calibration.json').read_text(encoding='utf-8'))
    gpu=json.loads((root/'gpu_calibration.json').read_text(encoding='utf-8'))
    lines += [f"- GPU：{gpu['gpu']}，PyTorch {gpu['torch']}，实际 CUDA 运算和联邦训练已执行。",
              f"- 同一公开合成工作量（253 个样本、batch=32），预热后 5 次中位数：CPU **{cpu['epoch_seconds']*1000:.3f} ms/epoch**，GPU **{gpu['epoch_seconds']*1000:.3f} ms/epoch**。",
              f"- 此小 MLP 的 GPU 约为 CPU 耗时的 **{gpu['epoch_seconds']/cpu['epoch_seconds']:.1f} 倍**；GPU 张量峰值分配约 {gpu['gpu_peak_allocated_mb']:.1f} MiB，包含框架运行时的主机 RSS 约 {gpu['worker_rss_mb']:.0f} MiB。",
              '- GPU 开关可用，但当前实验默认 CPU 更合适；以后换更大模型再重新测量。设备档位继续使用 CPU 参照，不将 GPU 测速代入边缘性能。','',
              '## 复现和保留边界','',
              '```text',
              'python -m energy_fl.calibrate --output evidence/cpu_calibration.json',
              'python -m energy_fl.benchmark --output benchmark_runs/new-study --evidence new_evidence --calibration evidence/cpu_calibration.json',
              '```','',
              '要比较另一批结果，请使用新目录，保留原 evidence；DP 评估会有随机变化。所有训练 checkpoint 存在本机忽略目录，不上传客户端私有状态。',
              '现有 INT8 只压缩存储。真正整数推理、量化部署性能和更大模型属于后续模型/部署整合，本报告不把旧 demo 的存储缩小当作新系统的部署成果。']
    Path('docs/EVALUATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Generated docs/EVALUATION.md and aggregate_metrics.json')


if __name__=='__main__': main()
