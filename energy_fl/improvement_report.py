"""Reproduce tables and scientific figures from the frozen improvement study."""
from collections import defaultdict
import json
from pathlib import Path
import statistics as stats
import numpy as np
from .state import atomic_json


def main():
    root=Path('evidence_v3')
    rows=json.loads((root/'evaluation_rows.json').read_text(encoding='utf-8'))
    if len(rows)!=40: raise ValueError('Wait for all 40 frozen evaluation rows')
    grouped=defaultdict(list)
    for row in rows: grouped[(row['households'],row['method'])].append(row)
    aggregates=[]
    metrics=['test_mae','test_rmse','worst_quintile_mae','target_seconds','three_round_target_seconds',
             'modeled_training_seconds','training_wire_bytes','evaluation_wire_bytes','jain_index','coverage',
             'epsilon','privacy_releases','mean_selected','mean_survivors']
    for (n,method),group in sorted(grouped.items()):
        item={'households':n,'method':method,'n':len(group)}
        for metric in metrics:
            values=[r[metric] for r in group if r.get(metric) is not None]
            if values:
                item[metric]={'mean':stats.mean(values),'std':stats.stdev(values) if len(values)>1 else 0.,'n':len(values)}
        aggregates.append(item)
    lookup={(g['households'],g['method']):g for g in aggregates}
    paired=[]
    for n in (10,30):
        cohorts={r['seed']:r for r in grouped[(n,'cohort')]}
        for comparator in ('uniform','legacy_resource','matched_uniform'):
            for old in grouped[(n,comparator)]:
                new=cohorts[old['seed']]
                paired.append({'households':n,'seed':old['seed'],'comparator':comparator,
                    'target_improvement_pct':100*(1-new['target_seconds']/old['target_seconds'])
                        if new['target_seconds'] is not None and old['target_seconds'] else None,
                    'stable_target_improvement_pct':100*(1-new['three_round_target_seconds']/old['three_round_target_seconds'])
                        if new['three_round_target_seconds'] is not None and old['three_round_target_seconds'] else None,
                    'mae_change_pct':100*(new['test_mae']/old['test_mae']-1)})
    atomic_json(root/'aggregate_metrics.json',{'groups':aggregates,'paired':paired})
    def fmt(n,method,metric,d=4,scale=1):
        val=lookup[(n,method)].get(metric)
        return f"{val['mean']/scale:.{d}f} ± {val['std']/scale:.{d}f}" if val else '—'
    def mean(n,method,metric): return lookup[(n,method)][metric]['mean']
    selected=[r for r in paired if r['households']==30 and r['comparator']=='uniform']
    improvements=[r['target_improvement_pct'] for r in selected if r['target_improvement_pct'] is not None]
    stable=[r['stable_target_improvement_pct'] for r in selected if r['stable_target_improvement_pct'] is not None]
    degradation=stats.mean(r['mae_change_pct'] for r in selected)
    dp_improvement=100*(1-mean(30,'warm_dp0001','test_mae')/mean(30,'cold_dp005','test_mae'))
    public_gap=100*(mean(30,'warm_dp0001','test_mae')/mean(30,'public_only','test_mae')-1)
    passed=len(improvements)==3 and stats.mean(improvements)>=20 and degradation<=5
    lines=['# 第二轮改进：设备档位轮换与公开预训练 DP 微调','',
        '日期：2026-11-13。先在验证集使用种子 101 选择设置、102 复核；随后冻结设置，在新的种子 201/202/203 上评估测试集。未按本轮测试结果修改方法或排除种子。测试时间段与前版相同，新增随机种子不意味着新的独立测试数据集。','',
        f"30 户新调度相对安全均匀 FedAvg，逐种子配对的平均 time-to-target 改善 **{stats.mean(improvements):.2f}%**，测试 MAE 相对变化 **{degradation:+.2f}%**。提案的 20% 加速且 MAE 劣化≤5% 目标：**{'在当前仿真达到' if passed else '在当前仿真未达到'}**。",
        f"公开预训练后用 C=0.001 进行 DP 微调，相对原 C=0.05 从头训练的 DP 模型，平均测试 MAE 降低 **{dp_improvement:.2f}%**。相对单独公开预训练模型，MAE 变化 **{public_gap:+.2f}%**；主要质量收益来自公开预训练，不能把全部收益归因于私有联邦学习。",'',
        '## 调度改动与消融','',
        '原资源策略每轮保留慢设备名额，阶段同步仍经常被低带宽链路拖住。新策略按高→中→低档位循环，组内优先成功参与少的设备；每轮人数不超过原上限，人数不足时从相邻档位补至至少 3 户。仅使用公开配置、在线状态和已发生的参与计数，不提前查看本轮掉线。',
        '由于新策略可能减少每轮人数，额外设置 matched_uniform：同一在线轨迹、同一轮人数，均匀随机选择家庭，以区分档位分组和减少参与数的效果。所有调度对照仍从相同随机初始化开始，未使用公开预训练。','',
        '| 户数 | 方法 | 测试 MAE / kWh | 到固定阈值的模拟秒数 | 训练载荷 MiB | Jain 指数 | 平均选中户数 |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for n in (10,30):
        for method in ('uniform','legacy_resource','matched_uniform','cohort'):
            lines.append(f"| {n} | {method} | {fmt(n,method,'test_mae')} | {fmt(n,method,'target_seconds',2)} | {fmt(n,method,'training_wire_bytes',2,2**20)} | {fmt(n,method,'jain_index',3)} | {fmt(n,method,'mean_selected',2)} |")
    lines += ['',
        '均值 ± 样本标准差，n=3。固定验证阈值沿用上一轮：10 户 0.26、30 户 0.29 kWh。通信统计包含真实 SecAgg+ protobuf 载荷；时间是校准 CPU 计算、模拟带宽和掉线等待的组合，不是实物网络测速。',
        f"为检查偶然越过阈值的影响，还统计连续 3 轮达到阈值的时间；30 户该指标相对 uniform 的平均配对改善为 **{stats.mean(stable):.2f}%**。该项是辅助分析。",'',
        '![30-household scheduling comparison](../evidence_v3/scheduling.png)','',
        '## DP 效用改动与消融','',
        '公开初始化模型使用另外 20 户（34–49、51–54），与联邦 30 户完全不重叠。仅用这些公开户的训练期训练，用其验证期选 epoch；没有访问联邦测试集。模型和来源 sidecar 在 artifacts/，启动时检查模型哈希和住户集合不重叠。模型架构仍是 2,488 参数 MLP。',
        '在验证阶段比较 C=0.05/0.01/0.003/0.001 后选定 C=0.001。较小裁剪降低更新敏感度与绝对噪声幅度，同时也限制可学到的更新；Gaussian 噪声倍数 σ=4 和保守 RDP 会计不变。默认 δ=1e-5，预算上限 ε=8。',
        '参数选择来自公开 Ausgrid 验证实验；搜索过程本身没有 DP 保证，不能把单次训练 ε 赋予整次调参研究。在真正私有部署中，需先在独立公开数据上固定设置，或另行保护/计费私有验证。','',
        '| 方法 | 初始模型 | C | 测试 MAE / kWh | ε | 已计费发布次数 |',
        '|---|---|---:|---:|---:|---:|']
    definitions=[('cold_dp005','随机','.05'),('cold_dp0001','随机','.001'),
                 ('public_only','公开预训练','—'),('warm_no_dp0001','公开预训练','.001'),
                 ('warm_dp0001','公开预训练','.001'),('warm_cohort_dp0001','公开预训练 + 档位轮换','.001')]
    for method,init,clip in definitions:
        lines.append(f"| {method} | {init} | {clip} | {fmt(30,method,'test_mae')} | {fmt(30,method,'epsilon',3)} | {fmt(30,method,'privacy_releases',0)} |")
    lines += ['',
        'public_only 是一个固定模型（n=1，无私有训练）；warm_no_dp0001 是同样初始化和裁剪、无噪声的对照。两个无 DP 方法的 ε 为“不适用”，不等于隐私保证为零。其余 n=3，噪声使用不可复现的 OS 随机源。',
        '隐私运行仍禁止输出未保护的精度；表中效用来自独立公开数据评估，不属于 DP 输出。中央 DP 保护在固定公开配置下发布的模型，依赖可信服务器加噪和诚实客户端；同机工作进程不构成抵御主机攻击的隔离。',
        '本轮改善保留了公开模型的质量，尚不能据此宣称私有微调对公开模型有显著精度提升。三个种子不支持统计显著性结论。','',
        '![DP utility ablations](../evidence_v3/privacy_utility.png)','',
        '## 数据、测试与运行','',
        '新增公开数据子集约 0.35 MiB，联邦 30 户子集约 0.51 MiB。原始下载归档不变；没有恢复成大型数据库。',
        '新增测试覆盖档位轮换、等人数对照、无未来掉线信息、稀疏组安全人数、公开初始化来源隔离/完整性、验证阶段不访问测试集以及小裁剪下 SecAgg+ 数值精度。详见 [测试记录](../evidence_v3/test_results.txt)。','',
        '```powershell',
        '.\\start_system.ps1 -Policy cohort -Output system_runs/cohort -Open',
        '.\\start_system.ps1 -Private -Output system_runs/private_warm -Open',
        '```','',
        'GPU 加 `-Backend cuda`；小 MLP 仍建议 CPU。公开初始化与 evaluation 模式已纳入配置指纹；新版本运行请使用新输出目录，旧版本 checkpoint 不保证兼容。恢复必须使用完全相同参数并加 -Resume。',
        '研究复现：`python -m energy_fl.improvement_study --evaluate`；统计/绘图：安装 requirements-analysis.txt 后运行 `python -m energy_fl.improvement_report`。已有完成实验仅在配置和数据匹配时复用；希望取得新的 DP 噪声样本应使用独立研究目录。',
        '原始结果：[evaluation.csv](../evidence_v3/evaluation.csv)，设计：[evaluation_design.json](../evidence_v3/evaluation_design.json)，配对统计：[aggregate_metrics.json](../evidence_v3/aggregate_metrics.json)。','',
        '## 方法参考','',
        '公开预训练与用户级 DP 组合的相关研究：Xu et al., “Learning to Generate Image Embeddings with User-Level Differential Privacy,” CVPR 2023. [论文](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Learning_To_Generate_Image_Embeddings_With_User-Level_Differential_Privacy_CVPR_2023_paper.pdf)。该工作提供方法背景；本项目的能源数据实验结果独立测量，不直接继承其性能结论。',
        '安全聚合仍使用 [Flower 官方 SecAgg+](https://flower.ai/docs/examples/flower-secure-aggregation.html)，RDP 假设见 [PRIVACY.md](PRIVACY.md)。']
    Path('docs/IMPROVEMENTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    figures(lookup,root)
    print(json.dumps({'scheduling_target_improvement_pct':stats.mean(improvements),
                      'scheduling_stable_target_improvement_pct':stats.mean(stable),
                      'test_mae_change_pct':degradation,'dp_mae_improvement_pct':dp_improvement,
                      'dp_vs_public_only_mae_change_pct':public_gap,'target_achieved':passed},indent=2))


def figures(lookup,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    names=['uniform','legacy_resource','matched_uniform','cohort']
    labels=['Uniform','Legacy\nresource','Matched-size\nuniform','Cohort']
    colors=['#4c78a8','#9ca3af','#e2a458','#169b83']
    fig,axes=plt.subplots(1,3,figsize=(13,4.1),layout='constrained')
    for ax,metric,title,ylabel in zip(axes,['target_seconds','test_mae','jain_index'],
        ['Time to validation MAE ≤ 0.29','Held-out forecasting error','Contribution fairness'],
        ['Modeled seconds (lower is better)','Test MAE / kWh (lower is better)','Jain index (higher is better)']):
        means=[lookup[(30,m)][metric]['mean'] for m in names]
        errors=[lookup[(30,m)][metric]['std'] for m in names]
        bars=ax.bar(labels,means,yerr=errors,capsize=4,color=colors,width=.65)
        ax.set(title=title,ylabel=ylabel);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
        ax.set_ylim(0,max(np.array(means)+errors)*1.25 if metric!='jain_index' else 1.12)
        ax.bar_label(bars,labels=[f'{v:.2f}' if metric!='test_mae' else f'{v:.4f}' for v in means],padding=11,fontsize=9)
    fig.suptitle('30 households · seeds 201, 202, 203 · mean ± sample SD',fontsize=13)
    fig.savefig(root/'scheduling.png');fig.savefig(root/'scheduling.svg');plt.close(fig)
    methods=['cold_dp005','cold_dp0001','public_only','warm_no_dp0001','warm_dp0001','warm_cohort_dp0001']
    labels=['Random + DP\nC=0.05','Random + DP\nC=0.001','Public model\n(no private training)',
            'Public + FL\n(no DP)','Public + DP\nC=0.001','Public + cohort DP\nC=0.001']
    means=[lookup[(30,m)]['test_mae']['mean'] for m in methods]
    errors=[lookup[(30,m)]['test_mae']['std'] for m in methods]
    fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained')
    bars=ax.bar(labels,means,yerr=errors,capsize=5,color=['#e09764','#e9bb91','#4c78a8','#9ca3af','#169b83','#68b9aa'],width=.65)
    ax.bar_label(bars,labels=[f'{v:.4f}' for v in means],padding=14)
    ax.set(ylabel='Held-out test MAE / kWh (lower is better)',ylim=(0,max(np.array(means)+errors)*1.25),
           title='DP utility ablations · 30 households · σ=4, δ=1e-5 for DP runs')
    ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Mean ± sample SD · seeds 201/202/203 (public-only: one fixed model)',fontsize=11)
    fig.savefig(root/'privacy_utility.png');fig.savefig(root/'privacy_utility.svg');plt.close(fig)


if __name__=='__main__': main()
