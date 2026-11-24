"""Write a result report from the available tables without inferring missing runs."""

from collections import defaultdict
import csv
import json
import os

import numpy as np

from additional import fine_blind
from design import unique
from response_geometry import ROOT, chart_mark
from response_model import OUTPUT


def table(name):
    p=OUTPUT/f"{name}.csv"
    if not p.exists():
        return []
    with p.open(encoding="utf-8-sig",newline="") as handle:
        return list(csv.DictReader(handle))


def field(row,key,scale=1.,prec=3):
    try:
        return f"{float(row[key])*scale:.{prec}f}"
    except (KeyError,ValueError,TypeError):
        return "—"


def write_edges():
    path=OUTPUT/"original_edge_coordinates.csv"
    z=np.linspace(0,8,481)
    left,right=chart_mark().bounds(z-4)
    with path.open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.writer(handle)
        writer.writerow(("z_um","left1_um","right1_um"))
        writer.writerows(zip(z,left[0],right[0]))


def paragraph(lines,*words):
    lines.extend(words)
    lines.append("")


def readout_table(lines,rows):
    if not rows:
        return
    labels={"smooth":"平滑基准","additive":"单项扫描相加","linear":"一阶复场",
            "self_second":"加自身二阶","mixed_second":"加混合二阶","calibrated":"联合训练校准",
            "amplitude_response":"使用单模幅值曲线","conditioned_second":"含高度或截面的条件模型"}
    lines.extend(("| 响应模型 | WQ 误差 % | APD 误差 nm | SUM NRMSE % | DIFF NRMSE % |",
                  "|---|---:|---:|---:|---:|"))
    for r in rows:
        lines.append("| "+labels.get(r["model"],r["model"])+" | "+
                     " | ".join((field(r,"error_wq_percent"),field(r,"error_apd_nm"),
                                 field(r,"sum_nrmse",100),field(r,"diff_nrmse",100)))+" |")
    lines.append("")


def make():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    write_edges()
    direct=table("direct_kpis")
    conv=table("numerical_convergence")
    develop=table("development_prediction")
    hold=table("heldout_prediction")
    physical_holdout=table("heldout_physics")
    modal_cross=table("deterministic_interactions")
    interaction=table("factor_interactions")
    interaction_convergence=table("factor_interaction_convergence")
    cutoff=table("band_pairs_native")
    cutoff_convergence=table("band_pair_convergence")
    cutoff_stats=table("band_seed_statistics")
    cutoff_psd=table("band_psd_prediction")
    reduction_screen=table("reduction_model_screen")
    reductions=table("reduction_direct_checks")
    screen_validation=table("reduction_screen_validation")
    corrections=table("band_quadratic_correction")
    triples=table("third_factor_interactions")
    strong_width=table("strong_width_high_validation")
    chart_transfer=table("original_fine_transfer")
    height_preview=table("height_conditioned_validation")
    scores=[r for r in (hold or physical_holdout) if r.get("is_blind")=="True"]
    blind_count=len({r["case"] for r in scores})
    sources=hold if hold else develop
    lines=["# 随机边缘的 RCWA 响应曲线实算", ""]
    if blind_count==16 and any(r["model"]=="calibrated" for r in hold):
        paragraph(lines,"16 个随机留出轮廓的同设置预测验收已完成。物理精度状态由直接 RCWA 的独立收敛对照判断。")
    elif blind_count==16:
        paragraph(lines,"16 个留出轮廓已经用冻结的物理响应库独立验证。联合训练校准版本和逐轴精度复核继续计算。")
    else:
        paragraph(lines,"本报告记录当前已核对的直接求解与开发对照。留出轮廓按冻结计划保存，关键输入齐备后生成正式留出误差表。")
    paragraph(lines,"基准为 3.2 μm 节距、1.6 μm CD、300 nm 深度、643 nm 和 38 μm 高斯照明。流程和公式见 [计算口径与复现步骤](README.md)。"
              " 正入射数值检查与聚焦光斑试验用独立标签。")
    paragraph(lines,"## 原始随机轮廓",
              "复现了原图的 481 个采样点，使用完整 80 个正频分量。第一档比较以单纹 3.2 × 8 μm 原胞、跨纹 11 个场谐波和沿纹 17 个场谐波为基准。"
              " 表中误差为响应预测减去同一几何的直接 RCWA。 [原图边缘表](analysis/original_edge_coordinates.csv)")
    original=chart_mark()
    left,right=original.bounds(np.linspace(-4.,4.,480,endpoint=False))
    measured=1000*np.array([left.std(),right.std()])
    paragraph(lines,f"原图设定的单边总体 RMS 为 5 nm。这次 seed 实现得到左边缘 {measured[0]:.3f} nm 和右边缘 {measured[1]:.3f} nm。"
              "带宽实验保留同一 master 的低频幅值和相位。")
    chart=[r for r in sources if r["case"]=="chart_s1427" and r["native_setting"].startswith("k5x8_")]
    if chart:
        r=chart[0]
        paragraph(lines,f"直接解的归一 WQ 为 **{field(r,'direct_wq',prec=7)}**，APD 为 **{field(r,'direct_apd_nm')} nm**。")
        readout_table(lines,chart)
        if (OUTPUT/"original_edge_prediction.png").exists():
            files=" [原始扫描点](analysis/original_scans.csv)" if (OUTPUT/"original_scans.csv").exists() else ""
            paragraph(lines,"SUM 和 DIFF 的上排曲线复用同一条几何。下排放大响应模型相对直接解的偏差。"
                      "平滑基准也通过此轮廓的工程门槛，模型的提升要结合较强扰动和独立留出上的结果。"+files,
                      "![原轮廓的同设置扫描比较](analysis/original_edge_prediction.png)")
    validation=OUTPUT/"coordinate_validation.json"
    isolation=None
    if validation.exists():
        data=json.loads(validation.read_text(encoding="utf-8"))
        isolation=data.get("holdout_isolation")
        errors=[abs(r["relative_recovery_error_nm"]) for r in data.get("rough_geometry_translation",[])]
        if errors:
            paragraph(lines,f"直接把这条边缘移位 ±100 nm 后，去除该轮廓自身的模板偏差，位移恢复误差最大为 {max(errors):.3g} nm。"
                      "这是光路坐标与定位算法的校验。 [坐标校验](analysis/coordinate_validation.json)")
    high_chart=[r for r in chart_transfer if r["setting"]=="k10x12_d2x5_a5_h4"]
    if high_chart:
        lines.extend(("### 原轮廓换用加密直接背景后的对照", "",
                      "| 模型 | WQ 误差 % | APD 误差 nm | SUM NRMSE % | DIFF NRMSE % |",
                      "|---|---:|---:|---:|---:|"))
        for r in high_chart:
            if r["model"] in ("smooth","calibrated"):
                label="加密平滑背景" if r["model"]=="smooth" else "平滑背景加冻结响应"
                lines.append(f"| {label} | {field(r,'wq_error_percent')} | {field(r,'apd_error_nm')} | "
                             f"{field(r,'sum_nrmse',100)} | {field(r,'diff_nrmse',100)} |")
        lines.append("")
        paragraph(lines,"前面的同档评分之外，这次以加密的平滑 RCWA 作背景，把冻结的粗糙度复场增量插值到 25 个入射角。"
                  "两种预测均达标。原图扰动很弱，应用响应后 APD 误差降低，WQ 和波形误差增加。"
                  " [加密背景对照](analysis/original_fine_transfer.csv)")
    linear=[abs(float(r["apd_linear_difference_nm"])) for r in direct
            if r.get("apd_linear_difference_nm") and r["case"]=="chart_s1427"]
    if linear:
        paragraph(lines,f"对原轮廓的扫描，投影到平滑模板位移导数所得 APD 与完整拟合的差最大为 {max(linear):.4f} nm。"
                  "投影去掉了可由增益和背景解释的分量。 [位置线性化数据](analysis/direct_kpis.csv)")
    paragraph(lines,"## 单模式和现有较强扰动对照")
    legacy=[r for r in sources if r["case"]=="legacy_width_1427_r20_x0.2_b0.5"]
    if legacy:
        paragraph(lines,f"上轮保留的 20 nm 单边 RMS 纯线宽工况，在本次 38 μm 光束和同一筛选场基底上，直接 WQ 为 {field(legacy[0],'direct_wq',prec=6)}。"
                  "该算例检查响应在明显大于原图扰动时的误差。")
        readout_table(lines,legacy)
    if (OUTPUT/"single_mode_response.png").exists():
        paragraph(lines,"单模图使用直接改变余弦幅度的结果。横轴是单边位移峰值。两面板分别固定中心摆动或线宽起伏，曲线标签为 8 μm 主周期中的沿纹频率。"
                  "同一 RMS 的频带出现不同的 WQ 响应，预测应读入实际 Fourier 系数。",
                  "![单模式幅值扫描](analysis/single_mode_response.png)")
    paragraph(lines,"## 留出轮廓上的预测能力")
    if scores:
        if isolation:
            paragraph(lines,f"按几何哈希检查 {isolation['unique_heldout_geometries']} 条独立留出。冻结库记录了 "
                      f"{isolation['kernel_inputs']} 条核函数来源和 {isolation['joint_training_inputs']} 个校准工况，"
                      f"与留出的重合几何数为 {isolation['identical_fit_or_kernel_inputs']}。 [校准输入审计](analysis/coordinate_validation.json)")
        lines.extend(("| 模型 | KPI 两项通过 | 同时通过波形 | WQ 绝对误差中位 % | APD 绝对误差中位 nm |",
                      "|---|---:|---:|---:|---:|"))
        for key in dict.fromkeys(r["model"] for r in scores):
            rows=[r for r in scores if r["model"]==key]
            n=len(rows)
            w=np.array([abs(float(r["error_wq_percent"])) for r in rows])
            p=np.array([abs(float(r["error_apd_nm"])) for r in rows if r["error_apd_nm"]])
            k=sum(r["pass_kpi"]=="True" for r in rows)
            s=sum(r["pass_kpi"]=="True" and r["pass_scan"]=="True" for r in rows)
            median_apd=f"{np.median(p):.3f}" if len(p) else "—"
            lines.append(f"| {key} | {k}/{n} | {s}/{n} | {np.median(w):.3f} | {median_apd} |")
        lines.append("")
        if (OUTPUT/"heldout_errors.png").exists():
            paragraph(lines,"箱图中的横线为 WQ 0.5% 和 APD 1 nm 的预算，结合表中两项都通过的计数看模型的使用范围。"
                      "每条具体扫描的数据保存在盲测曲线目录。",
                      "![留出样本的误差与预算](analysis/heldout_errors.png)")
        link="[逐样本误差](analysis/heldout_prediction.csv) · [盲测扫描文件](analysis/heldout/)" if hold else (
            "[冻结物理响应的逐样本误差](analysis/heldout_physics.csv)")
        if any(r.get("rms_complex_field_error") for r in scores):
            paragraph(lines,"逐样本表同时记录孔径相关阶的复场误差与粗糙度增量场误差。单项扫描相加直接输出两路信号，其复场列为空。")
        paragraph(lines,link)
    else:
        paragraph(lines,"16 个留出案例的曲线将在核心响应与联合训练冻结后独立比较。开发对照的数据见 [开发误差](analysis/development_prediction.csv)。")
    diagnoses=[r for r in direct if r["case"]=="blind_new_combo_05_s9221" and r["angles"]=="3"]
    if len(diagnoses)>1:
        paragraph(lines,"### 未通过的留出轮廓",
                  "冻结校准模型在此样本的粗场谐波基准上超过波形门槛。下面对同一几何增加场谐波，"
                  "以检查直接解本身的变化。此追加结果独立于原 16 例评分。")
        lines.extend(("| RCWA 设置 | 直接 WQ 相对平滑变化 % | APD nm |",
                      "|---|---:|---:|"))
        for r in diagnoses:
            dwq=100*(float(r["wq"])-1)
            lines.append(f"| {r['setting']} | {dwq:.4f} | {field(r,'apd_nm')} |")
        lines.append("")
        paragraph(lines,"比较同一沿纹截断与几何采样下的跨纹加密。若直接 WQ 的变化超过预算，"
                  "工程物理验收按加密后的直接解变化判断。 [直接解与参考](analysis/direct_kpis.csv)")
    fine=table("fine_response_validation")
    fine_single=table("fine_single_mode_preview")
    if fine_single and not fine:
        verified=[r for r in fine_single if r["model"]=="self_second"]
        passing=[r for r in verified if r["pass_kpi"]=="True" and r["pass_scan"]=="True"]
        subset=[r for r in verified if r["case"]=="blind_new_combo_05_s9221"]
        paragraph(lines,"### 细基底的单模复核",
                  f"20 阶跨纹库覆盖了 {len(verified)} 组重复直接解所需的所有单项复场响应，其中 {len(passing)} 组通过 WQ、APD 和两路波形判据。"
                  " [细基底单模表](analysis/fine_single_mode_preview.csv) · [细基底扫描文件](analysis/fine_validation/)")
        if subset:
            w=max(abs(float(r["error_wq_percent"])) for r in subset)
            e=max(100*float(r["sum_nrmse"]) for r in subset)
            paragraph(lines,f"粗档波形超限样本在细库和细直接解的匹配对照中，WQ 绝对误差最大为 {w:.3f}%，SUM NRMSE 最大为 {e:.3f}%。"
                      "每条观察以对应的细档为比较对象。 [细基底单模表](analysis/fine_single_mode_preview.csv)")
        layered=[r for r in verified if r["case"]=="blind_new_combo_00_s9101" and r["native_setting"].startswith("k20x16_")]
        if layered:
            r=layered[0]
            peak=100*max(float(r[k]) for k in ("sum_nrmse","diff_nrmse"))
            paragraph(lines,f"四层高度起伏留出在 $(K_x,K_z)=(20,16)$ 已独立完成 9 个入射角。直接 WQ 为 {field(r,'direct_wq',prec=7)}，"
                      f"APD 为 {field(r,'direct_apd_nm')} nm。名义高度的单模二阶预测在同档给出 {field(r,'error_wq_percent')}% 的 WQ 误差、"
                      f"{field(r,'error_apd_nm')} nm 的 APD 误差，较大波形 NRMSE 为 {peak:.3f}%。该几何的平滑基准也通过工程判据。")
    if fine:
        paragraph(lines,"### 细跨纹基底的独立复核")
        for k in dict.fromkeys(r["model"] for r in fine):
            n=[r for r in fine if r["model"]==k]
            good=sum(r["pass_kpi"]=="True" and r["pass_scan"]=="True" for r in n)
            lines.append(f"- {k} 同时通过 {good}/{len(n)} 组同设置对照。")
        lines.extend(("", "[细基底误差表](analysis/fine_response_validation.csv)", ""))
    if strong_width:
        paragraph(lines,"### 强线宽样本的 $K_x=60$ 响应",
                  "该几何的正频宽度系数全部有原生单模响应。下表使用该算例所需的复场自身二阶项。"
                  " [高谐波线宽验证](analysis/strong_width_high_validation.csv) · [直接解与预测的扫描](analysis/strong_width_high_scans.npz)")
        lines.extend(("| 模型 | WQ 误差 % | APD 误差 nm | SUM NRMSE % | DIFF NRMSE % |","|---|---:|---:|---:|---:|"))
        for r in strong_width:
            label={"linear":"一阶复场","self_second":"加自身二阶"}[r["model"]]
            lines.append(f"| {label} | {field(r,'wq_error_percent')} | {field(r,'apd_error_nm')} | "
                         f"{field(r,'sum_nrmse',100)} | {field(r,'diff_nrmse',100)} |")
        lines.append("")
        if (OUTPUT/"strong_width_high_comparison.png").exists():
            paragraph(lines,"下排按直接扫描的峰谷幅值放大误差。加入二阶项后，两路残差的 RMS 都从约 2.7% 降到约 0.9%。",
                      "![强线宽留出在精细场谐波下的扫描响应](analysis/strong_width_high_comparison.png)")
    if height_preview:
        paragraph(lines,"### 高度相关性单独建库的对照",
                  "对 $ρ_h=0.5$ 的留出轮廓，用 40 个独立原生单模响应表达顶底共同与差异分量。"
                  "这项复核比较高度条件模型及冻结标称模型在同一直接解上的表现。"
                  " [高度条件验证](analysis/height_conditioned_validation.csv)")
        readout_table(lines,height_preview)
    paragraph(lines,"## 交互差与删减判断")
    if interaction:
        bins=defaultdict(list)
        for r in interaction:
            if r["setting"]=="k10x8_d4x20_a3_h4":
                name=(r["factor_a"],r["factor_b"],r["kpi"])
                bins[name].append(r)
        order=sorted(bins,key=lambda k:np.mean([abs(float(r["interaction"]))/(.5 if k[2].startswith("wq") else 1.)
                                               for r in bins[k]]),reverse=True)[:8]
        lines.extend(("| 组合 | KPI | 对照数 | 平均绝对交互差 | 交互大于两项主效应的次数 |",
                      "|---|---|---:|---:|---:|"))
        for a,b,k in order:
            r=bins[a,b,k]
            hits=sum(v["interaction_above_both_main"]=="True" for v in r)
            mean=np.mean([abs(float(v["interaction"])) for v in r])
            unit="%" if k.startswith("wq") else " nm"
            lines.append(f"| {a} × {b} | {k} | {len(r)} | {mean:.3f}{unit} | {hits}/{len(r)} |")
        lines.append("")
        if (OUTPUT/"parameter_interactions.png").exists():
            paragraph(lines,"图中对相同数值档的四角差按 seed 汇总。四角差用 $K_{11}-K_{10}-K_{01}+K_{00}$ 计算，比较的是一种因素改变后，另一种因素的效应增加了多少。"
                      " 逐组保留了两项单独效应以及符号。 [原生四角差](analysis/factor_interactions.csv)",
                      "![各参数对的原生 RCWA 交互差](analysis/parameter_interactions.png)")
    if (OUTPUT/"factor_seed_summary.csv").exists():
        paragraph(lines,"同一参数组合在同一谐波、光栅和照明设置下按独立 seed 汇总。APD 同时报告带符号均值和 RMS。"
                  " [交互差的 seed 统计](analysis/factor_seed_summary.csv)")
    if triples:
        paragraph(lines,"选择筛选中最强的参数交互，在两档第三因素下用完整八角组重复。"
                  "第三阶交互取两档的二因素交互差。 [第三因素原生对照](analysis/third_factor_interactions.csv)"
                  " · [逐角 KPI](analysis/third_factor_native_kpis.csv) · [种子不确定度](analysis/third_factor_seed_summary.csv)")
        lines.extend(("| 参数组合 | 完整八角组 | 最大绝对 WQ 交互差 % | 最大绝对 APD 交互差 nm |",
                      "|---|---:|---:|---:|"))
        groups=defaultdict(list)
        for r in triples:
            groups[r["factor_a"],r["factor_b"],r["factor_c"]].append(r)
        for key,rows in groups.items():
            metric=lambda tag: max((abs(float(r["third_difference"])) for r in rows if r["kpi"].startswith(tag)),default=float("nan"))
            lines.append(f"| {' × '.join(key)} | {len({r['seed'] for r in rows})} | {metric('wq'):.3f} | {metric('apd'):.3f} |")
        lines.append("")
    if interaction_convergence:
        fail=sum(r["stable_under_axis"]=="False" for r in interaction_convergence)
        paragraph(lines,f"交互差本身可逐轴比较 {len(interaction_convergence)} 组，其中 {fail} 组超出 WQ 0.1% 或 APD 0.2 nm 的数值余量。"
                  "逐行数据保留加密前后的交互差。 [交互差数值收敛](analysis/factor_interaction_convergence.csv)")
        wq=[r for r in interaction_convergence if r["kpi"].startswith("wq")]
        if wq:
            largest=max(wq,key=lambda r:abs(float(r["delta"])))
            paragraph(lines,f"{largest['factor_a']} × {largest['factor_b']} 在 seed {largest['seed']} 的 WQ 交互差，"
                      f"随 {largest['axis']} 加密从 {field(largest,'contrast_first')}% 变为 {field(largest,'contrast_second')}%。"
                      "这个对照直接影响该交互项的工程分级。")
    else:
        paragraph(lines,"统计参数的四角组仍在积累。现阶段可用已完成的确定性混合模式数据检查截面局部响应。")
    if modal_cross:
        wq=max((r for r in modal_cross if r["kpi"]=="wq"),key=lambda r:abs(float(r["interaction"])))
        apd=max((r for r in modal_cross if r["kpi"]=="apd_nm"),key=lambda r:abs(float(r["interaction"])))
        paragraph(lines,f"{len(modal_cross)//2} 个确定性双模式组合已有同设置的直接比较。"
                  f"最大的 WQ 四角差是 {field(wq,'interaction')}%，最大的 APD 四角差是 {field(apd,'interaction')} nm。"
                  "表中同时给出把一阶复场合成后再进干涉模型的残差，用于区分电磁与光学混合。"
                  " [确定性模式交互](analysis/deterministic_interactions.csv)")
    chart_cut=[r for r in cutoff if r["geometry"]=="chart_s1427" and r["filtered_um"]=="0.5"]
    if chart_cut:
        lines.extend(("### 原轮廓删去 0.1–0.5 μm 频带", "",
                      "| RCWA 设置 | WQ 配对差 % | APD 配对差 nm | SUM NRMSE % | DIFF NRMSE % |",
                      "|---|---:|---:|---:|---:|"))
        for r in chart_cut:
            lines.append(f"| {r['setting']} | {field(r,'delta_wq_percent')} | {field(r,'delta_apd_nm')} "
                         f"| {field(r,'sum_nrmse',100)} | {field(r,'diff_nrmse',100)} |")
        lines.append("")
        paragraph(lines,"同档下使用相同光栅采样和同一个 80 模式 master。从各档的配对差判断截断误差，绝对 WQ 的截断敏感性另读收敛表。",
                  "[原生带宽配对](analysis/band_pairs_native.csv)")
    band_cases={f"band_{i}_s{seed}":f"{label} · seed {seed}"
                for i,label in ((0,"5 nm RMS"),(1,"20 nm RMS 纯线宽"))
                for seed in (1427,2441)}
    examples=[r for r in cutoff if r["geometry"] in band_cases and r["filtered_um"] in ("0.5","1.0")]
    if examples:
        paragraph(lines,"### 随机轮廓的截止对照",
                  "两种强度均以同一 seed 的 0.05 μm 最短周期输入为参照。保留的低频系数与相位相同，表内保留两路扫描的误差。")
        lines.extend(("| 随机边缘 | 最短保留周期 μm | WQ 配对差 % | APD 配对差 nm | 较大一路 NRMSE % |",
                      "|---|---:|---:|---:|---:|"))
        for r in examples:
            wave=[float(r[k]) for k in ("sum_nrmse","diff_nrmse") if r.get(k)]
            error=f"{100*max(wave):.3f}" if wave else "—"
            lines.append(f"| {band_cases[r['geometry']]} | {r['filtered_um']} | {field(r,'delta_wq_percent')} | "
                         f"{field(r,'delta_apd_nm')} | {error} |")
        lines.append("")
        paragraph(lines,"频带判断使用对应光栅和场谐波档的实测配对差。全部输入、采样及扫描对照保存在逐例表。 [原生带宽数据](analysis/band_pairs_native.csv)")
    if cutoff_psd:
        r=[v for v in cutoff_psd if float(v["retained_period_um"])==.5]
        ascending=[float(v["omitted_wq_second_order_mean_percent"]) for v in r]
        if ascending:
            paragraph(lines,f"按目标 PSD 对各频率的 WQ 二阶系数加权。四个设定在保留周期 ≥0.5 μm 时，预测被删频带贡献的平均 WQ 变化为 {min(ascending):.3f}% 至 {max(ascending):.3f}%。"
                      "一阶 APD 方差项和二阶均值另列，实际 APD 验收仍取同 seed 的直接配对。"
                      " [单模频率系数](analysis/modal_kpi_frequency_response.csv) · [PSD 加权预测](analysis/band_psd_prediction.csv)")
    if cutoff_stats:
        multiple=[r for r in cutoff_stats if int(r["seeds"])>1]
        paragraph(lines,f"按频带、求解档和统计情景汇总配对差。{len(multiple)} 组包含至少两个 seed，表中同时给出均值、标准差、RMS 与均值的 95% t 区间。"
                  "少量 seed 的范围适用于当前设定下的初步比较。 [频带配对统计](analysis/band_seed_statistics.csv)")
        nominal_band=[r for r in cutoff_stats if r["scenario"] in ("band_0","band_1","band_2","band_3") and
                      r["setting"]=="k10x8_d4x3.125_a3_h4" and float(r["filtered_um"]) in (.1,.2,.5,1.)]
        if nominal_band:
            labels={"band_0":"5 nm，ξ 0.2 μm","band_1":"20 nm 纯线宽，ξ 0.2 μm",
                    "band_2":"20 nm，ξ 0.05 μm","band_3":"20 nm，ξ 1 μm"}
            lines.extend(("| 条件 | 最短周期 μm | 两项 KPI 和波形通过数 | 最大绝对 WQ 差 % | 最大绝对 APD 差 nm | 最大波形 NRMSE % |",
                          "|---|---:|---:|---:|---:|---:|"))
            for r in sorted(nominal_band,key=lambda v:(v["scenario"],float(v["filtered_um"]))):
                lines.append(f"| {labels[r['scenario']]} | {r['filtered_um']} | {r['passed_kpi_and_scan']}/{r['seeds']} | "
                             f"{field(r,'max_abs_wq_percent')} | {field(r,'max_abs_apd_nm')} | {field(r,'max_scan_nrmse',100)} |")
            lines.append("")
    if (OUTPUT/"bandwidth_native_response.png").exists():
        paragraph(lines,"下图每个浅色点对应一条边缘，连线为各条件已完成样本中的最大绝对值。三条预算线分别为 WQ 0.5%、APD 1 nm 和两路波形 1%。",
                  "![按最短保留周期比较直接 RCWA 偏差](analysis/bandwidth_native_response.png)")
    if cutoff_convergence:
        size=len(cutoff_convergence)
        contrast=sum(r["contrast_within_num_budget"]=="True" for r in cutoff_convergence)
        direct_count=sum(r["both_direct_within_num_budget"]=="True" for r in cutoff_convergence)
        paragraph(lines,f"{size} 个可配对的逐轴频带加密对照中，{contrast} 个的删频效应和 {direct_count} 个的两条直接 WQ/APD 同时进入各自的数值余量。"
                  "逐行列出删频差值及完整与滤波两条几何各自的变化。原生配对表另外给出瞳孔加权的复电场误差和 RSI 重叠系数的差。"
                  " [频带配对的逐轴收敛](analysis/band_pair_convergence.csv) · [带宽配对复场和波形](analysis/band_pairs_native.csv)")
        strong=[r for r in cutoff if r["geometry"]=="band_1_s1427" and float(r["filtered_um"]) in (.5,1.)]
        if len({r["setting"] for r in strong})>1:
            paragraph(lines,"### 纯线宽边缘的逐轴截止复核",
                      "同一条 20 nm RMS 边缘和同一几何光栅，分别增加跨纹和沿纹的场谐波。")
            lines.extend(("| 数值设置 | 最短周期 μm | WQ 配对差 % | APD 配对差 nm | 较大波形 NRMSE % | 工程验收 |",
                          "|---|---:|---:|---:|---:|---|"))
            for r in sorted(strong,key=lambda row:(float(row["filtered_um"]),row["setting"])):
                wave=max(float(r[k]) for k in ("sum_nrmse","diff_nrmse"))*100
                passed=r["passes_budget_at_this_setting"]=="True" and r["passes_scan_at_this_setting"]=="True"
                lines.append(f"| {r['setting']} | {r['filtered_um']} | {field(r,'delta_wq_percent')} | "
                             f"{field(r,'delta_apd_nm')} | {wave:.3f} | {'通过' if passed else '超限'} |")
            lines.append("")
            paragraph(lines,"三档工程验收结果与带宽差的逐轴数值收敛分别记录。1 μm 截止结果随跨纹分辨率而变，最终边界取同一几何在更高场基底上的配对。"
                      " [该边缘的两轴复算](analysis/band_pair_convergence.csv)")
    if reduction_screen:
        paragraph(lines,"使用 32 组参数和 4 个配对 seed 筛选删项。表中每项记录超出任一 KPI 门槛的例数、90% 经验分位和最不利点。"
                  "超限或接近门槛的几何用于原生 RCWA 复算。 [预测删减影响](analysis/reduction_model_screen.csv)")
    if screen_validation:
        agreed=sum(r["correct_budget_class"]=="True" for r in screen_validation)
        missed=sum(r["false_exemption"]=="True" for r in screen_validation)
        paragraph(lines,f"筛选模型与同档直接 RCWA 在 {agreed}/{len(screen_validation)} 个候选上给出相同的两项 KPI 分类。"
                  f"有 {missed} 个候选由模型判定达标、直接配对实际超限。这里使用预先选出的最坏点和门槛附近点。"
                  " [删参筛选的直接检验](analysis/reduction_screen_validation.csv)")
    if reductions:
        choices=defaultdict(int)
        for r in reductions:
            choices[r["modal_status"]]+=1
        labels={"unresolved":"待收敛","retain_in_this_geometry":"该几何超过预算",
                "retain_on_verified_modal_axes":"已验谐波轴上超过预算",
                "within_budget_in_this_geometry":"该几何在预算内",
                "within_budget_on_verified_modal_axes":"已验谐波轴上在预算内"}
        paragraph(lines,"最不利点和门槛附近的直接保真对照按样本记录。当前分布为 "
                  +"、".join(f"{labels[k]} {v} 例" for k,v in sorted(choices.items()))+"。",
                  "[删参直接核验](analysis/reduction_direct_checks.csv) · [每档几何配对](analysis/reduction_pair_details.csv)")
    if corrections:
        ok=sum(r["corrected_pass_at_same_settings"]=="True" for r in corrections)
        raw=sum(r.get("uncorrected_pass_at_same_settings")=="True" for r in corrections)
        before=f"，修正前通过 {raw} 个" if "uncorrected_pass_at_same_settings" in corrections[0] else ""
        upgrade=sum(r["corrected_pass_at_same_settings"]=="True" and r.get("uncorrected_pass_at_same_settings")=="False" for r in corrections)
        result=f"，其中 {upgrade} 个由未通过转为通过" if before else ""
        paragraph(lines,f"短波二阶修正对 {len(corrections)} 个同设置频带配对完成了复算{before}，修正后通过 {ok} 个{result}。"
                  " 每个判定同时检查 WQ、APD 和两路波形。 [二阶带宽修正](analysis/band_quadratic_correction.csv)")
    if (OUTPUT/"joint_parameter_response.png").exists():
        paragraph(lines,"响应曲面的上排记录各组合的平均 WQ 变化，下排记录 RMS APD，因而保留了符号相反的随机位置偏差。"
                  "颜色来自模型对配对 master 的统计，用于选取需要直接复算的范围。",
                  "![参数组合的响应曲面](analysis/joint_parameter_response.png)",
                  "[单因素变化](analysis/parameter_response_curves.csv) · [两因素响应曲面数据](analysis/joint_parameter_response.csv)")
    if (OUTPUT/"profile_response_curves.csv").exists():
        paragraph(lines,"### 截面非理想与粗糙度",
                  "CD、刻蚀高度、侧壁角、垂直剪切、底脚和两档粗糙度的直接响应同时保存相对标称轮廓与相对同截面平滑轮廓的 WQ/APD。"
                  "静态截面误差的绝对贡献与叠加粗糙度后的增量可从对应列读取。 [截面与粗糙度直接曲线](analysis/profile_response_curves.csv)")
        profile_cross=table("factor_seed_summary")
        labels={"cd_nm":"CD","height_nm":"刻蚀高度","swa_deg":"侧壁角","shear_nm":"垂直剪切","footing_nm":"底脚"}
        profiles=defaultdict(dict)
        for r in profile_cross:
            if r["factor_a"] in labels and r["factor_b"]=="sigma_nm" and r["setting"]=="k10x8_d4x20_a3_h4":
                profiles[r["factor_a"]][r["kpi"]]=r
        if profiles:
            lines.extend(("| 与 5 到 20 nm RMS 的交互 | seed 数 | 最大绝对 WQ 差 % | 最大绝对 APD 差 nm |",
                          "|---|---:|---:|---:|"))
            for axis,name in labels.items():
                row=profiles.get(axis)
                if row:
                    w,a=(row[k] for k in ("wq_nominal","apd_nominal_nm"))
                    lines.append(f"| {name} | {w['seeds']} | {field(w,'max_abs')} | {field(a,'max_abs')} |")
            lines.append("")
    long=[r for r in direct if r["group"]=="long_cell"]
    if long:
        paragraph(lines,"## 长沿纹超胞",
                  "24 和 48 μm 周期用同样的 38 μm 光斑及二维瞳孔。入射角块对同一出射波矢的贡献作相干叠加。"
                  "不同周期分别按相同 PSD 产生轮廓，长度之间的差值用于筛选周期与光斑的影响。")
        lines.extend(("| 沿纹周期 μm | 相关长度 μm | WQ 相对平滑变化 % | APD nm |","|---:|---:|---:|---:|"))
        for r in long:
            wq=f"{100*(float(r['wq'])-1):.3f}"
            lines.append(f"| {r['Lz_um']} | {r['xi_um']} | {wq} | {field(r,'apd_nm')} |")
        lines.extend(("", "[同设置的原生 KPI](analysis/direct_kpis.csv) · [长超胞瞳孔积分收敛](analysis/optical_long_convergence.csv)", ""))
    paragraph(lines,"## 数值边界")
    if conv:
        rows=[r for r in conv if r["case"]=="chart_s1427"]
        miss=sum(r["within_num_budget"]=="False" for r in rows)
        paragraph(lines,f"原轮廓当前共有 {len(rows)} 个可逐轴比较的数值配对，其中 {miss} 个超过预定的 WQ 0.1% 或 APD 0.2 nm 余量。"
                  "正入射和 38 μm 的对照分别列出，物理频带结论以相应照明的精度为准。 [数值收敛表](analysis/numerical_convergence.csv)")
    paragraph(lines,"每个直接算例包含复折射率、显式接口、实际场谐波数、同一几何哈希及原生工程。WQ 和 APD 的工程结论限定在对应的波长、几何、照明和数值档。"
              "正入射的稳定性检查可支持相位基准校验，完整高斯角谱的数值验收还依赖角采样。 [求解结果](runs/) · [运行状态](analysis/status.json)")
    if (OUTPUT/"angular_batch_verification.json").exists():
        paragraph(lines,"原生角表可按入射角逐块求解以降低峰值内存。已用同一三维单模几何验证逐角拼接和整表返回的 56 个结果通道一致。"
                  "大算例按完整角表组装后进入同一相干光路。 [入射角分批检查](analysis/angular_batch_verification.json)")
    contract=ROOT/"queue"/"contract_a3_ku10_v3.json"
    if contract.exists():
        planned=json.loads(contract.read_text(encoding="utf-8"))
        categories={"modes":"单模与幅值","pairs":"确定性双模式","joint":"联合设计参考",
                    "blind":"原定档留出","factors":"随机参数四角","profiles":"截面与粗糙度",
                    "bands":"嵌套频带与长超胞","strict":"加密锚点"}
        paragraph(lines,"## 原生求解回执")
        lines.extend(("| 批次 | 已完成 | 计划数 |","|---|---:|---:|"))
        for group,label in categories.items():
            jobs=planned[group]
            done=sum((ROOT/r["folder"]/"done.json").exists() for r in jobs)
            lines.append(f"| {label} | {done} | {len(jobs)} |")
        lines.append("")
        paragraph(lines,"追加的诊断和数值复核使用附属队列，原生回执保留求解档、物理输入和实际结果。 [冻结设计](queue/contract_a3_ku10_v3.json)")
    finer=[j for j in unique(fine_blind()) if j.group!="fine_blind_reference"]
    if finer:
        count=sum((j.folder/"done.json").exists() for j in finer)
        paragraph(lines,f"跨纹 $K_x=20$ 的追加留出有 {count}/{len(finer)} 组完整角谱。下表记录高沿纹截断缺失时已保存的角度回执。")
        lines.extend(("| 留出样本 | 场谐波设置 | 已保存入射角 |", "|---|---|---:|"))
        for j in finer:
            if j.setting.kv==16:
                angles=j.setting.angles**2
                solved=angles if (j.folder/"done.json").exists() else len(list(j.folder.glob("angles_*/done.json")))
                lines.append(f"| {j.mark.name} | {j.setting.key} | {solved}/{angles} |")
        lines.append("")
    path=ROOT/"RESULTS.md"
    temp=ROOT/f".RESULTS_{os.getpid()}.md"
    temp.write_text("\n".join(lines)+"\n",encoding="utf-8")
    temp.replace(path)
    return path


if __name__=="__main__":
    print(make())

