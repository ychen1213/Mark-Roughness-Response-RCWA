# 随机 Mark 边缘的 RCWA 响应曲线

同一条边缘走两条计算路径。一条由边缘的 Fourier 系数预测反射复场，另外一条把完整轮廓交给原生三维 RCWA。两条路径在相同光路中得到 SUM/DIFF 扫描、WQ 和 APD。留出的随机轮廓用于判断响应曲线是否提高预测准确度，嵌套频带和四角对照用于判断哪些改变能省略。

[附件的方法与验收步骤](D:/download/mark_roughness_research_fixed_2026-11-23.pdf) · [原有模板定位代码](D:/工作/AH53_Gaussian38_Comparison_20261029/analysis/math_core.py:130)

## 基准和单位

硅基底上的标称矩形条纹节距 3.2 μm，宽 1.6 μm，高 300 nm。空气入射，波长 643 nm，沿纹偏振以入射方向的横波投影定义。Y 向上，X 为跨纹和扫描方向，Z 为沿纹方向。双纹主超胞为 6.4 × 8 μm，输入为原图单条纹时使用 3.2 × 8 μm。校准采用先前同机 Lumerical 2025 R2 的材料与复场参考，硅在 643 nm 的复折射率约为 3.862888 + 0.023540i。

这里的结构是 50% 占空比子周期条纹。原图单条纹使用原胞，线间相关性使用双条原胞。16 μm 母周期与完整有限 Mark 的结构、材料和仪器读出属于另一个仿真对象。当前 WQ 与 APD 使用下面的仿真定义。

光路使用强度 $1/e^2$ 直径 38 μm 的高斯光斑、NA 0.6、半径 0.2 rad/μm 的中央挡板和理想 180° RSI。复电场移到槽底以上 0.6 μm，并用同一入射归一化。短超胞可按分离的入射角块积分。长超胞在同一出射波矢相干叠加各入射角贡献，再对二维圆瞳积分。正入射与高斯照明分别保存。

SUM 的 1.6 μm 载波复幅度定义为 $A=2\langle S(x)e^{-4\pi i x/p}\rangle$。WQ 取 $|A|/|A_\mathrm{smooth}|$。APD 取平滑模板沿扫描坐标移动后的最佳拟合位移减真实位移，以 nm 表示。拟合允许正增益和常数背景，捕获区间为 ±350 nm。所有 WQ 和 APD 均使用本项目的计算口径。

## 从轮廓到复响应

1. 将边缘写成局部中心 $c=(h_L+h_R)/2$ 和局部半线宽变化 $d=(h_R-h_L)/2$。设第 $j$ 条边的正频系数为 $a_{j,n}$，实轮廓为 $2\operatorname{Re}\sum_{n>0} a_{j,n}e^{i2\pi nz/L_z}$。$a_{j,n}$ 存为 μm，单正弦峰值为 $2|a_{j,n}|$。
2. 随机种子产生标准复 Gaussian master。谱权为 $[1+(q\xi)^2]^{-H-1/2}$。左右相关 $\rho$ 把单边方差分配到中心和半线宽，比例分别为 $(1+\rho)/2$ 与 $(1-\rho)/2$。相邻条纹再分共同与差异分量。顶底不同的工况在高度上连续插值，并用强度因子保持目标局部方差。去掉高频只设置正频带宽，保留下来的系数和相位保持原值。
3. 以物理峰值 5 nm 的单模扰动为核心校准点。沿纹正频 $n$ 的一阶场响应保留 $n_\mathrm{out}=\pm n$，二阶自身项保留 $0,\pm 2n$。单模幅值和相位改变后分别按 $|a|$、$|a|^2$ 与对应的 Fourier 旋转缩放。另直接求解 2.5、10、20 nm 检查工作区间。
4. 用四角复场差除以 4 估计两模式在各自 5 nm 峰值时的交互贡献。频率 $n_i,n_j$ 的二阶交互只保留 $\pm n_i\pm n_j$。同频的零沿纹阶引入一组相对相位 90° 的直接解，以区分两个复乘积。两组符号的变换来自超胞的真实周期平移。
5. 将反射复场接入同一 RSI 计算。第一阶场也能通过场的乘积形成干涉二阶项。对比平滑基准、单项扫描相加、一阶复场、二阶自身项和带混合项的波形。只用联合训练场估计的校准版本另行标注。完整模型与各简化模型共享冻结的验证样本。
6. 幅值曲线版本用 2.5、5、10、20 nm 的同相位单模解。在先除以峰值或峰值平方后，对响应系数沿峰值平方插值。频率 $n=1,2,4,8,16$ 的同向中心和宽度有该组数据。其余项沿用 5 nm 响应系数，混合项也沿用相应四角标定。

对原浏览器轮廓保持 `seed=1427`、$\xi=0.25$ μm、$\rho=0.5$、单边目标 RMS 5 nm 和全部 80 个正频模式。原图沿纹 4 μm 对应仿真中心，保留最低 0.1 μm 的物理周期。直接解和预测使用同一几何哈希。短原胞扩展至双纹超胞时按真实跨纹位置修正相位。

### 有限差分和参数曲面

设 $E(a)$ 是某个实际入射角和出射阶的复电场，$a$ 是一种归一化边缘模式的真实峰值。用基准 $E_0$ 估计单模的斜率和自身二阶曲率。

$$
L\simeq\frac{E(+a)-E(-a)}{2a},\qquad
D\simeq\frac{E(+a)+E(-a)-2E_0}{2a^2}.
$$

这里的二阶贡献是 $a^2D$。混合项的系数由 $E(+a,+b)-E(+a,-b)-E(-a,+b)+E(-a,-b)$ 除以 $4ab$ 取得。正弦相位与负号在对应原胞中等效为沿纹平移，入射、出射波矢的相位变换在复场上执行。留出的独立相位原生计算用来验证该步骤。复响应先加入完整边缘的幅值与相位，再进 RSI 计算 $WQ$ 和 $APD$。 [相位与定位实算校验](analysis/coordinate_validation.json)

统计曲面固定一组配对 Gaussian master，只调指定的 RMS、谱形、相关或截止。对两个因素的高低档，以 $\Delta_{AB}=K_{11}-K_{10}-K_{01}+K_{00}$ 表示交互。对给定轮廓的频带删减，原生表取 $\Delta K=K_\mathrm{filtered}-K_\mathrm{full}$。WQ 差乘 100 表示相对同一平滑基准的百分点，APD 差保留符号并用 nm 表示。样本汇总同时给出有符号均值、RMS、最大绝对差和小样本均值区间。 [参数响应曲面](analysis/joint_parameter_response.csv) · [原生种子汇总](analysis/factor_seed_summary.csv) · [带宽配对](analysis/band_pairs_native.csv)

把实余弦和正弦系数组成 $a$，KPI 在平滑边缘附近可写成 $K\simeq K_0+g^Ta+a^THa/2$。零均值随机输入的局部一阶方差为 $g^TCg$，二阶均值变化为 $\mathrm{tr}(HC)/2$，其中 $C$ 是系数协方差。删频估计在相同物理频率上乘响应系数与被删 PSD。对相关边缘，先把左右、邻线和高度协方差投影到对应模式。在实算表中，频率响应给出一阶导数和自身二阶系数，PSD 预测给出总体量，独立配对给出每条边缘的误差和超限判断。 [逐频率响应](analysis/modal_kpi_frequency_response.csv) · [PSD 传播](analysis/band_psd_prediction.csv) · [原生删频配对](analysis/band_pairs_native.csv)

APD 的全量结果来自对完整波形拟合位移、增益和常数背景。线性检查把波形变化投影到模板的位移导数，先去掉模板与常数方向。这一步对应附件第 15 页的定位灵敏度，并在原生表中与完整拟合并列。 [位置投影检查](analysis/direct_kpis.csv)

## RCWA 怎样读取边缘

每一高度薄层按实际边界定义介电常数。设 $x_{L,j}(z),x_{R,j}(z)$ 是第 $j$ 条纹的边缘，$G_x=2\pi m/L_x$，$G_z=2\pi n/L_z$。可用下面的式子检查几何如何进入 Fourier 展开。

$$
\epsilon_{mn}=\epsilon_{\mathrm{air}}\delta_{m0}\delta_{n0}
+\frac{\epsilon_{\mathrm{Si}}-\epsilon_{\mathrm{air}}}{L_xL_z}
\int_0^{L_z}e^{-iG_z z}\sum_j\int_{x_{L,j}(z)}^{x_{R,j}(z)}e^{-iG_xx}\,dx\,dz.
$$

当 $G_x=0$ 时，内层积分等于线宽。当 $G_x\ne0$ 时，它由两端的复指数确定。边缘沿 $z$ 的斜率和位移改变介电常数的二维谱，随后由 $\epsilon_{m-m',n-n'}$ 耦合场谐波。Lumerical 采用显式多边形和指定的几何光栅，再在各高度层求 Fourier 模态。直接求解保存各入射角的复振幅，瞳孔与 RSI 在场层面处理这些通道。 [Ansys RCWA 求解说明](https://optics.ansys.com/hc/en-us/articles/4414575008787-RCWA-Solver-Introduction)

| 可调量 | 当前材料和几何的实算取点 | 验证时保持固定 |
|---|---|---|
| 多边形沿纹顶点间距 | 受 10 nm、沿纹光栅与最短周期的 1/12 共同约束 | 轮廓的连续 Fourier 系数 |
| 横向及沿纹几何光栅 $\Delta x,\Delta z$ | 筛选常用 4 × 20 nm，原图用 4 × 5 或 2 × 5 nm，锚点选更小步长 | 两轴场谐波和同一轮廓 |
| 跨纹、沿纹的场谐波 $K_x,K_z$ | 矩形保留 $-K_x{:}K_x$ 与 $-K_z{:}K_z$；单独增加各轴 | 光栅、截面、入射角节点 |
| 垂直切片 | 直立且顶底相同用一个刻蚀层；侧壁或顶底不同主要用四层与八层配对 | 高度与横截面的连续模型 |
| 入射角节点 | 正入射用于排查，3 × 3 与 5 × 5 网格对应高斯角谱 | 光斑、NA、挡板及同一几何 |

第一档的数值设置用于筛选，严格验收时逐轴检查 WQ 0.1% 和 APD 0.2 nm 的变化。边缘输入可保留比场基底更短的周期。短波若能通过两个散射事件重新进入瞳孔，对应的虚中间阶或受验证的二阶修正也要接受同几何的误差检查。 [Ansys RCWA 矩形谐波与光栅参数](https://optics.ansys.com/hc/en-us/articles/12959229278611-RCWA-Solver-Simulation-Object)

## 验收

| 项目 | 误差预算 |
|---|---:|
| 同一轮廓的 WQ 预测误差或带宽删减幅度 | 0.5% |
| 同一轮廓的 APD 预测误差或带宽删减幅度 | 1 nm |
| SUM/DIFF 的 RMS 波形误差，除以严格扫描峰谷幅度 | 1% |
| 单一数值轴加密造成的配对变化 | WQ 0.1%，APD 0.2 nm |

用冻结的整条几何区分训练与盲测。同一 master 的扫描点和低通版本共享归属。先检查模型能否再现同设置的 RCWA，再检查各电磁谐波轴、几何光栅、角节点和分层的差值。报告的参数删减判断绑定实际采样到的几何和数值收敛条件。样本的带符号 APD 与 RMS APD 分开报告。

## 文件与命令

| 文件或目录 | 内容 |
|---|---|
| [实算结果](RESULTS.md) | 关键对照、图表解读和使用边界 |
| [队列](queue/) | 冻结设计、已完成和失败状态、设置和求解耗时 |
| [几何](geometry/) | Fourier master、轮廓、参数、截断、签名和原始几何检查 |
| [原生 RCWA](runs/) | 每个工况的 FSP、LSF、结果 NPZ、回执和资源估算 |
| [分析](analysis/) | WQ/APD、留出误差、交互差、短波修正、曲线和模型 |

在本目录用已经安装的 Lumerical Python 运行。等待关键各组算例完成后，工作流建立冻结模型并揭示留出数据。各批次通过几何哈希和独立回执续算。

~~~powershell
$py = 'D:\Ansys Inc\v252\Lumerical\python\python.exe'
& $py -B .\design.py prepare
& $py -B .\design.py overnight --threads 6
& $py -B .\workflow.py --parameter-sweeps
& $py -B .\verify.py
~~~

用模型评估原轮廓或用户边缘。CSV 含一周期按升序的 `z_um,left1_um,right1_um`，可加 `left2_um,right2_um`。X 坐标以标称中心为参考，单位 μm。`--origin` 指定输入 Z 对应光斑中心的位置。给定截止后，用正弦和余弦拟合边缘，并记录拟合误差。响应库与输入边缘应共享已经校准的截面和 8 μm 沿纹周期。不同条纹的平均中心或 CD 仍可用于直接 RCWA。独立的平均偏置需要与之匹配的响应库。

~~~powershell
& $py -B .\response_model.py predict chart --out .\analysis\predicted_chart.npz
& $py -B .\response_model.py predict .\analysis\original_edge_coordinates.csv --origin 4 --cutoff .1 --out .\analysis\predicted_csv.npz
& $py -B .\rcwa_runner.py .\analysis\original_edge_coordinates.csv --origin 4 --cutoff .1 --ku 10 --kv 12 --du 2 --dv 5 --angles 3 --threads 4 --with-reference
~~~

跨截面或高度变化的已校准模型另存在 `analysis/conditional`。并行的细谐波结果使用相同几何和独立设置目录，因而可以从原生结果区分拟合误差与求解器差值。

