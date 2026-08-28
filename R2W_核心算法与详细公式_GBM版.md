# R2W 核心算法与详细公式（GBM 版）

> 本文档只保留 R2W 的核心算法链：**离线逐动作反事实测量 → 构造逐臂监督数据 → GBM 学习动作效应与命中率 → 第一关判断是否存 → 第二关逐臂预测并按净价值选择存法**。
> 不包含相关工作、实验表格、系统工程细节与论文叙事。

---

## 1. 动作空间与写时状态

对第 \(t\) 条 memory，写入动作空间定义为

\[
\mathcal A=\{\texttt{none}\}\cup\mathcal P,
\]

其中

\[
\mathcal P=
\{\texttt{raw},\texttt{raw+kv},\texttt{raw+event},\texttt{raw+graph},\texttt{raw+hq},
\texttt{sum},\texttt{sum+kv},\texttt{sum+event},\texttt{sum+graph},\texttt{sum+hq}\}.
\]

因此

\[
|\mathcal P|=10,\qquad |\mathcal A|=11.
\]

其中 `none` 表示不进入热索引；`raw` 是十个存储动作中的原文基准臂。

写入时允许使用的状态为

\[
x_t=
\Big(
 t,\;t_{<t},\;\mathcal I^{\mathrm{ref}}_{t^-},\;\mathcal I^{\mathrm{hot},\pi}_{t^-}
\Big),
\]

其中所有信息都必须在第 \(t\) 条 memory 到达时可观测，不能包含未来查询或未来索引状态。

第二关对具体动作 \(p\) 的模型输入记为

\[
z_{t,p}=\phi\!\left(x_t,\operatorname{arm}(p),d_{t,p}\right),
\]

其中 \(d_{t,p}\) 表示该候选动作在构建后可直接获得的草稿统计量；若 \(p=\texttt{raw}\)，则相应草稿统计取其确定性基准值。

---

## 2. 离线反事实测量

### 2.1 查询级结局

对查询 \(q\) 和索引背景 \(b\)，冻结完整的

\[
\text{retrieve}\rightarrow\text{read}\rightarrow\text{answer}\rightarrow\text{score}
\]

流程，定义端到端结局

\[
u(q;b)\in[0,1].
\]

R2W 的价值只由该端到端结局定义；检索指标只作为诊断量，不重复进入效用函数。

### 2.2 单条 memory 的逐动作反事实差分

主训练标签在固定全原文背景 \(b^{\mathrm{raw}}\) 下测量。对训练 memory \(t_i\) 和存储动作 \(p\in\mathcal P\)，定义查询级配对差分

\[
\delta_{i,p}(q)
=
u\!\left(q;b^{\mathrm{raw}}[t_i\!\to\!p]\right)
-
u\!\left(q;b^{\mathrm{raw}}[t_i\!\to\!\varnothing]\right).
\]

定义该 memory 被查询实际需要的概率

\[
\pi_i
=\Pr_{q\sim\mathcal D_Q}\!\left[t_i\in E(q)\right].
\]

相关查询集合与抽样无关查询集合分别记为

\[
\mathcal Q_i^{+}=\{q:t_i\in E(q)\},
\qquad
\mathcal Q_{i,\perp}\subseteq\{q:t_i\notin E(q)\}.
\]

令

\[
n_i^{+}=|\mathcal Q_i^{+}|,
\qquad
n_i^{-}=|\mathcal Q_{i,\perp}|.
\]

相关层的样本均值为

\[
\widehat b^{\mathrm{rel}}_{i,p}
=
\frac{1}{n_i^{+}}
\sum_{q\in\mathcal Q_i^{+}}
\delta_{i,p}(q).
\]

为降低少样本 memory 的标签方差，使用经验贝叶斯式收缩。令 \(c_i\) 为 memory 内容类型，\(\bar b_{p,c_i}\) 为训练区中对应“动作 × 类型”的冻结均值，则

\[
\alpha_i=\frac{n_i^{+}}{n_i^{+}+n_0},
\]

\[
\widetilde b^{\mathrm{rel}}_{i,p}
=
\alpha_i\widehat b^{\mathrm{rel}}_{i,p}
+(1-\alpha_i)\bar b_{p,c_i}.
\]

无关查询造成的有符号外部性定义为

\[
\widehat\Psi_{i,p}
=
-(1-\pi_i)
\frac{1}{n_i^{-}}
\sum_{q\in\mathcal Q_{i,\perp}}
\delta_{i,p}(q).
\]

因此动作 \(p\) 相对于完全不进入热索引的**在场效应**为

\[
\boxed{
\widehat{\Delta E}^{\,\mathrm{obs}}_{i,p}
=
\pi_i\widetilde b^{\mathrm{rel}}_{i,p}
-
\widehat\Psi_{i,p}
}
\]

等价地，

\[
\boxed{
\widehat{\Delta E}^{\,\mathrm{obs}}_{i,p}
=
\pi_i\widetilde b^{\mathrm{rel}}_{i,p}
+
(1-\pi_i)
\frac{1}{n_i^{-}}
\sum_{q\in\mathcal Q_{i,\perp}}
\delta_{i,p}(q)
}.
\]

若 \(\pi_i=0\)，约定相关收益项 \(\pi_i\widetilde b^{\mathrm{rel}}_{i,p}=0\)。

### 2.3 形式轴效应

以存原文 `raw` 为形式轴基准，定义

\[
\boxed{
\widehat{\Delta F}^{\,\mathrm{obs}}_{i,p}
=
\widehat{\Delta E}^{\,\mathrm{obs}}_{i,p}
-
\widehat{\Delta E}^{\,\mathrm{obs}}_{i,\texttt{raw}}
}.
\]

由于减去的是与 \(p\) 无关的常数，因此

\[
\boxed{
\arg\max_{p\in\mathcal P}\widehat{\Delta E}^{\,\mathrm{obs}}_{i,p}
=
\arg\max_{p\in\mathcal P}\widehat{\Delta F}^{\,\mathrm{obs}}_{i,p}
}.
\]

因此模型可以统一学习 \(\Delta E_p\)，存在轴和形式轴不需要各训练一套互不相容的 value 标签。

---

## 3. 离线成本与检索命中率

对动作 \(p\)，定义其在查询分布下进入 top-\(k\) 的命中率

\[
\rho_{i,p}^{\mathrm{obs}}
=
\mathbb E_{q\sim\mathcal D_Q}
\left[
\mathbf 1\!\left(
 t_i\in\operatorname{TopK}
 \big(q;b^{\mathrm{raw}}[t_i\!\to\!p]\big)
\right)
\right].
\]

若离线查询样本 \(q_j\) 带有归一化权重 \(\omega_j\)，其中 \(\sum_j\omega_j=1\)，经验估计为

\[
\widehat\rho_{i,p}^{\mathrm{obs}}
=
\sum_j\omega_j
\mathbf 1\!\left(
 t_i\in\operatorname{TopK}
 \big(q_j;b^{\mathrm{raw}}[t_i\!\to\!p]\big)
\right).
\]

三个资源成本中，能够在动作构建后精确计数的部分不交给 GBM 学习：

\[
C^{\mathrm{write}}_{i,p}
=
\operatorname{Cost}_{\mathrm{build}}(t_i,p),
\]

\[
C^{\mathrm{index}}_{i,p}
=
\operatorname{Size}_{\mathrm{index}}(t_i,p),
\]

\[
L^{\mathrm{ctx}}_{i,p}
=
\operatorname{Tok}\!\left(\text{正文进入 reader 的内容}\right).
\]

读取成本由正文长度和未来命中率共同决定：

\[
\boxed{
C^{\mathrm{read}}_{i,p}
=
L^{\mathrm{ctx}}_{i,p}\rho_{i,p}^{\mathrm{obs}}
}.
\]

给定运营汇率 \(\lambda=(\lambda_w,\lambda_r,\lambda_s)\)，离线可观测的逐动作净价值为

\[
\boxed{
V^{\mathrm{obs}}_{i,p}
=
\widehat{\Delta E}^{\,\mathrm{obs}}_{i,p}
-
\lambda_w C^{\mathrm{write}}_{i,p}
-
\lambda_r L^{\mathrm{ctx}}_{i,p}\rho_{i,p}^{\mathrm{obs}}
-
\lambda_s C^{\mathrm{index}}_{i,p}
}.
\]

`none` 是固定零点：

\[
\boxed{V^{\mathrm{obs}}_{i,\texttt{none}}=0}.
\]

因此训练样本上的 oracle 决策为

\[
\boxed{
V^{\mathrm{obs}}_{i,\mathrm{store}}
=
\max_{p\in\mathcal P}V^{\mathrm{obs}}_{i,p}
}
\]

以及

\[
\boxed{
a_i^{\mathrm{oracle}}
=
\begin{cases}
\texttt{none}, & V^{\mathrm{obs}}_{i,\mathrm{store}}\le 0,\\[4pt]
\displaystyle\arg\max_{p\in\mathcal P}V^{\mathrm{obs}}_{i,p}, & V^{\mathrm{obs}}_{i,\mathrm{store}}>0.
\end{cases}}
\]

这里的关键纪律是：**训练库保存效应、命中率和成本分量，不把带 \(\lambda\) 的净值作为固定标签。** 因此改变 \(\lambda\) 时只需要重新做净值合成，不需要重新做反事实测量。

---

## 4. 逐臂监督数据构造

若粗筛导致部分动作没有进入精确重放，定义测量掩码

\[
m_{i,p}\in\{0,1\},
\]

其中 \(m_{i,p}=1\) 表示动作 \(p\) 对 memory \(i\) 有可靠的层 2 反事实标签。

第二关的训练数据写为

\[
\mathcal D_{\mathrm{arm}}
=
\left\{
\left(
 z_{i,p},
 \widehat{\Delta E}^{\,\mathrm{obs}}_{i,p},
 \widehat\rho^{\mathrm{obs}}_{i,p},
 \operatorname{SE}_{i,p},
 m_{i,p}
\right)
\right\}_{i,p}.
\]

如果所有候选存储动作都被精确测量，则对所有 \((i,p)\) 有

\[
m_{i,p}=1.
\]

---

## 5. GBM 训练

R2W 不把训练时的最优动作压缩成一个分类标签。模型学习的是

\[
(x_i,p)\longmapsto \Delta E_{i,p}
\]

和

\[
(x_i,p)\longmapsto \rho_{i,p}.
\]

### 5.1 第二关：逐臂效应模型

使用一个跨动作共享数据的 pooled GBM：

\[
\boxed{
\widehat{\Delta E}_{i,p}
=
f_E^{\mathrm{GBM}}(z_{i,p})
}.
\]

令 Huber 损失为

\[
H_\delta(r)
=
\begin{cases}
\frac12r^2,& |r|\le\delta,\\[4pt]
\delta\left(|r|-\frac12\delta\right),&|r|>\delta.
\end{cases}
\]

则效应模型的训练目标可写为

\[
\boxed{
\theta_E^*
=
\arg\min_{\theta_E}
\frac{
\displaystyle\sum_{i,p}
m_{i,p}w_{i,p}
H_\delta\!\left(
\widehat{\Delta E}^{\,\mathrm{obs}}_{i,p}
-f_E(z_{i,p};\theta_E)
\right)
}{
\displaystyle\sum_{i,p}m_{i,p}w_{i,p}
}
}.
\]

默认可取

\[
w_{i,p}=1.
\]

若使用离线标签标准误进行逆方差加权，则可取

\[
w_{i,p}
=
\frac{1}{\operatorname{SE}_{i,p}^{2}+\varepsilon}.
\]

### 5.2 第二关：命中率模型

GBM 的原始输出记为 \(g_\rho(z)\in\mathbb R\)，通过 sigmoid 将预测约束在 \([0,1]\)：

\[
\boxed{
\widehat\rho_{i,p}
=
\sigma\!\left(g_\rho^{\mathrm{GBM}}(z_{i,p})\right)
=
\frac{1}{1+\exp[-g_\rho^{\mathrm{GBM}}(z_{i,p})]}
}.
\]

对 \(y\in[0,1]\) 使用 soft-label Bernoulli log loss：

\[
\ell_\rho(y,\hat y)
=
-y\log\hat y-(1-y)\log(1-\hat y).
\]

训练目标为

\[
\boxed{
\theta_\rho^*
=
\arg\min_{\theta_\rho}
\frac{
\displaystyle\sum_{i,p}
m_{i,p}
\ell_\rho\!\left(
\widehat\rho^{\mathrm{obs}}_{i,p},
\widehat\rho_{i,p}
\right)
}{
\displaystyle\sum_{i,p}m_{i,p}
}
}.
\]

### 5.3 第一关：收益包络模型

第一关只负责廉价判断“是否值得继续进入第二关”。定义训练标签

\[
\boxed{
Y_{b,i}
=
\max_{p\in\mathcal P_i^{\mathrm{obs}}}
\widehat{\Delta E}^{\,\mathrm{obs}}_{i,p}
}
\]

其中

\[
\mathcal P_i^{\mathrm{obs}}
=
\{p\in\mathcal P:m_{i,p}=1\}.
\]

若所有动作都测量，则 \(\mathcal P_i^{\mathrm{obs}}=\mathcal P\)。

使用仅包含廉价写时特征的输入 \(x_i^{\mathrm{cheap}}\)：

\[
\boxed{
\widehat y_{b,i}
=
f_b^{\mathrm{GBM}}(x_i^{\mathrm{cheap}})
}.
\]

其回归目标为

\[
\boxed{
\theta_b^*
=
\arg\min_{\theta_b}
\sum_i
H_\delta\!\left(
Y_{b,i}-f_b(x_i^{\mathrm{cheap}};\theta_b)
\right)
}.
\]

---

## 6. 预测不确定度

使用 \(K\) 个按对话分组训练的 GBM 实例或不同随机种子模型。对第二关动作 \(p\)，第 \(k\) 个模型输出

\[
\widehat{\Delta E}^{(k)}_p(x).
\]

集成均值为

\[
\boxed{
\widehat{\Delta E}_p(x)
=
\frac1K
\sum_{k=1}^{K}
\widehat{\Delta E}^{(k)}_p(x)
}
\]

对应的不确定度定义为模型间样本标准差

\[
\boxed{
\widehat\sigma_p(x)
=
\sqrt{
\frac{1}{K-1}
\sum_{k=1}^{K}
\left(
\widehat{\Delta E}^{(k)}_p(x)
-
\widehat{\Delta E}_p(x)
\right)^2
}
}.
\]

第一关同理：

\[
\widehat y_b(x)
=
\frac1K\sum_{k=1}^{K}\widehat y_b^{(k)}(x),
\]

\[
\boxed{
\widehat\sigma_b(x)
=
\sqrt{
\frac{1}{K-1}
\sum_{k=1}^{K}
\left(
\widehat y_b^{(k)}(x)-\widehat y_b(x)
\right)^2
}
}.
\]

---

## 7. 第一关：存在轴的廉价准入

真实的存在轴 oracle 目标是

\[
V_{\mathrm{store}}(t)
=
\max_{p\in\mathcal P}
\left[
\Delta E_p(t)
-
\lambda_w C^{\mathrm{write}}_p(t)
-
\lambda_r L^{\mathrm{ctx}}_p(t)\rho_p(t)
-
\lambda_s C^{\mathrm{index}}_p(t)
\right].
\]

在线第一关为了避免构建全部候选草稿，使用廉价成本近似。令 \(c(x)\) 为当前 memory 的类型，定义

\[
\boxed{
\bar C(x;\lambda)
=
\lambda_w\bar W_{c(x)}
+
\lambda_r L^{\mathrm{ctx}}_{\texttt{raw}}(x)\bar\rho_{c(x)}
+
\lambda_s\bar I(x)
}.
\]

其中 \(\bar W_{c(x)}\) 与 \(\bar\rho_{c(x)}\) 是训练区冻结的类型级统计量，\(\bar I(x)\) 是写时可确定计算的索引成本近似。

第一关的部署判决为

\[
\boxed{
\text{若 }
\widehat y_b(x_t)
-
\bar C(x_t;\lambda)
+
\kappa_E\widehat\sigma_b(x_t)
\le 0,
\quad
 a_t=\texttt{none}
}.
\]

否则进入第二关：

\[
\boxed{
\widehat y_b(x_t)
-
\bar C(x_t;\lambda)
+
\kappa_E\widehat\sigma_b(x_t)
>0
\quad\Longrightarrow\quad
\text{Stage 2}
}.
\]

这里 \(+\kappa_E\widehat\sigma_b\) 的方向是刻意保守：不确定度越高，越不容易把 memory 直接判为 `none`。

需要区分：上述第一关是**计算预算驱动的近似 gate**，不等同于精确的 \(\max_p V_p\)。若验证集发现该近似导致明显决策误差，可退化为廉价特征下的逐臂预测：

\[
\widehat V_{p}^{\mathrm{cheap}}(x)
=
\widehat{\Delta E}^{\mathrm{cheap}}_p(x)
-
\widehat C_p^{\mathrm{cheap}}(x),
\]

\[
\boxed{
\max_p\widehat V_p^{\mathrm{cheap}}(x)\le0
\Longrightarrow
\texttt{none}
}.
\]

---

## 8. 第二关：形式轴逐臂 GBM 决策

第一关放行后，系统只构建短名单候选动作。定义通过构建契约与事实证书的可行集

\[
\boxed{
\mathcal F_t
=
\{\texttt{raw}\}
\cup
\left\{
p:
\operatorname{build}(p)=1
\land
\operatorname{contract}(p)=1
\land
F_p\ge\theta_{\mathrm{fid}}
\right\}
}.
\]

对每个 \(p\in\mathcal F_t\)，使用同一个 pooled GBM 分别打分：

\[
\widehat{\Delta E}_p(t)
=
f_E^{\mathrm{GBM}}(z_{t,p}),
\]

\[
\widehat\rho_p(t)
=
\sigma\!\left(g_\rho^{\mathrm{GBM}}(z_{t,p})\right).
\]

构建和索引成本直接由当前候选产物精确计算：

\[
C_p^{\mathrm{write}}(t),
\qquad
C_p^{\mathrm{index}}(t),
\qquad
L_p^{\mathrm{ctx}}(t).
\]

预测读取成本为

\[
\boxed{
\widehat C_p^{\mathrm{read}}(t)
=
L_p^{\mathrm{ctx}}(t)\widehat\rho_p(t)
}.
\]

因此每个候选动作的风险调整预测净价值为

\[
\boxed{
\widehat V_p(t)
=
\widehat{\Delta E}_p(t)
-
\lambda_w C_p^{\mathrm{write}}(t)
-
\lambda_r L_p^{\mathrm{ctx}}(t)\widehat\rho_p(t)
-
\lambda_s C_p^{\mathrm{index}}(t)
-
\kappa_F\widehat\sigma_p(t)
}.
\]

最终形式选择为

\[
\boxed{
 p_t^*
=
\arg\max_{p\in\mathcal F_t}
\widehat V_p(t)
}.
\]

在线动作最终写为

\[
\boxed{
 a_t
=
\begin{cases}
\texttt{none},
&
\widehat y_b(x_t)-\bar C(x_t;\lambda)+\kappa_E\widehat\sigma_b(x_t)\le0,
\\[8pt]
\displaystyle
\arg\max_{p\in\mathcal F_t}
\left[
\widehat{\Delta E}_p(t)
-\lambda_w C_p^{\mathrm{write}}(t)
-\lambda_r L_p^{\mathrm{ctx}}(t)\widehat\rho_p(t)
-\lambda_s C_p^{\mathrm{index}}(t)
-\kappa_F\widehat\sigma_p(t)
\right],
&
\text{otherwise}.
\end{cases}
}.
\]

若第二行的 argmax 落在

\[
p_t^*=\texttt{raw},
\]

则直接存原文。

---

## 9. 测试时 GBM 实际执行的计算

对一条新的 test memory \(t_*\)，第二关本质上不是让 GBM 输出一个 action 类别，而是对每个候选 action 分别预测：

\[
\begin{aligned}
(\widehat{\Delta E}_{\texttt{raw}},\widehat\rho_{\texttt{raw}})
&=F(z_{*,\texttt{raw}}),\\
(\widehat{\Delta E}_{\texttt{raw+kv}},\widehat\rho_{\texttt{raw+kv}})
&=F(z_{*,\texttt{raw+kv}}),\\
(\widehat{\Delta E}_{\texttt{raw+event}},\widehat\rho_{\texttt{raw+event}})
&=F(z_{*,\texttt{raw+event}}),\\
&\ \vdots\\
(\widehat{\Delta E}_{p},\widehat\rho_{p})
&=F(z_{*,p}).
\end{aligned}
\]

其中 \(F\) 表示效应 GBM 与命中率 GBM 的联合使用，而不是一个 action classifier。

随后对每个动作计算

\[
\widehat V_p
=
\widehat{\Delta E}_p
-
\lambda_w C_p^{\mathrm{write}}
-
\lambda_r L_p^{\mathrm{ctx}}\widehat\rho_p
-
\lambda_s C_p^{\mathrm{index}}
-
\kappa_F\widehat\sigma_p,
\]

并执行

\[
\boxed{
p^*=\arg\max_p\widehat V_p}.
\]

因此 R2W 的核心学习问题可以压缩写成

\[
\boxed{
\text{Train:}
\qquad
(x_i,p)
\longrightarrow
\big(\Delta E_{i,p},\rho_{i,p}\big)
}
\]

\[
\boxed{
\text{Test:}
\qquad
(x_*,p)
\xrightarrow{\mathrm{GBM}}
\big(\widehat{\Delta E}_{*,p},\widehat\rho_{*,p}\big)
\longrightarrow
\widehat V_{*,p}
\longrightarrow
\arg\max_p
}
\]

而不是

\[
x_*\xrightarrow{\text{classifier}}p.
\]

---

## 10. 验证时的核心决策指标

对于验证/test 中拥有反事实真值的 memory，定义观测净价值

\[
V^{\mathrm{obs}}_{i,p}
=
\widehat{\Delta E}^{\,\mathrm{obs}}_{i,p}
-
\lambda_w C^{\mathrm{write}}_{i,p}
-
\lambda_r L^{\mathrm{ctx}}_{i,p}\rho^{\mathrm{obs}}_{i,p}
-
\lambda_s C^{\mathrm{index}}_{i,p},
\]

并设

\[
V^{\mathrm{obs}}_{i,\texttt{none}}=0.
\]

oracle 动作为

\[
a_i^*
=
\arg\max_{a\in\mathcal A_i^{\mathrm{obs}}}
V^{\mathrm{obs}}_{i,a}.
\]

模型实际选择为 \(\widehat a_i\)，则单条 memory 的 decision regret 为

\[
\boxed{
R_i
=
V^{\mathrm{obs}}_{i,a_i^*}
-
V^{\mathrm{obs}}_{i,\widehat a_i}
\ge0
}.
\]

平均决策遗憾为

\[
\boxed{
\overline R
=
\frac1N\sum_{i=1}^{N}R_i
}.
\]

模型选择时应优先最小化 \(\overline R\)，因为 R2W 的最终目标是选出高净价值动作，而不只是让 \(\Delta E\) 的均方误差最低。

---

## 11. 核心算法总式

整个 R2W-GBM 可以最终浓缩为以下三组公式。

### 离线测量

\[
\boxed{
\widehat{\Delta E}^{\,\mathrm{obs}}_{i,p}
=
\mathbb E_{q\sim\mathcal D_Q}
\left[
 u\!\left(q;b^{\mathrm{raw}}[t_i\!\to\!p]\right)
-
 u\!\left(q;b^{\mathrm{raw}}[t_i\!\to\!\varnothing]\right)
\right]
}
\]

### GBM 学习

\[
\boxed{
(x_i,p)
\xrightarrow{\mathrm{GBM}}
\left(
\widehat{\Delta E}_{i,p},
\widehat\rho_{i,p}
\right)
}
\]

### 在线决策

\[
\boxed{
\widehat V_p
=
\widehat{\Delta E}_p
-
\lambda_w C_p^{\mathrm{write}}
-
\lambda_r L_p^{\mathrm{ctx}}\widehat\rho_p
-
\lambda_s C_p^{\mathrm{index}}
-
\kappa_F\widehat\sigma_p
}
\]

\[
\boxed{
 p^*=\arg\max_{p\in\mathcal F_t}\widehat V_p
}
\]

加上第一关的廉价准入：

\[
\boxed{
\widehat y_b(x_t)-\bar C(x_t;\lambda)+\kappa_E\widehat\sigma_b(x_t)\le0
\quad\Longrightarrow\quad
 a_t=\texttt{none}.
}
\]

这四个式子构成当前 R2W 核心算法的最短数学描述。
