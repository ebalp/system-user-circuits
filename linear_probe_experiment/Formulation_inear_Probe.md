---
title: 'Mathematical Formulation: Linear Probe Pipeline'

---

# Mathematical Formulation: Linear Probe Pipeline

## Preliminaries

A transformer with $L$ layers processes a token sequence $(x_1, \ldots, x_T)$. Each token is mapped to an initial embedding $h_t^{(0)} \in \mathbb{R}^d$ where $d$ is the hidden dimension.

Each layer $l \in \{0, \ldots, L-1\}$ applies two sub-modules with residual connections:

$$h_t^{(l)} = h_t^{(l-1)} + a_t^{(l)} + m_t^{(l)}$$

where $a_t^{(l)}$ is the multi-head self-attention output and $m_t^{(l)}$ is the MLP (feed-forward) output. The cumulative sum $h_t^{(l)}$ is the **residual stream** at layer $l$, position $t$.

Multi-head self-attention computes queries, keys, values from the residual stream, produces per-head attention-weighted sums over values, and projects back to $\mathbb{R}^d$. It enables each token to read information from other positions. The MLP applies a nonlinear transformation independently to each token's representation, typically expanding to $4d$ dimensions and projecting back.

After the final layer, $h_t^{(L-1)}$ is passed through a layer norm and an unembedding matrix $W_U \in \mathbb{R}^{V \times d}$ to produce logits over the vocabulary of size $V$. The next token is sampled from $\text{softmax}(W_U \, h_T^{(L-1)})$.

**Notation used throughout:**

| Symbol | Meaning |
|---|---|
| $T$ | sequence length (number of tokens) |
| $L$ | number of transformer layers |
| $d$ | hidden dimension (e.g. 2048 for 1B, 4096 for 8B) |
| $h_t^{(l)}$ | residual stream at layer $l$, token $t$ |
| $a_t^{(l)}$ | attention output at layer $l$, token $t$ |
| $m_t^{(l)}$ | MLP output at layer $l$, token $t$ |
| $\tilde{h}$ | standardized activation (zero mean, unit variance per dimension) |
| $w^{(l)}, b^{(l)}$ | probe weight vector and bias at layer $l$ |
| $N$ | number of samples |
| $y_i$ | binary label (1 = followed system, 0 = followed user) |


## Setup

Input prompt $x$ consists of system instruction $x_s$ and user instruction $x_u$, wrapped in a chat template and tokenized into $(x_1, \ldots, x_T)$. Three segment boundaries partition the sequence: system segment $[1, T_s]$, user segment $[T_s+1, T_u]$, generation prompt $[T_u+1, T]$.

Each sample carries metadata: constraint type $c_i \in \mathcal{C}$, system strength $s_i \in \mathcal{S}$, user style $u_i \in \mathcal{U}$, direction $d_i \in \{a\_to\_b, b\_to\_a\}$.

The model generates completion $\hat{y}_{\text{gen}}$. A judge (deterministic parser or LLM) assigns binary label $y_i \in \{0, 1\}$ where $y_i = 1$ means the model followed the system instruction. System Compliance Rate is $\text{SCR} = \frac{1}{N}\sum_{i=1}^N y_i$.

At layer $l \in \{0, \ldots, L-1\}$, the residual stream output at token position $t$ is $h_t^{(l)} \in \mathbb{R}^d$. We extract activations at a chosen token position $t^*$ (default: $t^* = T$, the last prompt token before generation).

All probes operate on standardized activations. Given training set activations $\{h_{t^*,i}^{(l)}\}$, define per-dimension statistics $\mu_j = \frac{1}{N_{\text{train}}}\sum_i h_{t^*,i,j}^{(l)}$ and $\sigma_j = \text{std}(h_{t^*,\cdot,j}^{(l)})$, then $\tilde{h}_{i,j} = (h_{t^*,i,j}^{(l)} - \mu_j) / \sigma_j$ for $j = 1, \ldots, d$.


## Section 7: Linear Probe

At each layer $l$, fit logistic regression on standardized activations:

$$P(y_i = 1 \mid \tilde{h}_i^{(l)}) = \sigma\!\left(w^{(l)\top} \tilde{h}_i^{(l)} + b^{(l)}\right)$$

where $\sigma(z) = 1/(1+e^{-z})$, $w^{(l)} \in \mathbb{R}^d$, $b^{(l)} \in \mathbb{R}$. Training minimizes L2-regularized cross-entropy:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right] + \frac{1}{2C}\|w^{(l)}\|_2^2$$

with $C = 1.0$. Evaluation uses $K$-fold stratified CV with balanced accuracy:

$$\text{BalAcc} = \frac{1}{2}\left(\frac{TP}{TP+FN} + \frac{TN}{TN+FP}\right)$$

Peak layer: $l^* = \arg\max_l \text{BalAcc}^{(l)}$.


## Section 8: Permuted-Label Control

For $p = 1, \ldots, P$ (default $P = 10$), generate random permutation $\pi_p$ of indices $\{1, \ldots, N\}$. Fit the same probe on $(\tilde{h}_i^{(l)}, y_{\pi_p(i)})$ at each layer. Expected $\text{BalAcc} \approx 0.5$ under the null that activations carry no label-relevant information.


## Section 9: Metadata-Only Control

Construct metadata feature vector without any activations:

$$m_i = \left[T_i,\ T_{s,i},\ T_{u,i} - T_{s,i},\ T_{s,i},\ \text{onehot}(c_i),\ \text{onehot}(s_i),\ \text{onehot}(u_i),\ \mathbb{1}[d_i = b\_to\_a]\right] \in \mathbb{R}^{14}$$

where $T_i$ is total tokens, $T_{s,i}$ is system segment end, $T_{u,i} - T_{s,i}$ is user segment length. Note features 2 and 4 are linearly dependent ($T_{s,i}$ = system end = user start in the chat template).

Fit the same logistic regression on $m_i$. The gap $\Delta = \text{BalAcc}_{\text{probe}}^{(l^*)} - \text{BalAcc}_{\text{meta}}$ measures incremental information in activations beyond surface features.


## Section 10: Train/Test Split & Precedence Direction

Stratified train/test split (default 80/20). At $l^*$ (determined by CV in Section 7), fit probe on training set and extract the raw weight vector:

$$d = w^{(l^*)} \in \mathbb{R}^d$$

This is the candidate precedence direction in standardized activation space. When used for projection, it is normalized to $\hat{d} = d / \|d\|$.


## Section 11: Statistical Significance

Two tests on the gap $\Delta$ between probe and metadata control.

Bootstrap CI: for $K$-fold CV, compute paired fold-level differences $\delta_k = \text{BalAcc}_{\text{probe}}^{(k)} - \text{BalAcc}_{\text{meta}}^{(k)}$. Resample $(\delta_1, \ldots, \delta_K)$ with replacement $B = 1000$ times, take 2.5th and 97.5th percentiles. $p$-value $= \frac{1}{B}\sum_{b=1}^B \mathbb{1}[\bar{\delta}_b^* \leq 0]$.

Permutation test: fit probe on $y_{\pi_j(i)}$ for $j = 1, \ldots, 100$ random permutations, each evaluated by $K$-fold CV. $p = \frac{1}{100}\sum_{j=1}^{100}\mathbb{1}[\text{BalAcc}_{\pi_j} \geq \text{BalAcc}_{\text{real}}]$.


## Section 14: Neuron Importance

At each layer $l$, fit logistic regression on standardized activations and record the weight vector $w^{(l)} \in \mathbb{R}^d$. Global importance of neuron $j$:

$$I_j = \max_l |w_j^{(l)}|$$

Top-$K$ neurons are those with largest $I_j$.


## Section 15: Activation Difference Heatmap

For the top-$K$ neurons, compute per-layer mean activation difference between classes:

$$\Delta_{\text{act}}(j, l) = \frac{1}{N_1}\sum_{i:y_i=1} h_{t^*,i,j}^{(l)} - \frac{1}{N_0}\sum_{i:y_i=0} h_{t^*,i,j}^{(l)}$$

where $N_1 = \sum_i y_i$ and $N_0 = N - N_1$. Positive values indicate the neuron fires higher for system-following samples.


## Section 16: PCA

On standardized activations at $l^*$, compute PCA: $\tilde{H} = U \Sigma V^\top$, yielding principal components $v_1, \ldots, v_K$ (rows of $V^\top$) with explained variance ratios $\lambda_k / \sum_k \lambda_k$.

Alignment of precedence direction with each PC:

$$|\cos(d, v_k)| = \frac{|d^\top v_k|}{\|d\|}$$

since $\|v_k\| = 1$. If $|\cos(d, v_1)|$ is small, the conflict resolution signal is not a dominant mode of variation in the activation space.


## Section 17: Projection Gap

At each layer $l$, fit a probe, extract its normalized direction $\hat{d}^{(l)} = w^{(l)} / \|w^{(l)}\|$, and project all standardized activations:

$$z_i^{(l)} = \hat{d}^{(l)\top} \tilde{h}_i^{(l)}$$

Separation measured by Cohen's $d$:

$$d_{\text{Cohen}}^{(l)} = \frac{\bar{z}_{y=1}^{(l)} - \bar{z}_{y=0}^{(l)}}{s_{\text{pooled}}^{(l)}}, \quad s_{\text{pooled}}^{(l)} = \sqrt{\frac{s_{y=1}^2 + s_{y=0}^2}{2}}$$


## Section 18: Decision Boundary Visualization

Project standardized activations at $l^*$ onto $(v_1, v_2)$: $\bar{h}_i = (v_1^\top \tilde{h}_i,\ v_2^\top \tilde{h}_i) \in \mathbb{R}^2$. Fit 2D logistic regression:

$$P(y_i = 1 \mid \bar{h}_i) = \sigma(w_{2d}^\top \bar{h}_i + b_{2d})$$

Decision boundary is the set $\{\bar{h} : w_{2d}^\top \bar{h} + b_{2d} = 0\}$, a line in PC1-PC2 space. The gap between 2D accuracy and full-dimensional accuracy quantifies information loss from projecting to the top-2 variance directions.


## Section 19: Constraint-Specific Sub-Analysis

Restrict to samples with $c_i = c_{\text{target}}$. Repeat Sections 7, 8, 9 on this subset (with adjusted CV folds). Tests whether probe signal persists within a single constraint type, controlling for the constraint type confound.


## Section 20: Leave-One-Constraint-Type-Out

For each $c \in \mathcal{C}$: train on $\{i : c_i \neq c\}$, test on $\{i : c_i = c\}$. If $\text{BalAcc}_{\text{test}} \gg 0.5$, the probe direction generalizes across constraint types, supporting a constraint-invariant conflict resolution representation.


## Section 21: Per-Token Decision Transition

Fix $l = l^*$. For each token position $t = 0, 1, \ldots, T_{\min} - 1$ where $T_{\min} = \min_i T_i$:

$$\text{BalAcc}(t) = \text{CV-BalAcc}\!\left(\{(h_t^{(l^*)}, y_i)\}_{i=1}^N\right)$$

The onset point where $\text{BalAcc}(t)$ rises above chance indicates when conflict resolution information first appears in the residual stream. Comparing onset to segment boundaries $T_s$ and $T_u$ reveals whether the model encodes the decision during the system segment, user segment, or generation prompt.


## Section 22: Component Probing

The residual stream decomposes as:

$$h_t^{(l)} = h_t^{(l-1)} + a_t^{(l)} + m_t^{(l)}$$

where $a_t^{(l)}$ is the attention output and $m_t^{(l)}$ is the MLP output at layer $l$. Apply the same per-layer probe (Section 7) separately to $a_{t^*}^{(l)}$ and $m_{t^*}^{(l)}$. Comparing peak accuracies identifies which component contributes more to the conflict resolution representation.