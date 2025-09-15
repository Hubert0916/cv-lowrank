# Exploring Low-Rank and Depthwise Factorizations for Lightweight CNNs

<p align="center">
  <img width="690" height="390" alt="teaser" src="https://github.com/user-attachments/assets/0ada44bb-63ce-444f-8953-fffb713eca39" />
</p>

<p align="center">
  <b>Figure 1.</b> <i>Accuracy vs. Model Size on CIFAR-10.</i><br>
  Low-rank factorization achieves a favorable trade-off between accuracy and parameter count,
  with <b>rank ≈ 32</b> being a sweet spot that balances compression and performance.
</p>


## Motivation

**Deep convolutional neural networks (CNNs)** often contain redundant parameters. Many filters are **highly correlated**, which implies that the weight tensors have **low effective rank**.  

This research explores:
- **Low-rank factorization** (using SVD-inspired decompositions)
- **Depthwise-separable convolutions** (as in MobileNet-style architectures)

Our goal is to answer:
- *How much can we reduce the number of parameters without significantly hurting accuracy?*
- *Where is the "sweet spot" in rank selection for low-rank models?*


##  Model Architecture
I use a simple but expressive CNN (**TinyCNN**) as a testbed:
<p align="center"> 
  <img width="270" height="950" alt="model-arch" src="https://github.com/user-attachments/assets/b01d1471-9a68-4751-a42f-6cb600c61392" />
</p>


Variants:
- **Base**: Standard 3×3 convolutions  
- **LowRank**: Replace each 3×3 with `Conv(k×k, Cin→r) + Conv(1×1, r→Cout)`  
- **Dwsep**: Replace each 3×3 with depthwise + pointwise convolutions


## Mathematical Formulation

For a convolutional kernel  

$$
W \in \mathbb{R}^{C_\text{out} \times C_\text{in} \times k \times k}
$$

we reshape into  

$$
W^\flat \in \mathbb{R}^{C_\text{out} \times (C_\text{in}k^2)}
$$

and compute its **Singular Value Decomposition (SVD):**

$$
W^\flat = U \Sigma V^\top
$$

The best $rank-r$ approximation is:

$$
W^\flat_r = \sum_{i=1}^{r} \sigma_i u_i v_i^\top
$$

where $\sigma_i$ are singular values.  
Choosing $r$ to retain ≥95% of cumulative energy:

$$
\frac{\sum_{i=1}^{r}\sigma_i^2}{\sum_i \sigma_i^2} \ge 0.95
$$

provides a principled way to select the "sweet spot" rank.

Depthwise-separable convolution can be written as:

$$
W^\flat_{\text{dwsep}} = P D
$$

where $D$ is block-diagonal (per-channel spatial filter)  
and $P$ is a $1×1$ channel mixing matrix.  
Thus,

$$
\mathrm{rank}(W^\flat_{\text{dwsep}}) \le C_\text{in}
$$


## Experiments

**Dataset:** CIFAR-10  
**Optimizer:** SGD with momentum (0.9)  
**Scheduler:** Cosine annealing  
**Training:** 60 epochs for each configuration

I performed a **rank sweep** over \([8, 16, 24, 32, 48, 64]\) to study the effect of compression on accuracy.


## Results

| Model   | Rank | Parameters | Test Accuracy (%) |
|--------|------|-----------|------------------|
| **Base** | – | **261,834** | **87.7** |
| LowRank | 8  | 24,162  | 76.0 |
| LowRank | 16 | 45,882  | 83.2 |
| LowRank | 24 | 67,602  | 84.9 |
| LowRank | 32 | 89,322  | 85.6 |
| LowRank | 48 | 132,762 | 85.9 |
| LowRank | 64 | 176,202 | 86.0 |
| Dwsep   | –  | 33,637  | 83.7 |

**Observations:**
- **Base** achieves the highest accuracy but uses the most parameters.
- **LowRank with r = 32–64** retains >98% of Base accuracy with only 34–67% of parameters.
- **r = 8** compresses too aggressively, leading to underfitting.
- **Depthwise-separable (Dwsep)** is extremely parameter-efficient, close to LowRank r≈16–24.

## Conclusion

- **Low-rank factorization is a principled way to compress CNNs** while preserving most of the representational power.
- **Rank sweep analysis** reveals that most of the weight energy is captured by a small number of singular values.
- **Sweet spot:** `rank ≈ 32` for this architecture/dataset. This provides a **3× parameter reduction** with <2% accuracy drop.
