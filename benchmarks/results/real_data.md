# Real-data benchmark: plain flow vs prefit flow

prefit 0.4.0. Protocol: see the docstring of `benchmarks/real_data.py`. Optimism = how much better an estimate looks than the score on the final evaluation set E that neither flow saw (metric units; positive = optimistic). Mean ± standard deviation over seeds.

| dataset | metric | seeds | flow | CV optimism | own-test optimism | score on E | test rows seen in training | distinct training rows |
|---|---|---|---|---|---|---|---|---|
| wine_red | mae | 10 | plain | +0.0474 ± 0.0364 | +0.0674 ± 0.0431 | 0.5051 ± 0.0290 | 21.9% | 848 of 959 |
| wine_red | mae | 10 | prefit | -0.0121 ± 0.0323 | -0.0011 ± 0.0427 | 0.5109 ± 0.0260 | 0 (by construction) | 818 of 958 |
| breast_cancer | balanced_accuracy | 10 | plain | -0.0002 ± 0.0312 | +0.0207 ± 0.0334 | 0.9523 ± 0.0248 | 0.0% | 342 of 342 |
| breast_cancer | balanced_accuracy | 10 | prefit | +0.0090 ± 0.0301 | +0.0003 ± 0.0347 | 0.9471 ± 0.0245 | 0 (by construction) | 342 of 342 |
| telco | roc_auc | 5 | plain | -0.0104 ± 0.0069 | -0.0097 ± 0.0137 | 0.8381 ± 0.0061 | 0.0% | 4,225 of 4,225 |
| telco | roc_auc | 5 | prefit | -0.0130 ± 0.0087 | -0.0044 ± 0.0152 | 0.8388 ± 0.0084 | 0 (by construction) | 4,225 of 4,225 |
| california | mae | 3 | plain | -0.0023 ± 0.0078 | +0.0058 ± 0.0100 | 0.3162 ± 0.0062 | 0.0% | 12,384 of 12,384 |
| california | mae | 3 | prefit | -0.0002 ± 0.0095 | +0.0006 ± 0.0117 | 0.3127 ± 0.0111 | 0 (by construction) | 12,384 of 12,384 |
| adult | balanced_accuracy | 3 | plain | -0.0030 ± 0.0013 | -0.0007 ± 0.0082 | 0.7964 ± 0.0026 | 0.2% | 29,280 of 29,305 |
| adult | balanced_accuracy | 3 | prefit | -0.0069 ± 0.0033 | -0.0015 ± 0.0050 | 0.8466 ± 0.0013 | 0 (by construction) | 29,277 of 29,305 |

Reading it: an optimism near 0 means the reported estimate describes the model. A flow can have a better score on E and still report it dishonestly, or the reverse; both columns matter. prefit does not choose the model, so on data without hazards both flows should score the same on E. With exact duplicates, prefit keeps every copy on one side of the split: its training part then holds fewer distinct rows than a plain split of the same size (which puts copies on both sides and so scores some test rows on their own twins); the last column shows that cost.

Conditional or contextual steps prefit exported (seeds out of total):
- wine_red: none
- breast_cancer: none
- telco: none
- california: d.log_target 2/3
- adult: d.class_weight 3/3
