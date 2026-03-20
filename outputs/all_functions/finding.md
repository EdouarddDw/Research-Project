# my findings
## no overfiting model
the following is based on this model and this training:
| Item | Value |
|---|---|
| Input features | 10 |
| Hidden layers | 4 |
| Hidden units | 140, 100, 60, 20 |
| Architecture | Fully connected MLP |
| Main effect nets | True |
| Number of samples | 30,000 |
| Noise levels | 0.1, 0.5 |
| Epochs | 200 |
| Learning rate | 0.01 |
| Loss | MSE |
| Optimizer | Not specified in this file |
| Snapshots | Every 5 epochs |

**note that these models don't overfit, this is just notes about how interaction evolve as the model learns**

## what happends to none ground truth interactions durring training
[image](non_gt_emergence_delta_noise_comparison.png)

the image above shows the change in strength of pairwise interactions that are not in ground truth through the training process. we can notice a couple of things:

- Most of the non ground truth interaction strength appears early in training, then the rate of new spurious strength drops fast.

- At the start, both noise settings show a very large positive jump in non GT interaction strength. That means the model is assigning a lot of interaction mass to spurious pairs in the first epochs.

-  After that, the growth collapses quickly. By roughly epoch 25 to 40, the epoch to epoch increase is already much smaller.

- From there on, the curves flatten a lot:
    - for noise = 0.1, the median delta gets very close to zero and stays low
    - for noise = 0.5, the median delta remains a bit higher and more volatile across later epochs

we can also interpret thing about noice:
** Low noise, 0.1: **
the model creates most of its spurious pairwise interaction strength early, then largely stabilizes later training adds little extra non GT strength

**Higher noise, 0.5:**
the model also creates spurious strength early, but it keeps adding a modest amount for much longer training under more noise seems to sustain more ongoing spurious interaction growth

**What this suggests about behavior:**
noise seems to make the interaction structure less clean with more noise, the model continues to spread strength onto non ground truth pairs instead of settling as quickly

**in other words**
The emergence of spurious pairwise interactions is front loaded in training. Most non ground truth interaction strength is accumulated in early epochs, after which growth slows sharply. This slowdown is stroger for the low noise setting, while higher noise leads to more persistent and variable increases in non ground truth interaction strength throughout training.

## overfiting models


