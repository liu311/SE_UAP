# SE-UAP v2 Core Ablation Results

Recorded on 2026-09-08. Values are percentages.

| Dataset / metric | Fixed / $L_D$ off | Fixed / $L_D$ on | Distribution / $L_D$ off | Distribution / $L_D$ on |
|---|---:|---:|---:|---:|
| TIMIT FR | 99.46% | 96.35% | 91.49% | 86.22% |
| TIMIT FRD | 10.27% | 61.22% | 99.32% | 99.32% |
| LibriSpeech FR | 99.91% | 85.09% | 98.20% | 85.71% |
| LibriSpeech FRD | 3.49% | 70.59% | 85.30% | 97.38% |

## Ablation factors

- **Fixed:** one directly optimized base perturbation vector.
- **Distribution:** a perturbation distribution parameterized by `Generator1D(z)`.
- **$L_D$ off:** the detector loss is excluded from the training objective.
- **$L_D$ on:** training uses the joint objective $L_R + \alpha L_D$.

The corresponding experiment entry point is
[`train_se_uap_v2_core_ablation.py`](train_se_uap_v2_core_ablation.py).

These aggregate results were supplied after running the experiments. Raw logs and
checkpoints are not included in this record.
