# SE-UAP v2 Core Ablation Results

Result snapshot recorded on 2026-09-10. FR and FRD are percentages, SNR is
reported in dB, and PESQ is unitless.

| Dataset / metric | Fixed / $L_D$ off | Fixed / $L_D$ on | Distribution / $L_D$ off | Distribution / $L_D$ on |
|---|---:|---:|---:|---:|
| **TIMIT FR** | **99.46%** | 98.78% | 94.46% | 93.24% |
| **TIMIT FRD** | 3.78% | 97.97% | **99.19%** | 98.51% |
| TIMIT SNR | 13.60 | **13.99** | 10.27 | 10.27 |
| TIMIT PESQ | **2.9405** | 2.7538 | 2.6628 | 2.4175 |
| **LibriSpeech FR** | **99.93%** | 99.82% | 98.29% | 95.26% |
| **LibriSpeech FRD** | 4.40% | **97.88%** | 84.36% | 97.31% |
| LibriSpeech SNR | 11.78 | 11.79 | 11.79 | 11.79 |
| LibriSpeech PESQ | 2.1846 | **2.1945** | 2.0909 | 2.0941 |

## Ablation factors

- **Fixed:** one directly optimized base perturbation vector.
- **Distribution:** a perturbation distribution parameterized by `Generator1D(z)`.
- **$L_D$ off:** the detector loss is excluded from the training objective.
- **$L_D$ on:** training uses the joint objective $L_R + \alpha L_D$.

The corresponding experiment entry point is
[`train_se_uap_v2_core_ablation.py`](train_se_uap_v2_core_ablation.py).

These aggregate results were supplied after running the experiments and are stored
with the corresponding code revision for reproducible version lookup. Raw logs and
checkpoints are not included in this record. The earlier aggregate table remains
available in Git history at commit `cb72a44`.
