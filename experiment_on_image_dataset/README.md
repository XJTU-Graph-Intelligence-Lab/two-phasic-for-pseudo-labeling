
# Non-Stationary Predictions May Be More Informative: Exploring Pseudo-Labels with a Two-Phase Pattern of Training Dynamics

This code repository is intended to reproduce all the experiments described in Sections 5.2 and 5.3 of the paper on the **image** dataset.

## Prerequisites

### Install Packages
To install the required packages, you can create a conda environment:

```sh
conda create --name img_env python=3.8
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```

### Prepare Datasets

The detailed instructions for downloading and processing are shown in [Dataset Download](./preprocess/). Please follow it to download datasets before running or developing algorithms.

## Reproduce Booster Experiments

Here is an example to train Two-phase base on pseudo-label in CIFAR-100 Dataset with 200 labels. Training other base algorithms (or other datasets with different label settings) can be specified by other [config files](./config/two_phase/):

- Step1: Train the base model
```python
python train.py --c config/two_phase/pseudolabel/cifar_base200.yaml
```
- Step2: Continue training with two-phase pseudo-label.
```python
python train.py --c config/two_phase/pseudolabel/cifar_tp200.yaml
```

## Reproduce Quality of 2-phase Labels Experiments
After saving the trained checkpoint in the booster experiments, you can reproduce all the experiments from Section 5.3 of the paper using the following command.

```python
python case_study.py --load_path [path to checkpoint] --tp_load_path [path to two-phase checkpoint] --data_dir [path to data]
```

## Additional Experimental Results in Appendix E.4
We conduct an additional experiment using the default settings from [1] for a direct comparison, as described in Appendix E.4. The results are summarized in the table below. In this experiment, we train the models for 200 epochs.

| Dataset    | Cifar100 | Cifar100 |
| ---------- | :------: | :------: |
| # Label    |   200    |   400    |
| Confidence |  66.84   |   75.38  |
| +2-phasic  |  68.24   |   77.95  |

You can reproduce this experiment by the following command.
- Step1: Train the base model
```python
python train.py --c config/two_phase/pseudolabel/cifar_base200.yaml
```
- Step2: Continue training with two-phase pseudo-label.
```python
python train.py --c config/two_phase/pseudolabel/cifar_tp200—default.yaml
```

[1] Wang, Y., Chen, H., Fan, Y., Sun, W., Tao, R., Hou, W., ... & Zhang, Y. (2022). Usb: A unified semi-supervised learning benchmark for classification. Advances in Neural Information Processing Systems, 35, 3938-3961.