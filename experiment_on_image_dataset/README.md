
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