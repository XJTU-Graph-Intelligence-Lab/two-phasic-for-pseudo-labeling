# 2-phase pseudo-labeling algorithm
## Reproduce Booster Experiments
You can replicate our experimental results by following the steps outlined below:
- step 1 : Run baseline and record pseudo labels
```
python train_Modis_MS.py --dataset cora --train_size 3 # baseline is MoDis
```
- step 2 : Run 2-phase algorithm
```
python train_Two_Phase.py --name modis --dataset cora --train_size 3
```

## Reproduce Quality of 2-phase Labels Experiments
After saving the trained checkpoint in the booster experiments, you can reproduce all the experiments from Section 5.3 of the paper using the following command.
```python
python train_Two_Phase_exp2.py 
```